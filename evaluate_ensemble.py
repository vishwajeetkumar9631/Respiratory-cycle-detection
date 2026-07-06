from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader

import train_cnn_classifier as trainer


def load_model(checkpoint_path: Path, device: torch.device) -> nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    class_names = checkpoint.get("class_names", trainer.CLASS_NAMES)
    architecture = checkpoint.get("architecture", "paper")
    dropout = checkpoint.get("dropout")
    input_channels = int(checkpoint.get("input_channels", checkpoint.get("input_shape", [1])[0]))
    model = trainer.create_model(
        architecture,
        num_classes=len(class_names),
        dropout=dropout,
        input_channels=input_channels,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def model_scores(
    checkpoint_path: Path,
    data_dir: Path,
    rows: list[trainer.ManifestRow],
    batch_size: int,
    device: torch.device,
) -> list[float]:
    model = load_model(checkpoint_path, device)
    loader = DataLoader(
        trainer.MelDataset(rows, data_dir),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    scores: list[float] = []
    with torch.no_grad():
        for features, _ in loader:
            logits = model(features.to(device))
            probabilities = torch.softmax(logits, dim=1)
            scores.extend(probabilities[:, 1].cpu().tolist())
    return scores


def choose_best_threshold(targets: list[int], scores: list[float]) -> tuple[float, float]:
    best_threshold = 0.5
    best_score = -1.0
    for threshold in np.linspace(0.05, 0.95, 91):
        predictions = trainer.predictions_from_threshold(scores, float(threshold))
        score = balanced_accuracy_score(targets, predictions)
        if score > best_score:
            best_threshold = float(threshold)
            best_score = float(score)
    return best_threshold, best_score


def write_report(
    output: Path,
    models: list[Path],
    data_dirs: list[Path],
    val_manifest: Path,
    targets: list[int],
    scores: list[float],
    threshold: float,
) -> None:
    predictions = trainer.predictions_from_threshold(scores, threshold)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        file.write("Ensemble validation report\n\n")
        file.write(f"val_manifest: {val_manifest}\n")
        file.write(f"threshold: {threshold:.2f}\n")
        file.write("members:\n")
        for model_path, data_dir in zip(models, data_dirs):
            file.write(f"- model={model_path} data_dir={data_dir}\n")
        file.write("\n")
        file.write(f"val_acc: {accuracy_score(targets, predictions):.4f}\n")
        file.write(f"val_balanced_acc: {balanced_accuracy_score(targets, predictions):.4f}\n\n")
        file.write("classification report:\n")
        file.write(classification_report(targets, predictions, target_names=trainer.CLASS_NAMES, zero_division=0))
        file.write("\nconfusion matrix:\n")
        file.write(f"{confusion_matrix(targets, predictions)}\n")


def evaluate(args: argparse.Namespace) -> None:
    if len(args.model) != len(args.data_dir):
        raise ValueError("--model and --data-dir must be provided the same number of times.")
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    rows = trainer.load_manifest(args.val_manifest)
    targets = [row.label for row in rows]
    member_scores = [
        model_scores(model_path, data_dir, rows, args.batch_size, device)
        for model_path, data_dir in zip(args.model, args.data_dir)
    ]
    ensemble_scores = np.mean(np.asarray(member_scores, dtype=np.float64), axis=0).tolist()
    threshold = args.threshold
    if args.auto_threshold:
        threshold, best_score = choose_best_threshold(targets, ensemble_scores)
        print(f"Best validation threshold={threshold:.2f} balanced_acc={best_score:.4f}")
    predictions = trainer.predictions_from_threshold(ensemble_scores, threshold)
    print(classification_report(targets, predictions, target_names=trainer.CLASS_NAMES, zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(targets, predictions))
    print(f"val_acc={accuracy_score(targets, predictions):.4f}")
    print(f"val_balanced_acc={balanced_accuracy_score(targets, predictions):.4f}")
    write_report(args.output_report, args.model, args.data_dir, args.val_manifest, targets, ensemble_scores, threshold)
    print(f"Ensemble report saved to: {args.output_report}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Average predictions from multiple trained respiratory models.")
    parser.add_argument("--model", type=Path, action="append", required=True)
    parser.add_argument("--data-dir", type=Path, action="append", required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--auto-threshold", action="store_true")
    parser.add_argument("--output-report", type=Path, default=Path("models") / "ensemble_evaluation.txt")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
