from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader

import train_cnn_classifier as trainer


def checkpoint_fold(path: Path) -> int | None:
    stem = path.stem
    if not stem.startswith("fold_"):
        return None
    try:
        return int(stem.split("_", maxsplit=1)[1])
    except ValueError:
        return None


def evaluate_checkpoint(
    checkpoint_path: Path,
    val_manifest: Path,
    data_dir: Path,
    batch_size: int,
    cpu: bool,
) -> tuple[dict[str, float | str], np.ndarray, str]:
    device = torch.device("cuda" if torch.cuda.is_available() and not cpu else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    class_names = checkpoint.get("class_names", trainer.CLASS_NAMES)
    architecture = checkpoint.get("architecture", "paper")
    dropout = checkpoint.get("dropout")
    input_channels = int(checkpoint.get("input_channels", checkpoint.get("input_shape", [1])[0]))
    threshold = float(checkpoint.get("diseased_threshold", 0.5))

    model = trainer.create_model(
        architecture,
        num_classes=len(class_names),
        dropout=dropout,
        input_channels=input_channels,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    rows = trainer.load_manifest(val_manifest)
    loader = DataLoader(
        trainer.MelDataset(rows, data_dir),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    criterion = nn.CrossEntropyLoss()
    val_loss, scores, targets = trainer.run_epoch(model, loader, criterion, device)
    predictions = trainer.predictions_from_threshold(scores, threshold)
    acc = accuracy_score(targets, predictions)
    bal_acc = balanced_accuracy_score(targets, predictions)
    matrix = confusion_matrix(targets, predictions)
    report = classification_report(targets, predictions, target_names=class_names, zero_division=0)

    fold = checkpoint_fold(checkpoint_path)
    metrics: dict[str, float | str] = {
        "fold": float(fold or 0),
        "checkpoint": str(checkpoint_path),
        "val_manifest": str(val_manifest),
        "samples": float(len(rows)),
        "val_loss": float(val_loss),
        "val_acc": float(acc),
        "val_balanced_acc": float(bal_acc),
        "threshold": float(threshold),
        "true_healthy": float(sum(target == 0 for target in targets)),
        "true_diseased": float(sum(target == 1 for target in targets)),
    }
    return metrics, matrix, report


def write_metrics_csv(rows: list[dict[str, float | str]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_evaluation(metrics: list[dict[str, float | str]], output: Path) -> None:
    if not metrics:
        return
    folds = [int(row["fold"]) for row in metrics]
    val_acc = [float(row["val_acc"]) for row in metrics]
    val_bal_acc = [float(row["val_balanced_acc"]) for row in metrics]
    val_loss = [float(row["val_loss"]) for row in metrics]

    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(folds, val_acc, marker="o", label="validation accuracy")
    axes[0].plot(folds, val_bal_acc, marker="o", label="validation balanced accuracy")
    axes[0].axhline(np.mean(val_acc), color="#2b6cb0", linestyle="--", linewidth=1, label="mean val accuracy")
    axes[0].axhline(np.mean(val_bal_acc), color="#c05621", linestyle="--", linewidth=1, label="mean balanced accuracy")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(folds, val_loss, marker="o", color="#4a5568", label="validation loss")
    axes[1].set_xlabel("Fold")
    axes[1].set_ylabel("Loss")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def write_report(
    metrics: list[dict[str, float | str]],
    matrices: list[np.ndarray],
    reports: list[str],
    output: Path,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        file.write("K-fold checkpoint evaluation report\n\n")
        if metrics:
            val_acc = np.array([float(row["val_acc"]) for row in metrics])
            val_bal_acc = np.array([float(row["val_balanced_acc"]) for row in metrics])
            file.write(f"Evaluated folds: {len(metrics)}\n")
            file.write(f"val_acc mean={val_acc.mean():.4f} std={val_acc.std(ddof=0):.4f}\n")
            file.write(f"val_bal_acc mean={val_bal_acc.mean():.4f} std={val_bal_acc.std(ddof=0):.4f}\n\n")

        for index, row in enumerate(metrics):
            file.write(f"Fold {int(row['fold']):02d}\n")
            file.write(f"checkpoint: {row['checkpoint']}\n")
            file.write(f"val_manifest: {row['val_manifest']}\n")
            file.write(f"samples: {int(float(row['samples']))}\n")
            file.write(f"val_loss: {float(row['val_loss']):.4f}\n")
            file.write(f"val_acc: {float(row['val_acc']):.4f}\n")
            file.write(f"val_balanced_acc: {float(row['val_balanced_acc']):.4f}\n")
            file.write(f"threshold: {float(row['threshold']):.2f}\n")
            file.write("classification report:\n")
            file.write(reports[index])
            file.write("\nconfusion matrix:\n")
            file.write(f"{matrices[index]}\n\n")


def evaluate_kfold(args: argparse.Namespace) -> None:
    checkpoint_paths = sorted(
        args.model_dir.glob("fold_*.pt"),
        key=lambda path: checkpoint_fold(path) or 0,
    )
    if not checkpoint_paths:
        raise FileNotFoundError(f"No fold_*.pt checkpoints found in {args.model_dir}")

    metrics: list[dict[str, float | str]] = []
    matrices: list[np.ndarray] = []
    reports: list[str] = []
    for checkpoint_path in checkpoint_paths:
        fold = checkpoint_fold(checkpoint_path)
        if fold is None:
            continue
        val_manifest = args.kfold_dir / f"fold_{fold:02d}" / "val_manifest.csv"
        if not val_manifest.exists():
            print(f"[skip] Missing validation manifest for {checkpoint_path}: {val_manifest}")
            continue
        print(f"Evaluating fold {fold:02d}: {checkpoint_path}", flush=True)
        fold_metrics, matrix, report = evaluate_checkpoint(
            checkpoint_path,
            val_manifest,
            args.data_dir,
            args.batch_size,
            args.cpu,
        )
        metrics.append(fold_metrics)
        matrices.append(matrix)
        reports.append(report)

    write_metrics_csv(metrics, args.output_csv)
    plot_evaluation(metrics, args.output_plot)
    write_report(metrics, matrices, reports, args.output_report)
    print(f"Evaluation metrics CSV saved to: {args.output_csv}")
    print(f"Evaluation graph saved to: {args.output_plot}")
    print(f"Evaluation report saved to: {args.output_report}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate saved k-fold checkpoints and create graph/report files.")
    parser.add_argument("--data-dir", type=Path, default=Path("mel_dataset"))
    parser.add_argument("--kfold-dir", type=Path, default=Path("mel_dataset_kfold"))
    parser.add_argument("--model-dir", type=Path, default=Path("models") / "kfold")
    parser.add_argument("--output-csv", type=Path, default=Path("models") / "kfold" / "evaluation_metrics.csv")
    parser.add_argument("--output-plot", type=Path, default=Path("models") / "kfold" / "evaluation_graph.png")
    parser.add_argument("--output-report", type=Path, default=Path("models") / "kfold" / "evaluation_report.txt")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    evaluate_kfold(args)


if __name__ == "__main__":
    main()
