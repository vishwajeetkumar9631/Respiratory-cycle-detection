from __future__ import annotations

import argparse
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader

import train_cnn_classifier as trainer


def evaluate(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    checkpoint = torch.load(args.model, map_location=device)
    class_names = checkpoint.get("class_names", trainer.CLASS_NAMES)
    architecture = checkpoint.get("architecture", "paper")
    dropout = checkpoint.get("dropout")
    input_channels = int(checkpoint.get("input_channels", checkpoint.get("input_shape", [1])[0]))
    threshold = float(checkpoint.get("diseased_threshold", args.threshold))

    rows = trainer.load_manifest(args.val_manifest)
    rows = trainer.filter_cycle_rows(rows, args.min_cycle_duration, args.max_cycle_duration)
    model = trainer.create_model(
        architecture,
        num_classes=len(class_names),
        dropout=dropout,
        input_channels=input_channels,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    loader = DataLoader(
        trainer.MelDataset(rows, args.data_dir),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    criterion = nn.CrossEntropyLoss()
    val_loss, scores, targets = trainer.run_epoch(model, loader, criterion, device)
    predictions = trainer.predictions_from_threshold(scores, threshold)
    specificity, sensitivity, average_score = trainer.paper_classification_metrics(targets, predictions)
    patient_acc, patient_specificity, patient_sensitivity, patient_average_score, patient_count = trainer.grouped_scores_to_metrics(
        rows,
        scores,
        threshold,
        group_by="patient",
    )

    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    with args.output_report.open("w", encoding="utf-8") as file:
        file.write("Validation classification report:\n")
        file.write(classification_report(targets, predictions, labels=[0, 1], target_names=class_names, zero_division=0))
        file.write("\nConfusion matrix:\n")
        file.write(f"{confusion_matrix(targets, predictions, labels=[0, 1])}\n")
        file.write(f"Validation loss: {val_loss:.4f}\n")
        file.write(f"Validation accuracy: {accuracy_score(targets, predictions):.4f}\n")
        file.write(f"Validation balanced accuracy: {balanced_accuracy_score(targets, predictions):.4f}\n")
        file.write(f"Positive-class threshold: {threshold:.2f}\n")
        file.write(f"Specificity (Sp): {specificity:.4f}\n")
        file.write(f"Sensitivity (Se): {sensitivity:.4f}\n")
        file.write(f"Average Score (AS): {average_score:.4f}\n")
        file.write("\nPatient-level validation metrics:\n")
        file.write(f"patients: {patient_count}\n")
        file.write(f"patient_acc: {patient_acc:.4f}\n")
        file.write(f"patient_specificity: {patient_specificity:.4f}\n")
        file.write(f"patient_sensitivity: {patient_sensitivity:.4f}\n")
        file.write(f"patient_average_score: {patient_average_score:.4f}\n")
        file.write(f"Model: {args.model}\n")

    print(classification_report(targets, predictions, labels=[0, 1], target_names=class_names, zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(targets, predictions, labels=[0, 1]))
    print(f"val_loss={val_loss:.4f}")
    print(f"val_acc={accuracy_score(targets, predictions):.4f}")
    print(f"val_balanced_acc={balanced_accuracy_score(targets, predictions):.4f}")
    print(f"threshold={threshold:.2f}")
    print(f"report saved to: {args.output_report}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one saved CNN classifier checkpoint.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-cycle-duration", type=float, default=0.35)
    parser.add_argument("--max-cycle-duration", type=float, default=6.0)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    if args.max_cycle_duration is not None and args.max_cycle_duration <= 0:
        args.max_cycle_duration = None
    evaluate(args)


if __name__ == "__main__":
    main()
