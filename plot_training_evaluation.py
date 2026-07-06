from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import train_cnn_classifier as trainer


def read_history_csv(path: Path) -> list[dict[str, float]]:
    import csv

    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        return [{key: float(value) for key, value in row.items()} for row in reader]


def plot_epoch_graph(history_csv: Path, output: Path) -> None:
    history = read_history_csv(history_csv)
    if not history:
        raise ValueError(f"No history rows found in {history_csv}")

    output.parent.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in history]
    best_row = min(history, key=lambda row: row["val_loss"])
    generalization_gap = [
        row["train_orig_bal_acc"] - row["val_bal_acc"]
        for row in history
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    loss_ax, accuracy_ax, gap_ax, lr_ax = axes.flat
    loss_ax.plot(epochs, [row["train_loss"] for row in history], label="train loss", color="#2563eb")
    loss_ax.plot(epochs, [row["val_loss"] for row in history], label="validation loss", color="#dc2626")
    loss_ax.axvline(best_row["epoch"], linestyle="--", color="#525252", linewidth=1, label="lowest val loss")
    loss_ax.set_ylabel("Loss")
    loss_ax.grid(alpha=0.25)
    loss_ax.legend()

    accuracy_ax.plot(
        epochs,
        [row["train_orig_bal_acc"] for row in history],
        label="train original balanced accuracy",
        color="#16a34a",
    )
    accuracy_ax.plot(
        epochs,
        [row["val_bal_acc"] for row in history],
        label="validation balanced accuracy",
        color="#7c3aed",
    )
    accuracy_ax.plot(epochs, [row["val_acc"] for row in history], label="validation accuracy", color="#ea580c", alpha=0.65)
    accuracy_ax.set_ylabel("Accuracy")
    accuracy_ax.set_ylim(0.0, 1.0)
    accuracy_ax.grid(alpha=0.25)
    accuracy_ax.legend()

    gap_ax.plot(epochs, generalization_gap, color="#dc2626", label="train original - validation")
    gap_ax.axhline(0.0, color="#525252", linewidth=1)
    gap_ax.fill_between(epochs, 0.0, generalization_gap, color="#dc2626", alpha=0.15)
    gap_ax.set_xlabel("Epoch")
    gap_ax.set_ylabel("Balanced accuracy gap")
    gap_ax.grid(alpha=0.25)
    gap_ax.legend()

    learning_rates = [row["lr"] if row["lr"] > 0.0 else np.nan for row in history]
    lr_ax.plot(epochs, learning_rates, color="#7c3aed", label="learning rate")
    lr_ax.set_xlabel("Epoch")
    lr_ax.set_ylabel("Learning rate")
    lr_ax.set_yscale("log")
    lr_ax.grid(alpha=0.25)
    lr_ax.legend()

    fig.suptitle(
        f"Training diagnostics - lowest validation loss {best_row['val_loss']:.4f} "
        f"at epoch {int(best_row['epoch'])}"
    )
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def parse_confusion_matrix(text: str) -> np.ndarray:
    match = re.search(
        r"confusion matrix:\s*\n"
        r"\[\[\s*(\d+)\s+(\d+)\s*\]\s*"
        r"\[\s*(\d+)\s+(\d+)\s*\]\]",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        raise ValueError("Could not find a 2x2 confusion matrix in the evaluation report.")
    return np.array([int(value) for value in match.groups()], dtype=np.int64).reshape(2, 2)


def parse_metric(text: str, label: str) -> float | None:
    match = re.search(rf"{re.escape(label)}:\s*([0-9.]+)", text)
    return float(match.group(1)) if match else None


def plot_evaluation_graph(evaluation_report: Path, output: Path) -> None:
    text = evaluation_report.read_text(encoding="utf-8")
    matrix = parse_confusion_matrix(text)
    total = matrix.sum()
    accuracy = float(np.trace(matrix) / total) if total else 0.0
    best_balanced = parse_metric(text, "Best val-balanced-acc")
    threshold = parse_metric(text, "Best positive-class threshold")
    if threshold is None:
        threshold = parse_metric(text, "Best diseased threshold")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], labels=trainer.CLASS_NAMES)
    ax.set_yticks([0, 1], labels=trainer.CLASS_NAMES)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            percent = 100.0 * value / matrix[row].sum() if matrix[row].sum() else 0.0
            ax.text(col, row, f"{value}\n{percent:.1f}%", ha="center", va="center", color="#111827")

    title = f"Evaluation Confusion Matrix - accuracy {accuracy:.4f}"
    if best_balanced is not None:
        title += f", balanced {best_balanced:.4f}"
    if threshold is not None:
        title += f", threshold {threshold:.2f}"
    ax.set_title(title)
    fig.colorbar(image, ax=ax, label="Samples")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create epoch and evaluation graphs from model artifacts.")
    parser.add_argument("--history-csv", type=Path, required=True)
    parser.add_argument("--evaluation-report", type=Path, required=True)
    parser.add_argument("--epoch-output", type=Path, required=True)
    parser.add_argument("--evaluation-output", type=Path, required=True)
    args = parser.parse_args()
    plot_epoch_graph(args.history_csv, args.epoch_output)
    plot_evaluation_graph(args.evaluation_report, args.evaluation_output)
    print(f"Epoch graph saved to: {args.epoch_output}")
    print(f"Evaluation graph saved to: {args.evaluation_output}")


if __name__ == "__main__":
    main()
