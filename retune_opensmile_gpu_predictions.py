from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix

from train_opensmile_gpu_classifier import choose_threshold, write_graphs


def write_report(path: Path, y_true: np.ndarray, y_pred: np.ndarray, threshold: float, score_column: str) -> None:
    with path.open("w", encoding="utf-8") as file:
        file.write("Retuned OpenSmile GPU recording evaluation\n\n")
        file.write(f"score_column: {score_column}\n")
        file.write(f"threshold: {threshold:.3f}\n\n")
        file.write(f"accuracy: {accuracy_score(y_true, y_pred):.4f}\n")
        file.write(f"balanced_accuracy: {balanced_accuracy_score(y_true, y_pred):.4f}\n\n")
        file.write("classification report:\n")
        file.write(classification_report(y_true, y_pred, target_names=["healthy", "unhealthy"], zero_division=0))
        file.write("\nconfusion matrix:\n")
        file.write(f"{confusion_matrix(y_true, y_pred)}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Retune threshold/aggregation for saved OpenSmile GPU predictions.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--recording-agg", choices=["min", "max", "mean"], default="max")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--auto-threshold", action="store_true")
    parser.add_argument("--threshold-metric", choices=["accuracy", "balanced_accuracy"], default="accuracy")
    parser.add_argument("--min-class0-recall", type=float, default=0.0)
    parser.add_argument("--min-class1-recall", type=float, default=0.0)
    args = parser.parse_args()

    output_dir = args.output_dir or args.predictions.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.predictions)
    score_column = f"prob_{args.recording_agg}"
    y_true = frame["label"].to_numpy(dtype=np.int64)
    scores = frame[score_column].to_numpy(dtype=np.float64)
    threshold = args.threshold
    if args.auto_threshold:
        threshold, tuned_acc, tuned_bal_acc = choose_threshold(
            y_true,
            scores,
            args.threshold_metric,
            args.min_class0_recall,
            args.min_class1_recall,
        )
        print(f"Auto threshold={threshold:.3f} accuracy={tuned_acc:.4f} balanced_accuracy={tuned_bal_acc:.4f}")

    predictions = (scores >= threshold).astype(np.int64)
    retuned = frame.copy()
    retuned["score"] = scores
    retuned["prediction"] = predictions
    retuned.to_csv(output_dir / "gpu_loso_recording_predictions_retuned.csv", index=False)
    write_report(output_dir / "gpu_loso_evaluation_retuned.txt", y_true, predictions, threshold, score_column)
    write_graphs(output_dir / "gpu_loso_retuned", y_true, predictions, scores, threshold)
    print(classification_report(y_true, predictions, target_names=["healthy", "unhealthy"], zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(y_true, predictions))
    print(f"accuracy={accuracy_score(y_true, predictions):.4f}")
    print(f"balanced_accuracy={balanced_accuracy_score(y_true, predictions):.4f}")


if __name__ == "__main__":
    main()
