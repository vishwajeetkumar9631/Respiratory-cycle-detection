from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


DEFAULT_FEATURES = Path("icbhi_opensmile_stacking_dataset") / "segment_features.csv"
DEFAULT_MODEL_DIR = Path("models") / "icbhi_opensmile_gpu_loso"


class OpenSmileMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_features(path: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    frame = pd.read_csv(path)
    feature_columns = [column for column in frame.columns if column.startswith("smile_")]
    if not feature_columns:
        raise ValueError(f"No OpenSmile feature columns found in {path}")
    x = frame[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
    y = frame["label"].to_numpy(dtype=np.int64)
    return frame, x, y, feature_columns


def train_one_fold(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> np.ndarray:
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_set = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=False)
    model = OpenSmileMLP(x_train.shape[1], args.hidden_dim, args.dropout).to(device)

    if args.class_weight_mode == "balanced":
        counts = np.bincount(y_train, minlength=2).astype(np.float32)
        weights = counts.sum() / np.maximum(counts, 1.0)
        weights = weights / weights.mean()
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    model.train()
    for _ in range(args.epochs):
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    scores: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x_test), args.batch_size):
            batch = torch.from_numpy(x_test[start : start + args.batch_size]).to(device)
            probabilities = torch.softmax(model(batch), dim=1)[:, 1]
            scores.append(probabilities.cpu().numpy())
    return np.concatenate(scores)


def aggregate_recordings(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    score_column: str,
    threshold: float,
) -> pd.DataFrame:
    scored = frame[["source_file", "patient", "label"]].copy()
    scored["probability"] = probabilities
    rows: list[dict[str, object]] = []
    for source_file, group in scored.groupby("source_file", sort=True):
        values = group["probability"].to_numpy(dtype=np.float64)
        scores = {
            "prob_min": float(values.min()),
            "prob_max": float(values.max()),
            "prob_mean": float(values.mean()),
        }
        rows.append(
            {
                "source_file": source_file,
                "patient": str(group["patient"].iloc[0]),
                "label": int(group["label"].max()),
                "segments": int(len(group)),
                **scores,
                "score": scores[score_column],
                "prediction": int(scores[score_column] >= threshold),
            }
        )
    return pd.DataFrame(rows)


def choose_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    metric: str,
    min_class0_recall: float,
    min_class1_recall: float,
) -> tuple[float, float, float]:
    best_threshold = 0.5
    best_accuracy = -1.0
    best_balanced_accuracy = -1.0
    for threshold in np.linspace(0.01, 0.99, 197):
        predictions = (scores >= threshold).astype(np.int64)
        matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
        class0_recall = matrix[0, 0] / max(matrix[0].sum(), 1)
        class1_recall = matrix[1, 1] / max(matrix[1].sum(), 1)
        if class0_recall < min_class0_recall or class1_recall < min_class1_recall:
            continue
        accuracy = accuracy_score(y_true, predictions)
        balanced_accuracy = balanced_accuracy_score(y_true, predictions)
        candidate = accuracy if metric == "accuracy" else balanced_accuracy
        current = best_accuracy if metric == "accuracy" else best_balanced_accuracy
        if candidate > current:
            best_threshold = float(threshold)
            best_accuracy = float(accuracy)
            best_balanced_accuracy = float(balanced_accuracy)
    return best_threshold, best_accuracy, best_balanced_accuracy


def write_report(path: Path, y_true: np.ndarray, y_pred: np.ndarray, extra: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        file.write("OpenSmile GPU MLP LOSO evaluation\n\n")
        for key, value in extra.items():
            file.write(f"{key}: {value}\n")
        file.write("\n")
        file.write(f"accuracy: {accuracy_score(y_true, y_pred):.4f}\n")
        file.write(f"balanced_accuracy: {balanced_accuracy_score(y_true, y_pred):.4f}\n\n")
        file.write("classification report:\n")
        file.write(classification_report(y_true, y_pred, target_names=["healthy", "unhealthy"], zero_division=0))
        file.write("\nconfusion matrix:\n")
        file.write(f"{confusion_matrix(y_true, y_pred)}\n")


def write_graphs(
    output_prefix: Path,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> None:
    matrix = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], labels=["healthy", "unhealthy"])
    ax.set_yticks([0, 1], labels=["healthy", "unhealthy"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("OpenSmile GPU LOSO Confusion Matrix")
    for row in range(2):
        for col in range(2):
            color = "white" if matrix[row, col] > matrix.max() / 2 else "#111827"
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center", color=color, fontsize=13)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_prefix.with_name(f"{output_prefix.name}_confusion_matrix.png"), dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, name, color in [(0, "healthy", "#2563eb"), (1, "unhealthy", "#dc2626")]:
        ax.hist(scores[y_true == label], bins=20, alpha=0.62, label=name, color=color, edgecolor="white")
    ax.axvline(threshold, color="#111827", linestyle="--", linewidth=1, label=f"threshold={threshold:.3f}")
    ax.set_xlabel("Predicted unhealthy probability")
    ax.set_ylabel("Recordings")
    ax.set_xlim(0.0, 1.0)
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_prefix.with_name(f"{output_prefix.name}_probabilities.png"), dpi=160)
    plt.close(fig)


def run_loso(args: argparse.Namespace) -> None:
    frame, x, y, feature_columns = load_features(args.features)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    groups = frame["patient"].astype(str).to_numpy()
    patients = np.array(sorted(np.unique(groups), key=lambda value: int(value) if value.isdigit() else value))
    probabilities = np.zeros(len(frame), dtype=np.float32)
    args.model_dir.mkdir(parents=True, exist_ok=True)

    for fold_index, patient in enumerate(patients, start=1):
        train_mask = groups != patient
        test_mask = groups == patient
        if len(np.unique(y[train_mask])) < 2:
            continue
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x[train_mask]).astype(np.float32)
        x_test = scaler.transform(x[test_mask]).astype(np.float32)
        probabilities[test_mask] = train_one_fold(
            x_train,
            y[train_mask],
            x_test,
            args,
            device,
            args.seed + fold_index,
        )
        if fold_index == 1 or fold_index % args.log_every == 0:
            print(f"[{fold_index}/{len(patients)}] patient {patient}", flush=True)

    segment_output = frame[["source_file", "patient", "label", "segment_index"]].copy()
    segment_output["probability"] = probabilities
    segment_output.to_csv(args.model_dir / "gpu_loso_segment_predictions.csv", index=False)

    score_column = f"prob_{args.recording_agg}"
    preliminary_recordings = aggregate_recordings(frame, probabilities, score_column, args.threshold)
    y_true = preliminary_recordings["label"].to_numpy(dtype=np.int64)
    scores = preliminary_recordings["score"].to_numpy(dtype=np.float64)
    threshold = args.threshold
    if args.auto_threshold:
        threshold, tuned_acc, tuned_bal_acc = choose_threshold(
            y_true,
            scores,
            args.threshold_metric,
            args.min_class0_recall,
            args.min_class1_recall,
        )
        print(
            f"Auto threshold={threshold:.3f} "
            f"accuracy={tuned_acc:.4f} balanced_accuracy={tuned_bal_acc:.4f}",
            flush=True,
        )
    recordings = aggregate_recordings(frame, probabilities, score_column, threshold)
    recordings.to_csv(args.model_dir / "gpu_loso_recording_predictions.csv", index=False)
    y_true = recordings["label"].to_numpy(dtype=np.int64)
    y_pred = recordings["prediction"].to_numpy(dtype=np.int64)
    scores = recordings["score"].to_numpy(dtype=np.float64)
    write_report(
        args.model_dir / "gpu_loso_evaluation.txt",
        y_true,
        y_pred,
        {
            "features": args.features,
            "segments": len(frame),
            "recordings": len(recordings),
            "subjects": len(patients),
            "feature_count": len(feature_columns),
            "device": str(device),
            "epochs_per_fold": args.epochs,
            "aggregation": f"{args.recording_agg} segment probability per recording",
            "threshold": f"{threshold:.3f}",
            "class_weight_mode": args.class_weight_mode,
        },
    )
    write_graphs(args.model_dir / "gpu_loso", y_true, y_pred, scores, threshold)
    print(classification_report(y_true, y_pred, target_names=["healthy", "unhealthy"], zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))
    print(f"recording_acc={accuracy_score(y_true, y_pred):.4f}")
    print(f"recording_balanced_acc={balanced_accuracy_score(y_true, y_pred):.4f}")
    print(f"Saved outputs to: {args.model_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="GPU PyTorch classifier for OpenSmile segment features.")
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--recording-agg", choices=["min", "max", "mean"], default="max")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--auto-threshold", action="store_true")
    parser.add_argument("--threshold-metric", choices=["accuracy", "balanced_accuracy"], default="accuracy")
    parser.add_argument("--min-class0-recall", type=float, default=0.0)
    parser.add_argument("--min-class1-recall", type=float, default=0.0)
    parser.add_argument("--class-weight-mode", choices=["balanced", "none"], default="balanced")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    run_loso(args)


if __name__ == "__main__":
    main()
