from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset


CLASS_NAMES = ["healthy", "diseased"]


@dataclass(frozen=True)
class ManifestRow:
    feature_path: Path
    patient: str
    label: int
    augmented: bool


class MelDataset(Dataset):
    def __init__(
        self,
        rows: list[ManifestRow],
        root_dir: Path,
        training: bool = False,
        specaugment: bool = False,
        freq_mask: int = 8,
        time_mask: int = 16,
    ) -> None:
        self.rows = rows
        self.root_dir = root_dir
        self.training = training
        self.specaugment = specaugment
        self.freq_mask = freq_mask
        self.time_mask = time_mask

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        feature = np.load(self.root_dir / row.feature_path).astype(np.float32)
        feature = torch.from_numpy(feature).unsqueeze(0)
        if self.training and self.specaugment:
            feature = apply_specaugment(feature, self.freq_mask, self.time_mask)
        label = torch.tensor(row.label, dtype=torch.long)
        return feature, label


def apply_specaugment(feature: torch.Tensor, freq_mask: int, time_mask: int) -> torch.Tensor:
    augmented = feature.clone()
    _, n_mels, n_frames = augmented.shape
    fill_value = float(augmented.mean())

    if freq_mask > 0 and n_mels > 1:
        width = int(torch.randint(0, min(freq_mask, n_mels) + 1, (1,)).item())
        if width > 0:
            start = int(torch.randint(0, n_mels - width + 1, (1,)).item())
            augmented[:, start : start + width, :] = fill_value

    if time_mask > 0 and n_frames > 1:
        width = int(torch.randint(0, min(time_mask, n_frames) + 1, (1,)).item())
        if width > 0:
            start = int(torch.randint(0, n_frames - width + 1, (1,)).item())
            augmented[:, :, start : start + width] = fill_value

    return augmented


class CompactRespiratoryCNN(nn.Module):
    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        self.frontend = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.head3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.head5 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.head7 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=7, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(96, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.35),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.frontend(x)
        x = torch.cat([self.head3(x), self.head5(x), self.head7(x)], dim=1)
        return self.classifier(x)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def load_manifest(manifest_path: Path) -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    with manifest_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            crackle = int(row["crackle"])
            wheeze = int(row["wheeze"])
            label = int(crackle > 0 or wheeze > 0)
            rows.append(
                ManifestRow(
                    feature_path=Path(row["feature_path"]),
                    patient=row["patient"],
                    label=label,
                    augmented=bool(int(row.get("augmented") or "0")),
                )
            )
    return rows


def split_by_patient(
    rows: list[ManifestRow],
    test_size: float,
    seed: int,
    include_augmented_val: bool,
) -> tuple[list[ManifestRow], list[ManifestRow]]:
    original_rows = [row for row in rows if not row.augmented]
    split_rows = original_rows or rows
    patients = sorted({row.patient for row in split_rows})
    patient_labels = []
    for patient in patients:
        labels = [row.label for row in split_rows if row.patient == patient]
        patient_labels.append(int(any(labels)))

    stratify = patient_labels if len(set(patient_labels)) == 2 else None
    train_patients, val_patients = train_test_split(
        patients,
        test_size=test_size,
        random_state=seed,
        stratify=stratify,
    )
    train_patient_set = set(train_patients)
    train_rows = [row for row in rows if row.patient in train_patient_set]
    val_rows = [
        row
        for row in rows
        if row.patient not in train_patient_set and (include_augmented_val or not row.augmented)
    ]
    return train_rows, val_rows


def class_weights(rows: list[ManifestRow], device: torch.device) -> torch.Tensor:
    counts = np.bincount([row.label for row in rows], minlength=2).astype(np.float64)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def class_zero_recall(targets: list[int], predictions: list[int]) -> float:
    zero_count = sum(target == 0 for target in targets)
    if zero_count == 0:
        return 0.0
    true_zero = sum(target == 0 and prediction == 0 for target, prediction in zip(targets, predictions))
    return true_zero / zero_count


def predictions_from_threshold(scores: list[float], threshold: float) -> list[int]:
    return [int(score >= threshold) for score in scores]


def choose_threshold(
    targets: list[int],
    scores: list[float],
    min_class_zero_recall: float,
) -> tuple[float, list[int], float]:
    fallback_threshold = 0.5
    fallback_predictions = predictions_from_threshold(scores, fallback_threshold)
    fallback_score = balanced_accuracy_score(targets, fallback_predictions)
    best_threshold = fallback_threshold
    best_predictions = fallback_predictions
    best_score = -1.0
    thresholds = np.linspace(0.05, 0.95, 91)

    for threshold in thresholds:
        predictions = predictions_from_threshold(scores, float(threshold))
        zero_recall = class_zero_recall(targets, predictions)
        if zero_recall < min_class_zero_recall:
            continue
        score = balanced_accuracy_score(targets, predictions)
        if score > best_score:
            best_threshold = float(threshold)
            best_predictions = predictions
            best_score = score

    if best_score < 0.0:
        return fallback_threshold, fallback_predictions, fallback_score
    return best_threshold, best_predictions, best_score


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, list[float], list[int]]:
    training = optimizer is not None
    model.train(training)
    losses: list[float] = []
    diseased_scores: list[float] = []
    targets: list[int] = []

    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)
        if training:
            optimizer.zero_grad()
        with torch.set_grad_enabled(training):
            logits = model(features)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()

        losses.append(float(loss.item()))
        probabilities = torch.softmax(logits.detach(), dim=1)
        diseased_scores.extend(probabilities[:, 1].cpu().tolist())
        targets.extend(labels.cpu().tolist())

    return float(np.mean(losses)), diseased_scores, targets


def train(args: argparse.Namespace) -> None:
    rows = load_manifest(args.manifest)
    labels = [row.label for row in rows]
    if len(set(labels)) < 2:
        raise ValueError(
            "The manifest contains only one class. Regenerate mel_dataset after "
            "mapping original ICBHI annotation labels, then train again."
        )

    train_rows, val_rows = split_by_patient(rows, args.val_size, args.seed, args.include_augmented_val)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = CompactRespiratoryCNN(num_classes=len(CLASS_NAMES)).to(device)
    train_augmented = sum(row.augmented for row in train_rows)
    val_augmented = sum(row.augmented for row in val_rows)
    print(f"Device: {device}")
    print(f"Train samples: {len(train_rows)} ({train_augmented} augmented)")
    print(f"Validation samples: {len(val_rows)} ({val_augmented} augmented)")
    print(f"Trainable parameters: {count_parameters(model):,}")

    train_loader = DataLoader(
        MelDataset(
            train_rows,
            args.data_dir,
            training=True,
            specaugment=args.specaugment,
            freq_mask=args.freq_mask,
            time_mask=args.time_mask,
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        MelDataset(val_rows, args.data_dir),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    weights = class_weights(train_rows, device) if args.class_weight_mode == "balanced" else None
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    best_score = float("inf") if args.save_metric == "val-loss" else -1.0
    best_threshold = 0.5
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_scores, train_true = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_scores, val_true = run_epoch(model, val_loader, criterion, device)
        scheduler.step()
        if args.auto_threshold:
            threshold, val_pred, val_balanced_acc = choose_threshold(
                val_true,
                val_scores,
                args.min_class_zero_recall,
            )
        else:
            threshold = args.diseased_threshold
            val_pred = predictions_from_threshold(val_scores, threshold)
            val_balanced_acc = balanced_accuracy_score(val_true, val_pred)
        train_pred = predictions_from_threshold(train_scores, threshold)
        train_acc = accuracy_score(train_true, train_pred)
        val_acc = accuracy_score(val_true, val_pred)
        val_zero_recall = class_zero_recall(val_true, val_pred)
        print(
            f"epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
            f"val_bal_acc={val_balanced_acc:.4f} class0_recall={val_zero_recall:.4f} "
            f"threshold={threshold:.2f}",
            flush=True,
        )

        current_score = val_loss if args.save_metric == "val-loss" else val_acc
        improved = current_score < best_score if args.save_metric == "val-loss" else current_score > best_score
        if improved:
            best_score = current_score
            best_threshold = threshold
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "class_names": CLASS_NAMES,
                    "input_shape": [1, 64, 128],
                    "parameters": count_parameters(model),
                    "diseased_threshold": best_threshold,
                    "class_weight_mode": args.class_weight_mode,
                    "min_class_zero_recall": args.min_class_zero_recall,
                    "label_smoothing": args.label_smoothing,
                    "specaugment": args.specaugment,
                    "save_metric": args.save_metric,
                },
                args.output,
            )

    print("Validation classification report:")
    print(classification_report(val_true, val_pred, target_names=CLASS_NAMES, zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(val_true, val_pred))
    print(f"Best diseased threshold: {best_threshold:.2f}")
    print(f"Best model saved to: {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a compact 2D CNN on mel-spectrogram cycles.")
    parser.add_argument("--data-dir", type=Path, default=Path("mel_dataset"))
    parser.add_argument("--manifest", type=Path, default=Path("mel_dataset") / "manifest.csv")
    parser.add_argument("--output", type=Path, default=Path("models") / "compact_cnn.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.03)
    parser.add_argument("--specaugment", action="store_true")
    parser.add_argument("--freq-mask", type=int, default=8)
    parser.add_argument("--time-mask", type=int, default=16)
    parser.add_argument("--save-metric", choices=["val-acc", "val-loss"], default="val-acc")
    parser.add_argument("--class-weight-mode", choices=["none", "balanced"], default="balanced")
    parser.add_argument(
        "--diseased-threshold",
        type=float,
        default=0.5,
        help="Probability threshold for predicting diseased when auto thresholding is off.",
    )
    parser.add_argument(
        "--auto-threshold",
        action="store_true",
        help="Tune the diseased threshold on validation data to protect class 0 recall.",
    )
    parser.add_argument(
        "--min-class-zero-recall",
        type=float,
        default=0.85,
        help="Minimum validation recall target for class 0 when --auto-threshold is enabled.",
    )
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--include-augmented-val",
        action="store_true",
        help="Include augmented samples in validation. By default validation uses original cycles only.",
    )
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train(args)


if __name__ == "__main__":
    main()
