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
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


CLASS_NAMES = ["healthy", "diseased"]


@dataclass(frozen=True)
class ManifestRow:
    feature_path: Path
    patient: str
    label: int
    augmented: bool
    source_file: str = ""
    duration_s: float = 0.0
    feature_scope: str = "cycle"
    equipment: str = "unknown"


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
        feature = np.ascontiguousarray(np.load(self.root_dir / row.feature_path), dtype=np.float32)
        if feature.ndim == 2:
            feature = feature[np.newaxis, :, :]
        elif feature.ndim != 3:
            raise ValueError(f"Expected 2D or 3D feature tensor, got shape {feature.shape}")
        feature = torch.from_numpy(feature)
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


class PaperRespiratoryCNN(nn.Module):
    """CNN matching the layer sizes shown in the paper's architecture figure."""

    input_size = (113, 133)

    def __init__(self, num_classes: int = 2, dropout: float = 0.2, input_channels: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 24, kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=(4, 2), stride=(4, 2))
        self.conv2 = nn.Conv2d(24, 48, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=(4, 2), stride=(4, 2), ceil_mode=True)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.feature_dropout = nn.Dropout(dropout)
        self.multi_heads = nn.ModuleList(
            [nn.Conv2d(48, 6, kernel_size=5) for _ in range(3)]
        )
        self.relu = nn.ReLU(inplace=True)
        self.flatten_dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(936, 100)
        self.dense_dropout = nn.Dropout(dropout)
        self.output = nn.Linear(100, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != self.input_size:
            x = F.interpolate(x, size=self.input_size, mode="bilinear", align_corners=False)
        x = self.feature_dropout(self.leaky_relu(self.pool1(self.conv1(x))))
        x = self.feature_dropout(self.leaky_relu(self.pool2(self.conv2(x))))
        x = self.relu(torch.stack([head(x) for head in self.multi_heads], dim=1))
        x = self.flatten_dropout(torch.flatten(x, start_dim=1))
        x = self.dense_dropout(self.relu(self.dense(x)))
        return self.output(x)


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden_channels = max(channels // reduction, 8)
        self.layers = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.layers(x)


class ConvNeXtSpectrogramBlock(nn.Module):
    def __init__(self, channels: int, expansion: int = 4) -> None:
        super().__init__()
        hidden_channels = channels * expansion
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=7,
            padding=3,
            groups=channels,
        )
        self.norm = nn.GroupNorm(1, channels)
        self.expand = nn.Conv2d(channels, hidden_channels, kernel_size=1)
        self.project = nn.Conv2d(hidden_channels, channels, kernel_size=1)
        self.attention = ChannelAttention(channels)
        self.layer_scale = nn.Parameter(torch.full((1, channels, 1, 1), 1e-6))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.depthwise(x)
        x = self.norm(x)
        x = self.project(F.gelu(self.expand(x)))
        x = self.attention(x)
        return residual + self.layer_scale * x


class AdvancedSpectrogramClassifier(nn.Module):
    """ConvNeXt-style image classifier specialized for mel-spectrograms."""

    def __init__(self, num_classes: int = 2, dropout: float = 0.35, input_channels: int = 1) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.GroupNorm(1, 32),
        )
        self.features = nn.Sequential(
            ConvNeXtSpectrogramBlock(32),
            ConvNeXtSpectrogramBlock(32),
            self._downsample(32, 64),
            ConvNeXtSpectrogramBlock(64),
            ConvNeXtSpectrogramBlock(64),
            self._downsample(64, 128),
            ConvNeXtSpectrogramBlock(128),
            ConvNeXtSpectrogramBlock(128),
            ConvNeXtSpectrogramBlock(128),
            self._downsample(128, 192),
            ConvNeXtSpectrogramBlock(192),
            ConvNeXtSpectrogramBlock(192),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(384),
            nn.Linear(384, 192),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(192, num_classes),
        )

    @staticmethod
    def _downsample(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.GroupNorm(1, in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(self.stem(x))
        avg_features = torch.mean(x, dim=(2, 3))
        max_features = torch.amax(x, dim=(2, 3))
        return self.classifier(torch.cat([avg_features, max_features], dim=1))


class ResidualSpectrogramBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: tuple[int, int] = (1, 1)) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        if in_channels != out_channels or stride != (1, 1):
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.layers(x) + self.shortcut(x))


class NativeResidualLstmClassifier(nn.Module):
    """Residual CNN + temporal BiLSTM that preserves the original feature size."""

    def __init__(self, num_classes: int = 2, dropout: float = 0.35, input_channels: int = 1) -> None:
        super().__init__()
        self.frontend = nn.Sequential(
            ResidualSpectrogramBlock(input_channels, 32),
            ResidualSpectrogramBlock(32, 64, stride=(2, 1)),
            ResidualSpectrogramBlock(64, 96, stride=(2, 1)),
            ResidualSpectrogramBlock(96, 128, stride=(2, 1)),
        )
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=96,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )
        self.attention = nn.Sequential(
            nn.Linear(192, 96),
            nn.Tanh(),
            nn.Linear(96, 1),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(384),
            nn.Linear(384, 192),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(192, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.frontend(x)
        x = x.mean(dim=2).transpose(1, 2)
        sequence, _ = self.lstm(x)
        weights = torch.softmax(self.attention(sequence), dim=1)
        attended = torch.sum(sequence * weights, dim=1)
        pooled = torch.amax(sequence, dim=1)
        return self.classifier(torch.cat([attended, pooled], dim=1))


def create_model(
    architecture: str,
    num_classes: int = 2,
    dropout: float | None = None,
    input_channels: int = 1,
) -> nn.Module:
    if architecture == "paper":
        return PaperRespiratoryCNN(
            num_classes=num_classes,
            dropout=0.2 if dropout is None else dropout,
            input_channels=input_channels,
        )
    if architecture == "advanced":
        return AdvancedSpectrogramClassifier(
            num_classes=num_classes,
            dropout=0.35 if dropout is None else dropout,
            input_channels=input_channels,
        )
    if architecture == "native_residual_lstm":
        return NativeResidualLstmClassifier(
            num_classes=num_classes,
            dropout=0.35 if dropout is None else dropout,
            input_channels=input_channels,
        )
    raise ValueError(f"Unknown architecture: {architecture}")


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def infer_class_names(manifest_paths: list[Path]) -> list[str]:
    names_by_label: dict[int, str] = {}
    for manifest_path in manifest_paths:
        with manifest_path.open("r", encoding="utf-8", newline="") as file:
            for row in csv.DictReader(file):
                label_text = row.get("binary_label")
                class_name = row.get("binary_class_name")
                if label_text not in (None, "") and class_name:
                    names_by_label[int(label_text)] = class_name
    if set(names_by_label) == {0, 1}:
        return [names_by_label[0], names_by_label[1]]
    return ["healthy", "diseased"]


def load_manifest(manifest_path: Path) -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    with manifest_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row.get("binary_label") not in (None, ""):
                label = int(row["binary_label"])
            else:
                crackle = int(row["crackle"])
                wheeze = int(row["wheeze"])
                label = int(crackle > 0 or wheeze > 0)
            if label not in (0, 1):
                raise ValueError(f"Expected binary label 0 or 1, got {label}")
            rows.append(
                ManifestRow(
                    feature_path=Path(row["feature_path"]),
                    patient=row["patient"],
                    label=label,
                    augmented=bool(int(row.get("augmented") or "0")),
                    source_file=row.get("source_file", ""),
                    duration_s=float(row.get("duration_s") or 0.0),
                    feature_scope=row.get("feature_scope") or "cycle",
                    equipment=row.get("equipment") or row.get("device_prefix") or "unknown",
                )
            )
    return rows


def filter_cycle_rows(
    rows: list[ManifestRow],
    min_duration: float,
    max_duration: float | None,
) -> list[ManifestRow]:
    filtered = [
        row
        for row in rows
        if row.feature_scope != "cycle"
        or (
            row.duration_s >= min_duration
            and (max_duration is None or row.duration_s <= max_duration)
        )
    ]
    if not filtered:
        raise ValueError(
            f"Cycle duration filtering removed all rows. min={min_duration}, max={max_duration}"
        )
    return filtered


def infer_input_shape(rows: list[ManifestRow], data_dir: Path) -> tuple[int, int, int]:
    if not rows:
        raise ValueError("Cannot infer input shape from an empty manifest.")
    feature = np.load(data_dir / rows[0].feature_path)
    if feature.ndim == 2:
        return 1, int(feature.shape[0]), int(feature.shape[1])
    if feature.ndim == 3:
        return int(feature.shape[0]), int(feature.shape[1]), int(feature.shape[2])
    raise ValueError(f"Expected 2D or 3D feature tensor, got shape {feature.shape}")


def split_by_patient(
    rows: list[ManifestRow],
    test_size: float,
    seed: int,
    include_augmented_val: bool,
    split_candidates: int,
) -> tuple[list[ManifestRow], list[ManifestRow]]:
    original_rows = [row for row in rows if not row.augmented]
    split_rows = original_rows or rows
    patients = sorted({row.patient for row in split_rows})
    patient_labels = []
    labels_by_patient: dict[str, list[int]] = {}
    for patient in patients:
        labels = [row.label for row in split_rows if row.patient == patient]
        labels_by_patient[patient] = labels
        patient_labels.append(int(any(labels)))

    stratify = patient_labels if len(set(patient_labels)) == 2 else None
    full_positive_ratio = float(np.mean([row.label for row in split_rows]))
    equipment_names = sorted({row.equipment for row in split_rows})
    full_equipment_counts = {
        equipment: sum(row.equipment == equipment for row in split_rows)
        for equipment in equipment_names
    }
    full_equipment_ratios = {
        equipment: count / len(split_rows)
        for equipment, count in full_equipment_counts.items()
    }
    target_val_count = max(1, int(round(len(split_rows) * test_size)))
    best_train_patients: list[str] | None = None
    best_score = float("inf")
    candidates = max(split_candidates, 1)

    for offset in range(candidates):
        train_patients_candidate, val_patients_candidate = train_test_split(
            patients,
            test_size=test_size,
            random_state=seed + offset,
            stratify=stratify,
        )
        val_labels = [
            label
            for patient in val_patients_candidate
            for label in labels_by_patient[patient]
        ]
        if len(set(val_labels)) < 2:
            continue
        val_positive_ratio = float(np.mean(val_labels))
        ratio_error = abs(val_positive_ratio - full_positive_ratio)
        size_error = abs(len(val_labels) - target_val_count) / target_val_count
        val_patient_set = set(val_patients_candidate)
        val_rows_candidate = [row for row in split_rows if row.patient in val_patient_set]
        equipment_error = float(
            np.mean(
                [
                    abs(
                        sum(row.equipment == equipment for row in val_rows_candidate)
                        / len(val_rows_candidate)
                        - full_equipment_ratios[equipment]
                    )
                    for equipment in equipment_names
                ]
            )
        )
        score = ratio_error + 0.25 * size_error + 0.5 * equipment_error
        if score < best_score:
            best_train_patients = list(train_patients_candidate)
            best_score = score

    if best_train_patients is None:
        best_train_patients, _ = train_test_split(
            patients,
            test_size=test_size,
            random_state=seed,
            stratify=stratify,
        )
    train_patients = best_train_patients
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


def label_counts(rows: list[ManifestRow]) -> dict[str, int]:
    counts = np.bincount([row.label for row in rows], minlength=2)
    return {CLASS_NAMES[index]: int(counts[index]) for index in range(len(CLASS_NAMES))}


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
    grad_clip: float = 0.0,
) -> tuple[float, list[float], list[int]]:
    training = optimizer is not None
    model.train(training)
    loss_sum = 0.0
    sample_count = 0
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
                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

        batch_size = labels.size(0)
        loss_sum += float(loss.item()) * batch_size
        sample_count += batch_size
        probabilities = torch.softmax(logits.detach(), dim=1)
        diseased_scores.extend(probabilities[:, 1].cpu().tolist())
        targets.extend(labels.cpu().tolist())

    return loss_sum / max(sample_count, 1), diseased_scores, targets


def metric_value(name: str, loss: float, accuracy: float, balanced_accuracy: float) -> float:
    if name == "val-loss":
        return loss
    if name == "val-acc":
        return accuracy
    if name == "val-balanced-acc":
        return balanced_accuracy
    raise ValueError(f"Unknown save metric: {name}")


def metric_improved(name: str, current_score: float, best_score: float) -> bool:
    return current_score < best_score if name == "val-loss" else current_score > best_score


def metric_delta(name: str, current_score: float, best_score: float) -> float:
    return best_score - current_score if name == "val-loss" else current_score - best_score


def scores_to_metrics(
    targets: list[int],
    scores: list[float],
    threshold: float,
) -> tuple[float, float]:
    predictions = predictions_from_threshold(scores, threshold)
    return accuracy_score(targets, predictions), balanced_accuracy_score(targets, predictions)


def paper_classification_metrics(
    targets: list[int],
    predictions: list[int],
) -> tuple[float, float, float]:
    matrix = confusion_matrix(targets, predictions, labels=[0, 1])
    true_negative, false_positive, false_negative, true_positive = matrix.ravel()
    specificity = true_negative / max(true_negative + false_positive, 1)
    sensitivity = true_positive / max(true_positive + false_negative, 1)
    average_score = (specificity + sensitivity) / 2.0
    return float(specificity), float(sensitivity), float(average_score)


def grouped_scores_to_metrics(
    rows: list[ManifestRow],
    scores: list[float],
    threshold: float,
    group_by: str = "patient",
) -> tuple[float, float, float, float, int]:
    grouped_targets: dict[str, list[int]] = {}
    grouped_scores: dict[str, list[float]] = {}
    for row, score in zip(rows, scores):
        group = row.patient if group_by == "patient" else row.source_file or str(row.feature_path)
        grouped_targets.setdefault(group, []).append(row.label)
        grouped_scores.setdefault(group, []).append(float(score))

    targets = [int(any(grouped_targets[group])) for group in grouped_targets]
    predictions = [
        int(float(np.mean(grouped_scores[group])) >= threshold)
        for group in grouped_targets
    ]
    specificity, sensitivity, average_score = paper_classification_metrics(targets, predictions)
    return accuracy_score(targets, predictions), specificity, sensitivity, average_score, len(targets)


def default_artifact_path(output: Path, suffix: str) -> Path:
    return output.with_name(f"{output.stem}_{suffix}")


def write_history_csv(history: list[dict[str, float]], path: Path) -> None:
    if not history:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def plot_training_history(
    history: list[dict[str, float]],
    path: Path,
    save_metric: str = "val-balanced-acc",
) -> None:
    if not history:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in history]
    if save_metric == "val-loss":
        best_row = min(history, key=lambda row: row["val_loss"])
    elif save_metric == "val-acc":
        best_row = max(history, key=lambda row: row["val_acc"])
    else:
        best_row = max(history, key=lambda row: row["val_bal_acc"])
    for row in history:
        specificity = row.get("specificity", row["class0_recall"])
        sensitivity = row.get("sensitivity", 2.0 * row["val_bal_acc"] - specificity)
        row.setdefault("specificity", float(np.clip(specificity, 0.0, 1.0)))
        row.setdefault("sensitivity", float(np.clip(sensitivity, 0.0, 1.0)))
        row.setdefault("average_score", row["val_bal_acc"])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(17, 9), sharex=True)

    train_loss_ax, val_loss_ax, accuracy_ax, paper_metrics_ax, lr_ax, gap_ax = axes.flat
    train_loss_ax.plot(epochs, [row["train_loss"] for row in history], color="#1f77b4", label="train loss")
    train_loss_ax.axvline(best_row["epoch"], linestyle="--", color="#525252", linewidth=1, label="selected checkpoint")
    train_loss_ax.set_ylabel("Training loss")
    train_loss_ax.legend()
    train_loss_ax.grid(alpha=0.25)

    val_losses = [row["val_loss"] for row in history]
    val_loss_ax.plot(epochs, val_losses, color="#ff7f0e", marker=".", markersize=3, label="validation loss")
    val_loss_ax.axvline(best_row["epoch"], linestyle="--", color="#525252", linewidth=1, label="selected checkpoint")
    val_loss_ax.set_ylabel("Validation loss")
    val_loss_ax.set_ylim(min(val_losses) - 0.05 * max(np.ptp(val_losses), 0.01), max(val_losses) + 0.05 * max(np.ptp(val_losses), 0.01))
    val_loss_ax.legend()
    val_loss_ax.grid(alpha=0.25)

    accuracy_ax.plot(epochs, [row["train_orig_bal_acc"] for row in history], label="train original balanced acc")
    accuracy_ax.plot(epochs, [row["val_bal_acc"] for row in history], label="validation balanced acc")
    accuracy_ax.plot(epochs, [row["val_acc"] for row in history], label="validation acc", alpha=0.65)
    accuracy_ax.axvline(best_row["epoch"], linestyle="--", color="#525252", linewidth=1)
    accuracy_ax.set_ylabel("Accuracy")
    accuracy_ax.set_ylim(0.0, 1.0)
    accuracy_ax.legend()
    accuracy_ax.grid(alpha=0.25)

    paper_metrics_ax.plot(epochs, [row["specificity"] for row in history], label="specificity (Sp)")
    paper_metrics_ax.plot(epochs, [row["sensitivity"] for row in history], label="sensitivity (Se)")
    paper_metrics_ax.plot(epochs, [row["average_score"] for row in history], label="average score (AS)", linewidth=2)
    paper_metrics_ax.axvline(best_row["epoch"], linestyle="--", color="#525252", linewidth=1)
    paper_metrics_ax.set_xlabel("Epoch")
    paper_metrics_ax.set_ylabel("Paper metrics")
    paper_metrics_ax.set_ylim(0.0, 1.0)
    paper_metrics_ax.legend()
    paper_metrics_ax.grid(alpha=0.25)

    learning_rates = [row["lr"] if row["lr"] > 0.0 else np.nan for row in history]
    lr_ax.plot(epochs, learning_rates, color="#7c3aed", label="learning rate")
    lr_ax.set_xlabel("Epoch")
    lr_ax.set_ylabel("Learning rate")
    lr_ax.set_yscale("log")
    lr_ax.legend()
    lr_ax.grid(alpha=0.25)

    generalization_gap = [
        row["train_orig_bal_acc"] - row["val_bal_acc"]
        for row in history
    ]
    gap_ax.plot(epochs, generalization_gap, color="#dc2626", label="train original - validation")
    gap_ax.axhline(0.0, color="#525252", linewidth=1)
    gap_ax.axvline(best_row["epoch"], linestyle="--", color="#525252", linewidth=1)
    gap_ax.fill_between(epochs, 0.0, generalization_gap, color="#dc2626", alpha=0.15)
    gap_ax.set_xlabel("Epoch")
    gap_ax.set_ylabel("Balanced accuracy gap")
    gap_ax.legend()
    gap_ax.grid(alpha=0.25)

    fig.suptitle(
        f"Training diagnostics - selected by {save_metric} at epoch {int(best_row['epoch'])} "
        f"(Sp={best_row['specificity']:.3f}, Se={best_row['sensitivity']:.3f}, "
        f"AS={best_row['average_score']:.3f})"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_evaluation_report(
    path: Path,
    val_true: list[int],
    val_pred: list[int],
    best_epoch: int,
    best_score: float,
    save_metric: str,
    final_threshold: float,
    model_path: Path,
    patient_acc: float | None = None,
    patient_specificity: float | None = None,
    patient_sensitivity: float | None = None,
    patient_average_score: float | None = None,
    patient_count: int | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    report = classification_report(
        val_true,
        val_pred,
        labels=[0, 1],
        target_names=CLASS_NAMES,
        zero_division=0,
    )
    matrix = confusion_matrix(val_true, val_pred, labels=[0, 1])
    specificity, sensitivity, average_score = paper_classification_metrics(val_true, val_pred)
    with path.open("w", encoding="utf-8") as file:
        file.write("Validation classification report:\n")
        file.write(report)
        file.write("\nConfusion matrix:\n")
        file.write(f"{matrix}\n")
        file.write(f"Best epoch: {best_epoch}\n")
        file.write(f"Best {save_metric}: {best_score:.4f}\n")
        file.write(f"Best positive-class threshold: {final_threshold:.2f}\n")
        file.write(f"Specificity (Sp): {specificity:.4f}\n")
        file.write(f"Sensitivity (Se): {sensitivity:.4f}\n")
        file.write(f"Average Score (AS): {average_score:.4f}\n")
        if (
            patient_acc is not None
            and patient_specificity is not None
            and patient_sensitivity is not None
            and patient_average_score is not None
            and patient_count is not None
        ):
            file.write("\nPatient-level validation metrics:\n")
            file.write(f"patients: {patient_count}\n")
            file.write(f"patient_acc: {patient_acc:.4f}\n")
            file.write(f"patient_specificity: {patient_specificity:.4f}\n")
            file.write(f"patient_sensitivity: {patient_sensitivity:.4f}\n")
            file.write(f"patient_average_score: {patient_average_score:.4f}\n")
        file.write(f"Best model saved to: {model_path}\n")


def train(args: argparse.Namespace) -> dict[str, float]:
    global CLASS_NAMES
    manifest_paths = (
        [args.train_manifest, args.val_manifest]
        if args.train_manifest and args.val_manifest
        else [args.manifest]
    )
    CLASS_NAMES = infer_class_names(manifest_paths)

    if args.train_manifest and args.val_manifest:
        train_rows = load_manifest(args.train_manifest)
        val_rows = load_manifest(args.val_manifest)
        rows = train_rows + val_rows
    elif args.train_manifest or args.val_manifest:
        raise ValueError("Use both --train-manifest and --val-manifest, or neither.")
    else:
        rows = load_manifest(args.manifest)
    labels = [row.label for row in rows]
    if len(set(labels)) < 2:
        raise ValueError(
            "The manifest contains only one class. Regenerate mel_dataset after "
            "mapping original ICBHI annotation labels, then train again."
        )

    if not (args.train_manifest and args.val_manifest):
        train_rows, val_rows = split_by_patient(
            rows,
            args.val_size,
            args.seed,
            args.include_augmented_val,
            args.split_candidates,
        )
    before_train_count = len(train_rows)
    before_val_count = len(val_rows)
    train_rows = filter_cycle_rows(train_rows, args.min_cycle_duration, args.max_cycle_duration)
    val_rows = filter_cycle_rows(val_rows, args.min_cycle_duration, args.max_cycle_duration)
    if len(train_rows) != before_train_count or len(val_rows) != before_val_count:
        print(
            "Cycle duration filter: "
            f"train {before_train_count}->{len(train_rows)}, "
            f"val {before_val_count}->{len(val_rows)} "
            f"(min={args.min_cycle_duration:.3f}s, max={args.max_cycle_duration})"
        )
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    input_shape = infer_input_shape(train_rows, args.data_dir)
    model = create_model(
        args.architecture,
        num_classes=len(CLASS_NAMES),
        dropout=args.dropout,
        input_channels=input_shape[0],
    ).to(device)
    train_original_rows = [row for row in train_rows if not row.augmented]
    train_augmented = sum(row.augmented for row in train_rows)
    val_augmented = sum(row.augmented for row in val_rows)
    print(f"Device: {device}")
    print(f"Architecture: {args.architecture}")
    print(f"Input shape: {input_shape}")
    print(f"Dropout: {args.dropout if args.dropout is not None else 'architecture default'}")
    print(f"Train samples: {len(train_rows)} ({train_augmented} augmented)")
    print(f"Train label counts: {label_counts(train_rows)}")
    print(f"Validation samples: {len(val_rows)} ({val_augmented} augmented)")
    print(f"Validation label counts: {label_counts(val_rows)}")
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
    train_original_loader = DataLoader(
        MelDataset(train_original_rows or train_rows, args.data_dir),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    weights = class_weights(train_rows, device) if args.class_weight_mode == "balanced" else None
    train_criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=args.label_smoothing)
    # Validation loss should reflect generalization, independent of train-only
    # class weighting and label smoothing.
    val_criterion = nn.CrossEntropyLoss()
    optimizer_name = getattr(args, "optimizer", "adamw")
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.lr_factor,
            patience=args.lr_patience,
            min_lr=args.min_lr,
        )
    elif args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    else:
        scheduler = None

    best_score = float("inf") if args.save_metric == "val-loss" else -1.0
    best_threshold = 0.5
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, float]] = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    history_csv = args.history_csv or default_artifact_path(args.output, "history.csv")
    history_plot = args.history_plot or default_artifact_path(args.output, "history.png")
    eval_report = args.eval_report or default_artifact_path(args.output, "evaluation.txt")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_scores, train_true = run_epoch(
            model,
            train_loader,
            train_criterion,
            device,
            optimizer,
            args.grad_clip,
        )
        val_loss, val_scores, val_true = run_epoch(model, val_loader, val_criterion, device)
        if args.scheduler == "plateau":
            scheduler.step(val_loss)
        elif scheduler is not None:
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
        specificity, sensitivity, average_score = paper_classification_metrics(val_true, val_pred)
        _, train_original_scores, train_original_true = run_epoch(
            model,
            train_original_loader,
            val_criterion,
            device,
        )
        train_original_acc, train_original_bal_acc = scores_to_metrics(
            train_original_true,
            train_original_scores,
            threshold,
        )
        print(
            f"epoch {epoch:03d} | "
            f"train_acc={train_acc:.4f} val_acc={val_acc:.4f} | "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f}",
            flush=True,
        )
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(train_loss),
                "train_acc": float(train_acc),
                "train_orig_acc": float(train_original_acc),
                "train_orig_bal_acc": float(train_original_bal_acc),
                "val_loss": float(val_loss),
                "val_acc": float(val_acc),
                "val_bal_acc": float(val_balanced_acc),
                "specificity": specificity,
                "sensitivity": sensitivity,
                "average_score": average_score,
                "class0_recall": float(val_zero_recall),
                "threshold": float(threshold),
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
        )
        write_history_csv(history, history_csv)
        plot_training_history(history, history_plot, args.save_metric)

        current_score = metric_value(args.save_metric, val_loss, val_acc, val_balanced_acc)
        improvement = metric_delta(args.save_metric, current_score, best_score)
        if metric_improved(args.save_metric, current_score, best_score) and improvement > args.min_delta:
            best_score = current_score
            best_threshold = threshold
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "class_names": CLASS_NAMES,
                    "architecture": args.architecture,
                    "input_shape": list(input_shape),
                    "input_channels": input_shape[0],
                    "parameters": count_parameters(model),
                    "diseased_threshold": best_threshold,
                    "class_weight_mode": args.class_weight_mode,
                    "min_class_zero_recall": args.min_class_zero_recall,
                    "label_smoothing": args.label_smoothing,
                    "specaugment": args.specaugment,
                    "save_metric": args.save_metric,
                    "dropout": args.dropout,
                    "scheduler": args.scheduler,
                    "optimizer": optimizer_name,
                    "grad_clip": args.grad_clip,
                },
                args.output,
            )
        else:
            stale_epochs += 1

        if args.patience > 0 and stale_epochs >= args.patience:
            print(
                f"Early stopping at epoch {epoch:03d}: no {args.save_metric} "
                f"improvement > {args.min_delta:g} for {args.patience} epochs.",
                flush=True,
            )
            break

    checkpoint = torch.load(args.output, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    _, val_scores, val_true = run_epoch(model, val_loader, val_criterion, device)
    final_threshold = float(checkpoint.get("diseased_threshold", best_threshold))
    val_pred = predictions_from_threshold(val_scores, final_threshold)
    final_val_acc = accuracy_score(val_true, val_pred)
    specificity, sensitivity, average_score = paper_classification_metrics(val_true, val_pred)
    patient_acc, patient_specificity, patient_sensitivity, patient_average_score, patient_count = grouped_scores_to_metrics(
        val_rows,
        val_scores,
        final_threshold,
        group_by="patient",
    )
    print("Validation classification report:")
    print(
        classification_report(
            val_true,
            val_pred,
            labels=[0, 1],
            target_names=CLASS_NAMES,
            zero_division=0,
        )
    )
    print("Confusion matrix:")
    print(confusion_matrix(val_true, val_pred, labels=[0, 1]))
    print(f"Best epoch: {best_epoch}")
    print(f"Best {args.save_metric}: {best_score:.4f}")
    print(f"Best positive-class threshold: {final_threshold:.2f}")
    print(
        f"Paper metrics: specificity={specificity:.4f} "
        f"sensitivity={sensitivity:.4f} average_score={average_score:.4f}"
    )
    print(
        f"Patient-level validation: patients={patient_count} "
        f"acc={patient_acc:.4f} specificity={patient_specificity:.4f} "
        f"sensitivity={patient_sensitivity:.4f} average_score={patient_average_score:.4f}"
    )
    print(f"Best model saved to: {args.output}")
    write_evaluation_report(
        eval_report,
        val_true,
        val_pred,
        best_epoch,
        best_score,
        args.save_metric,
        final_threshold,
        args.output,
        patient_acc,
        patient_specificity,
        patient_sensitivity,
        patient_average_score,
        patient_count,
    )
    print(f"Training history CSV saved to: {history_csv}")
    print(f"Training graph saved to: {history_plot}")
    print(f"Evaluation report saved to: {eval_report}")
    return {
        "best_epoch": float(best_epoch),
        "best_score": float(best_score),
        "val_acc": float(final_val_acc),
        "val_balanced_acc": float(average_score),
        "specificity": float(specificity),
        "sensitivity": float(sensitivity),
        "average_score": float(average_score),
        "patient_acc": float(patient_acc),
        "patient_balanced_acc": float(patient_average_score),
        "patient_specificity": float(patient_specificity),
        "patient_sensitivity": float(patient_sensitivity),
        "patient_average_score": float(patient_average_score),
        "threshold": float(final_threshold),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a 2D CNN on mel-spectrogram cycles.")
    parser.add_argument("--data-dir", type=Path, default=Path("mel_dataset"))
    parser.add_argument("--manifest", type=Path, default=Path("mel_dataset") / "manifest.csv")
    parser.add_argument("--train-manifest", type=Path, default=None)
    parser.add_argument("--val-manifest", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("models") / "residual_cnn.pt")
    parser.add_argument("--history-csv", type=Path, default=None)
    parser.add_argument("--history-plot", type=Path, default=None)
    parser.add_argument("--eval-report", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--optimizer", choices=["adam", "adamw"], default="adamw")
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--label-smoothing", type=float, default=0.08)
    parser.add_argument(
        "--scheduler",
        choices=["none", "plateau", "cosine"],
        default="plateau",
        help="Learning-rate scheduler. Use none to keep a fixed learning rate.",
    )
    parser.add_argument("--lr-patience", type=int, default=3)
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
        help="Maximum gradient norm. Use 0 to disable clipping.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=None,
        help="Dropout probability. Defaults: 0.2 for paper and 0.35 for advanced/native_residual_lstm.",
    )
    parser.add_argument("--specaugment", action="store_true")
    parser.add_argument("--freq-mask", type=int, default=8)
    parser.add_argument("--time-mask", type=int, default=16)
    parser.add_argument(
        "--architecture",
        choices=["paper", "advanced", "native_residual_lstm"],
        default="paper",
    )
    parser.add_argument("--save-metric", choices=["val-acc", "val-balanced-acc", "val-loss"], default="val-balanced-acc")
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
    parser.add_argument(
        "--min-cycle-duration",
        type=float,
        default=0.35,
        help="Drop cycles shorter than this duration in seconds before training/evaluation.",
    )
    parser.add_argument(
        "--max-cycle-duration",
        type=float,
        default=6.0,
        help="Drop cycles longer than this duration in seconds. Use 0 to disable the upper bound.",
    )
    parser.add_argument(
        "--split-candidates",
        type=int,
        default=200,
        help="Number of patient-wise random splits to try before choosing the most class-balanced validation split.",
    )
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--include-augmented-val",
        action="store_true",
        help="Include augmented samples in validation. By default validation uses original cycles only.",
    )
    args = parser.parse_args()
    if args.max_cycle_duration is not None and args.max_cycle_duration <= 0:
        args.max_cycle_duration = None
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.deterministic:
        torch.use_deterministic_algorithms(True)
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = False
    train(args)


if __name__ == "__main__":
    main()
