from __future__ import annotations

import argparse
import copy
import csv
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:  # pragma: no cover - older scikit-learn fallback
    StratifiedGroupKFold = None

import train_cnn_classifier as trainer
from split_mel_dataset import read_manifest_rows, select_rows, write_split_manifest


def patient_labels(rows: list[trainer.ManifestRow]) -> tuple[list[str], list[int]]:
    original_rows = [row for row in rows if not row.augmented]
    split_rows = original_rows or rows
    patients = sorted({row.patient for row in split_rows})
    labels = []
    for patient in patients:
        patient_rows = [row for row in split_rows if row.patient == patient]
        labels.append(int(any(row.label for row in patient_rows)))
    return patients, labels


def rows_for_patients(
    rows: list[trainer.ManifestRow],
    patient_set: set[str],
    include_augmented: bool,
) -> list[trainer.ManifestRow]:
    return [
        row
        for row in rows
        if row.patient in patient_set and (include_augmented or not row.augmented)
    ]


def build_fold_manifests(
    manifest: Path,
    output_dir: Path,
    folds: int,
    seed: int,
    include_augmented_val: bool,
) -> list[tuple[Path, Path, list[trainer.ManifestRow], list[trainer.ManifestRow]]]:
    manifest_rows, fieldnames = read_manifest_rows(manifest)
    rows = trainer.load_manifest(manifest)
    original_rows = [row for row in rows if not row.augmented] or rows
    patients, labels = patient_labels(rows)
    if folds > len(patients):
        raise ValueError(f"Requested {folds} folds but only found {len(patients)} patients.")
    if len(set(labels)) < 2:
        raise ValueError("K-fold cross-validation needs at least two patient-level classes.")

    fold_paths: list[tuple[Path, Path, list[trainer.ManifestRow], list[trainer.ManifestRow]]] = []
    if StratifiedGroupKFold is not None:
        split_iterator = StratifiedGroupKFold(
            n_splits=folds,
            shuffle=True,
            random_state=seed,
        ).split(
            np.zeros(len(original_rows)),
            np.array([row.label for row in original_rows]),
            np.array([row.patient for row in original_rows]),
        )
        patient_folds = [
            (
                {original_rows[index].patient for index in train_indices},
                {original_rows[index].patient for index in val_indices},
            )
            for train_indices, val_indices in split_iterator
        ]
    else:
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        patient_folds = [
            (
                {patients[index] for index in train_indices},
                {patients[index] for index in val_indices},
            )
            for train_indices, val_indices in splitter.split(np.array(patients), np.array(labels))
        ]

    for fold_index, (train_patients, val_patients) in enumerate(patient_folds, start=1):
        train_rows = rows_for_patients(rows, train_patients, include_augmented=True)
        val_rows = rows_for_patients(rows, val_patients, include_augmented=include_augmented_val)

        fold_dir = output_dir / f"fold_{fold_index:02d}"
        train_manifest = fold_dir / "train_manifest.csv"
        val_manifest = fold_dir / "val_manifest.csv"
        train_paths = {str(row.feature_path) for row in train_rows}
        val_paths = {str(row.feature_path) for row in val_rows}
        write_split_manifest(
            train_manifest,
            select_rows(manifest_rows, train_paths),
            fieldnames,
            f"fold_{fold_index:02d}_train",
        )
        write_split_manifest(
            val_manifest,
            select_rows(manifest_rows, val_paths),
            fieldnames,
            f"fold_{fold_index:02d}_val",
        )
        fold_paths.append((train_manifest, val_manifest, train_rows, val_rows))

    return fold_paths


def print_fold_summary(
    fold_index: int,
    folds: int,
    train_rows: list[trainer.ManifestRow],
    val_rows: list[trainer.ManifestRow],
) -> None:
    train_original = [row for row in train_rows if not row.augmented]
    val_original = [row for row in val_rows if not row.augmented]
    print(f"\nFold {fold_index:02d}/{folds}")
    print(f"Train samples: {len(train_rows)} ({len(train_rows) - len(train_original)} augmented)")
    print(f"Train label counts: {trainer.label_counts(train_rows)}")
    print(f"Validation samples: {len(val_rows)} ({len(val_rows) - len(val_original)} augmented)")
    print(f"Validation label counts: {trainer.label_counts(val_rows)}")


def train_kfold(args: argparse.Namespace) -> None:
    fold_paths = build_fold_manifests(
        args.manifest,
        args.output_dir,
        args.folds,
        args.seed,
        args.include_augmented_val,
    )
    if args.split_only:
        for fold_index, (train_manifest, val_manifest, train_rows, val_rows) in enumerate(fold_paths, start=1):
            print_fold_summary(fold_index, args.folds, train_rows, val_rows)
            print(f"Train manifest: {train_manifest}")
            print(f"Validation manifest: {val_manifest}")
        return

    metrics: list[dict[str, float]] = []
    for fold_index, (train_manifest, val_manifest, train_rows, val_rows) in enumerate(fold_paths, start=1):
        print_fold_summary(fold_index, args.folds, train_rows, val_rows)
        fold_args = copy.copy(args)
        fold_args.train_manifest = train_manifest
        fold_args.val_manifest = val_manifest
        fold_args.output = args.model_dir / f"fold_{fold_index:02d}.pt"
        fold_args.history_csv = args.model_dir / f"fold_{fold_index:02d}_history.csv"
        fold_args.history_plot = args.model_dir / f"fold_{fold_index:02d}_history.png"
        fold_args.eval_report = args.model_dir / f"fold_{fold_index:02d}_evaluation.txt"
        fold_args.include_augmented_val = args.include_augmented_val
        torch.manual_seed(args.seed + fold_index)
        np.random.seed(args.seed + fold_index)
        fold_metrics = trainer.train(fold_args)
        fold_metrics["fold"] = float(fold_index)
        metrics.append(fold_metrics)

    val_acc = np.array([metric["val_acc"] for metric in metrics], dtype=np.float64)
    val_balanced_acc = np.array([metric["val_balanced_acc"] for metric in metrics], dtype=np.float64)
    print("\n10-fold cross-validation summary")
    print(f"val_acc mean={val_acc.mean():.4f} std={val_acc.std(ddof=0):.4f}")
    print(f"val_bal_acc mean={val_balanced_acc.mean():.4f} std={val_balanced_acc.std(ddof=0):.4f}")
    print(f"Fold models saved to: {args.model_dir}")
    print(f"Fold histories/evaluation reports saved to: {args.model_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train respiratory CNN with patient-wise k-fold cross-validation.")
    parser.add_argument("--data-dir", type=Path, default=Path("mel_dataset"))
    parser.add_argument("--manifest", type=Path, default=Path("mel_dataset") / "manifest.csv")
    parser.add_argument("--output-dir", type=Path, default=Path("mel_dataset_kfold"))
    parser.add_argument("--model-dir", type=Path, default=Path("models") / "kfold")
    parser.add_argument("--history-csv", type=Path, default=None)
    parser.add_argument("--history-plot", type=Path, default=None)
    parser.add_argument("--eval-report", type=Path, default=None)
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--label-smoothing", type=float, default=0.08)
    parser.add_argument("--scheduler", choices=["plateau", "cosine"], default="plateau")
    parser.add_argument("--lr-patience", type=int, default=3)
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--specaugment", action="store_true")
    parser.add_argument("--freq-mask", type=int, default=8)
    parser.add_argument("--time-mask", type=int, default=16)
    parser.add_argument(
        "--architecture",
        choices=["paper", "advanced"],
        default="paper",
    )
    parser.add_argument("--save-metric", choices=["val-acc", "val-balanced-acc", "val-loss"], default="val-balanced-acc")
    parser.add_argument("--class-weight-mode", choices=["none", "balanced"], default="balanced")
    parser.add_argument("--diseased-threshold", type=float, default=0.5)
    parser.add_argument("--auto-threshold", action="store_true")
    parser.add_argument("--min-class-zero-recall", type=float, default=0.85)
    parser.add_argument("--min-cycle-duration", type=float, default=0.35)
    parser.add_argument("--max-cycle-duration", type=float, default=6.0)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--include-augmented-val", action="store_true")
    parser.add_argument("--split-only", action="store_true")
    parser.set_defaults(train_manifest=None, val_manifest=None, val_size=0.2, split_candidates=1)
    args = parser.parse_args()
    if args.max_cycle_duration is not None and args.max_cycle_duration <= 0:
        args.max_cycle_duration = None
    train_kfold(args)


if __name__ == "__main__":
    main()
