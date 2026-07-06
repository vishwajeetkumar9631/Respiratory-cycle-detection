from __future__ import annotations

import argparse
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def binary_label(row: dict[str, str]) -> int:
    return int(int(row["crackle"]) > 0 or int(row["wheeze"]) > 0)


def class_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    counts = Counter(binary_label(row) for row in rows)
    return {"healthy": counts[0], "diseased": counts[1]}


def patient_sets(rows: list[dict[str, str]]) -> set[str]:
    return {row["patient"] for row in rows}


def duration_summary(rows: list[dict[str, str]]) -> str:
    durations = np.array([float(row.get("duration_s") or 0.0) for row in rows], dtype=np.float64)
    if durations.size == 0:
        return "none"
    short = int(np.sum(durations < 0.35))
    long = int(np.sum(durations > 6.0))
    return (
        f"min={durations.min():.3f}s median={np.median(durations):.3f}s "
        f"p95={np.percentile(durations, 95):.3f}s max={durations.max():.3f}s "
        f"short<0.35={short} long>6={long}"
    )


def equipment_counts(rows: list[dict[str, str]]) -> dict[str, dict[str, int]]:
    result: dict[str, Counter[int]] = defaultdict(Counter)
    for row in rows:
        equipment = row.get("equipment") or row.get("device_prefix") or "unknown"
        result[equipment][binary_label(row)] += 1
    return {
        equipment: {"healthy": counts[0], "diseased": counts[1]}
        for equipment, counts in sorted(result.items())
    }


def best_history(path: Path) -> dict[str, float] | None:
    rows = read_csv(path)
    if not rows:
        return None
    numeric_rows = [{key: float(value) for key, value in row.items()} for row in rows]
    return max(numeric_rows, key=lambda row: row["val_bal_acc"])


def parse_report(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    metrics: dict[str, float] = {}
    for key in ("patient_acc", "patient_balanced_acc"):
        match = re.search(rf"{key}:\s*([0-9.]+)", text)
        if match:
            metrics[key] = float(match.group(1))
    match = re.search(r"accuracy\s+([0-9.]+)\s+\d+", text)
    if match:
        metrics["report_accuracy"] = float(match.group(1))
    return metrics


def print_split_debug(train_rows: list[dict[str, str]], val_rows: list[dict[str, str]]) -> None:
    print("Split check")
    print(f"  train rows: {len(train_rows)} labels={class_counts(train_rows)}")
    print(f"  val rows:   {len(val_rows)} labels={class_counts(val_rows)}")
    overlap = patient_sets(train_rows) & patient_sets(val_rows)
    print(f"  train patients: {len(patient_sets(train_rows))}")
    print(f"  val patients:   {len(patient_sets(val_rows))}")
    print(f"  patient overlap: {len(overlap)}")
    print(f"  augmented rows in validation: {sum(int(row.get('augmented') or 0) for row in val_rows)}")
    print(f"  train durations: {duration_summary(train_rows)}")
    print(f"  val durations:   {duration_summary(val_rows)}")


def print_distribution_debug(rows: list[dict[str, str]]) -> None:
    print("\nDataset distribution")
    print(f"  all rows: {len(rows)} labels={class_counts(rows)}")
    print(f"  durations: {duration_summary(rows)}")
    print("  equipment label counts:")
    for equipment, counts in equipment_counts(rows).items():
        print(f"    {equipment}: {counts}")


def print_history_debug(history_path: Path, report_path: Path) -> None:
    print("\nTraining curve check")
    best = best_history(history_path)
    if not best:
        print(f"  missing history: {history_path}")
        return
    train_gap = best["train_acc"] - best["val_acc"]
    orig_gap = best["train_orig_acc"] - best["val_acc"]
    print(f"  best epoch: {int(best['epoch'])}")
    print(f"  train_acc={best['train_acc']:.4f} train_orig_acc={best['train_orig_acc']:.4f}")
    print(f"  val_acc={best['val_acc']:.4f} val_bal_acc={best['val_bal_acc']:.4f}")
    print(f"  overfit gap train-val={train_gap:.4f}; original-train-val={orig_gap:.4f}")
    report_metrics = parse_report(report_path)
    for key, value in report_metrics.items():
        print(f"  {key}={value:.4f}")


def print_kfold_debug(kfold_dir: Path) -> None:
    reports = sorted(kfold_dir.glob("fold_*_evaluation.txt"))
    if not reports:
        return
    values = []
    patient_values = []
    for report in reports:
        metrics = parse_report(report)
        if "report_accuracy" in metrics:
            values.append(metrics["report_accuracy"])
        if "patient_balanced_acc" in metrics:
            patient_values.append(metrics["patient_balanced_acc"])
    print("\nK-fold check")
    print(f"  completed folds: {len(reports)}")
    if values:
        print(f"  fold acc mean={np.mean(values):.4f} std={np.std(values):.4f} values={values}")
    if patient_values:
        print(
            f"  patient balanced acc mean={np.mean(patient_values):.4f} "
            f"std={np.std(patient_values):.4f} values={patient_values}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug respiratory ML training artifacts.")
    parser.add_argument("--manifest", type=Path, default=Path("mel_dataset") / "manifest.csv")
    parser.add_argument("--train-manifest", type=Path, default=Path("mel_dataset_split") / "train_manifest.csv")
    parser.add_argument("--val-manifest", type=Path, default=Path("mel_dataset_split") / "val_manifest.csv")
    parser.add_argument("--history", type=Path, default=Path("models") / "residual_cnn_filtered_history.csv")
    parser.add_argument("--report", type=Path, default=Path("models") / "residual_cnn_filtered_evaluation.txt")
    parser.add_argument("--kfold-model-dir", type=Path, default=Path("models") / "kfold_filtered")
    args = parser.parse_args()

    all_rows = read_csv(args.manifest)
    train_rows = read_csv(args.train_manifest)
    val_rows = read_csv(args.val_manifest)
    print_split_debug(train_rows, val_rows)
    print_distribution_debug(all_rows)
    print_history_debug(args.history, args.report)
    print_kfold_debug(args.kfold_model_dir)


if __name__ == "__main__":
    main()
