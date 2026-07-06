from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def is_diseased(row: dict[str, str]) -> bool:
    return int(row["crackle"]) > 0 or int(row["wheeze"]) > 0


def load_manifest_rows(manifest: Path) -> list[dict[str, str]]:
    with manifest.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def group_subject_rows(
    rows: list[dict[str, str]],
    diseased: bool,
) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        if int(row.get("augmented") or "0") != 0:
            continue
        if is_diseased(row) != diseased:
            continue
        grouped.setdefault(row["patient"], []).append(row)
    return grouped


def select_subjects(grouped: dict[str, list[dict[str, str]]], count: int) -> list[str]:
    def sort_key(patient: str) -> tuple[int, str]:
        return (int(patient), patient) if patient.isdigit() else (10**9, patient)

    return sorted(grouped, key=sort_key)[:count]


def load_feature(data_dir: Path, row: dict[str, str]) -> np.ndarray:
    feature = np.load(data_dir / row["feature_path"])
    if feature.ndim != 2:
        raise ValueError(f"Expected 2D feature for {row['feature_path']}, got {feature.shape}")
    return feature


def plot_subject_panel(
    data_dir: Path,
    patient: str,
    label: str,
    rows: list[dict[str, str]],
    output_path: Path,
    max_cycles: int,
) -> list[dict[str, str]]:
    selected_rows = rows[:max_cycles]
    features = [load_feature(data_dir, row) for row in selected_rows]
    vmin = min(float(feature.min()) for feature in features)
    vmax = max(float(feature.max()) for feature in features)

    cols = min(3, len(selected_rows))
    rows_count = math.ceil(len(selected_rows) / cols)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(rows_count, cols, figsize=(4.8 * cols, 3.6 * rows_count), squeeze=False)

    last_image = None
    for axis in axes.ravel():
        axis.axis("off")

    for index, (axis, row, feature) in enumerate(zip(axes.ravel(), selected_rows, features), start=1):
        axis.axis("on")
        last_image = axis.imshow(
            feature,
            origin="lower",
            aspect="auto",
            interpolation="bilinear",
            cmap="magma",
            vmin=vmin,
            vmax=vmax,
        )
        axis.set_title(
            f"{label.title()} feature {index}\n"
            f"{row['class_name']} | cycle {row['cycle_index']} | {float(row['duration_s']):.2f}s",
            fontsize=10,
        )
        axis.set_xlabel("Time frame")
        axis.set_ylabel("Mel bin")

    fig.suptitle(f"Patient {patient} - {label.title()} Extracted Features", fontsize=14)
    if last_image is not None:
        fig.colorbar(last_image, ax=axes.ravel().tolist(), label="Normalized log-mel energy")
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return selected_rows


def write_index(index_rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        fieldnames = [
            "label",
            "patient",
            "panel_path",
            "class_name",
            "source_file",
            "cycle_index",
            "start_time_s",
            "end_time_s",
            "duration_s",
            "feature_path",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(index_rows)


def plot_panels(args: argparse.Namespace) -> None:
    rows = load_manifest_rows(args.manifest)
    healthy_grouped = group_subject_rows(rows, diseased=False)
    diseased_grouped = group_subject_rows(rows, diseased=True)
    healthy_subjects = select_subjects(healthy_grouped, args.subjects)
    diseased_subjects = select_subjects(diseased_grouped, args.subjects)

    index_rows: list[dict[str, str]] = []
    for label, grouped, subjects in (
        ("healthy", healthy_grouped, healthy_subjects),
        ("diseased", diseased_grouped, diseased_subjects),
    ):
        for patient in subjects:
            output_path = args.output_dir / label / f"patient_{patient}_{label}_features.png"
            selected_rows = plot_subject_panel(
                args.data_dir,
                patient,
                label,
                grouped[patient],
                output_path,
                args.cycles_per_subject,
            )
            for row in selected_rows:
                index_rows.append(
                    {
                        "label": label,
                        "patient": patient,
                        "panel_path": str(output_path),
                        "class_name": row["class_name"],
                        "source_file": row["source_file"],
                        "cycle_index": row["cycle_index"],
                        "start_time_s": row["start_time_s"],
                        "end_time_s": row["end_time_s"],
                        "duration_s": row["duration_s"],
                        "feature_path": row["feature_path"],
                    }
                )
            print(f"Saved {label} subject panel: {output_path}")

    index_path = args.output_dir / "per_subject_feature_panel_index.csv"
    write_index(index_rows, index_path)
    print(f"Saved index: {index_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot extracted feature panels per healthy/diseased subject.")
    parser.add_argument("--data-dir", type=Path, default=Path("mel_dataset"))
    parser.add_argument("--manifest", type=Path, default=Path("mel_dataset") / "manifest.csv")
    parser.add_argument("--output-dir", type=Path, default=Path("plots") / "per_subject_feature_panels")
    parser.add_argument("--subjects", type=int, default=5)
    parser.add_argument("--cycles-per-subject", type=int, default=6)
    args = parser.parse_args()
    plot_panels(args)


if __name__ == "__main__":
    main()
