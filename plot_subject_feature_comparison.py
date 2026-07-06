from __future__ import annotations

import argparse
import csv
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


def select_subject_pairs(rows: list[dict[str, str]], count: int) -> list[tuple[str, dict[str, str], dict[str, str]]]:
    originals = [row for row in rows if int(row.get("augmented") or "0") == 0]
    by_patient: dict[str, dict[str, list[dict[str, str]]]] = {}
    for row in originals:
        patient = row["patient"]
        group = "diseased" if is_diseased(row) else "healthy"
        by_patient.setdefault(patient, {"healthy": [], "diseased": []})[group].append(row)

    pairs: list[tuple[str, dict[str, str], dict[str, str]]] = []
    for patient in sorted(by_patient, key=lambda value: int(value) if value.isdigit() else value):
        healthy_rows = by_patient[patient]["healthy"]
        diseased_rows = by_patient[patient]["diseased"]
        if healthy_rows and diseased_rows:
            pairs.append((patient, healthy_rows[0], diseased_rows[0]))
        if len(pairs) == count:
            break
    return pairs


def load_feature(data_dir: Path, row: dict[str, str]) -> np.ndarray:
    feature = np.load(data_dir / row["feature_path"])
    if feature.ndim != 2:
        raise ValueError(f"Expected a 2D feature for {row['feature_path']}, got shape {feature.shape}")
    return feature


def short_label(row: dict[str, str]) -> str:
    return (
        f"{row['class_name']} | {row['source_file']}\n"
        f"cycle {row['cycle_index']} | {float(row['start_time_s']):.2f}-{float(row['end_time_s']):.2f}s"
    )


def plot_subject_pair(
    data_dir: Path,
    patient: str,
    healthy_row: dict[str, str],
    diseased_row: dict[str, str],
    output_path: Path,
) -> None:
    healthy_feature = load_feature(data_dir, healthy_row)
    diseased_feature = load_feature(data_dir, diseased_row)
    vmin = min(float(healthy_feature.min()), float(diseased_feature.min()))
    vmax = max(float(healthy_feature.max()), float(diseased_feature.max()))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    images = []
    for ax, feature, title in (
        (axes[0], healthy_feature, f"Patient {patient} healthy\n{short_label(healthy_row)}"),
        (axes[1], diseased_feature, f"Patient {patient} diseased\n{short_label(diseased_row)}"),
    ):
        image = ax.imshow(
            feature,
            origin="lower",
            aspect="auto",
            interpolation="bilinear",
            cmap="magma",
            vmin=vmin,
            vmax=vmax,
        )
        images.append(image)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Time frame")
    axes[0].set_ylabel("Mel bin")
    fig.colorbar(images[-1], ax=axes.ravel().tolist(), label="Normalized log-mel energy")
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_combined_grid(
    data_dir: Path,
    pairs: list[tuple[str, dict[str, str], dict[str, str]]],
    output_path: Path,
) -> None:
    if not pairs:
        return
    features = []
    for _, healthy_row, diseased_row in pairs:
        features.append(load_feature(data_dir, healthy_row))
        features.append(load_feature(data_dir, diseased_row))
    vmin = min(float(feature.min()) for feature in features)
    vmax = max(float(feature.max()) for feature in features)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(pairs), 2, figsize=(13, 3.2 * len(pairs)), sharex=True, sharey=True)
    if len(pairs) == 1:
        axes = np.array([axes])

    last_image = None
    for row_index, (patient, healthy_row, diseased_row) in enumerate(pairs):
        for col_index, (label, manifest_row) in enumerate((("Healthy", healthy_row), ("Diseased", diseased_row))):
            ax = axes[row_index, col_index]
            feature = load_feature(data_dir, manifest_row)
            last_image = ax.imshow(
                feature,
                origin="lower",
                aspect="auto",
                interpolation="bilinear",
                cmap="magma",
                vmin=vmin,
                vmax=vmax,
            )
            ax.set_title(f"Patient {patient} - {label}\n{manifest_row['class_name']}", fontsize=10)
            if row_index == len(pairs) - 1:
                ax.set_xlabel("Time frame")
            if col_index == 0:
                ax.set_ylabel("Mel bin")

    fig.colorbar(last_image, ax=axes.ravel().tolist(), label="Normalized log-mel energy")
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_index(
    pairs: list[tuple[str, dict[str, str], dict[str, str]]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "patient",
                "label",
                "class_name",
                "source_file",
                "cycle_index",
                "start_time_s",
                "end_time_s",
                "feature_path",
            ]
        )
        for patient, healthy_row, diseased_row in pairs:
            for label, row in (("healthy", healthy_row), ("diseased", diseased_row)):
                writer.writerow(
                    [
                        patient,
                        label,
                        row["class_name"],
                        row["source_file"],
                        row["cycle_index"],
                        row["start_time_s"],
                        row["end_time_s"],
                        row["feature_path"],
                    ]
                )


def plot_subject_comparisons(args: argparse.Namespace) -> None:
    rows = load_manifest_rows(args.manifest)
    pairs = select_subject_pairs(rows, args.count)
    if len(pairs) < args.count:
        print(f"[warn] Only found {len(pairs)} subjects with both healthy and diseased original cycles.")

    for index, (patient, healthy_row, diseased_row) in enumerate(pairs, start=1):
        output_path = args.output_dir / "subjects" / f"subject_{index:02d}_patient_{patient}_healthy_vs_diseased.png"
        plot_subject_pair(args.data_dir, patient, healthy_row, diseased_row, output_path)
        print(f"Saved subject comparison {index}: {output_path}")

    combined_path = args.output_dir / "healthy_vs_diseased_5_subjects_grid.png"
    index_path = args.output_dir / "subject_feature_comparison_index.csv"
    plot_combined_grid(args.data_dir, pairs, combined_path)
    write_index(pairs, index_path)
    print(f"Saved combined grid: {combined_path}")
    print(f"Saved index: {index_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot per-subject healthy vs diseased extracted feature comparisons."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("mel_dataset"))
    parser.add_argument("--manifest", type=Path, default=Path("mel_dataset") / "manifest.csv")
    parser.add_argument("--output-dir", type=Path, default=Path("plots") / "subject_feature_comparison")
    parser.add_argument("--count", type=int, default=5)
    args = parser.parse_args()
    plot_subject_comparisons(args)


if __name__ == "__main__":
    main()
