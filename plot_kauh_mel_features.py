from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_rows(manifest_path: Path) -> list[dict[str, str]]:
    with manifest_path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"No feature rows found in {manifest_path}")
    return rows


def feature_extent(row: dict[str, str]) -> list[float]:
    return [0.0, float(row["duration_seconds"]), float(row["f_min"]), float(row["f_max"])]


def plot_feature(row: dict[str, str], data_dir: Path, output_path: Path) -> None:
    feature = np.load(data_dir / row["feature_path"])
    if feature.ndim != 2:
        raise ValueError(f"Expected 2D feature, got {feature.shape}")

    fig, ax = plt.subplots(figsize=(9, 5))
    image = ax.imshow(
        feature,
        origin="lower",
        aspect="auto",
        interpolation="bilinear",
        cmap="magma",
        extent=feature_extent(row),
    )
    ax.set_title(
        f"{row['recording_id']} | {row['diagnosis']} | {row['sound_type']} | "
        f"{row['chest_location']} | age {row['age']} | {row['sex']}"
    )
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Frequency (Hz)")
    fig.colorbar(image, ax=ax, label="Normalized log-mel energy")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_overview(
    rows: list[dict[str, str]],
    data_dir: Path,
    output_path: Path,
    count_per_class: int,
) -> None:
    selected: list[dict[str, str]] = []
    for label in ("healthy", "diseased"):
        class_rows = [row for row in rows if row["binary_class_name"] == label]
        selected.extend(class_rows[:count_per_class])

    columns = count_per_class
    fig, axes = plt.subplots(2, columns, figsize=(4 * columns, 7), squeeze=False)
    for index, row in enumerate(selected):
        axis = axes[index // columns][index % columns]
        feature = np.load(data_dir / row["feature_path"])
        axis.imshow(
            feature,
            origin="lower",
            aspect="auto",
            interpolation="bilinear",
            cmap="magma",
            extent=feature_extent(row),
        )
        axis.set_title(f"{row['recording_id']} | {row['diagnosis']}", fontsize=9)
        axis.set_xlabel("Time (seconds)")
        axis.set_ylabel("Frequency (Hz)")

    fig.suptitle("KAUH extracted log-mel feature examples", fontsize=14)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_dataset(
    manifest_path: Path,
    data_dir: Path,
    output_dir: Path,
    overview_count: int,
) -> None:
    rows = load_rows(manifest_path)
    individual_dir = output_dir / "individual"

    for index, row in enumerate(rows, start=1):
        output_path = individual_dir / f"{Path(row['feature_path']).stem}.png"
        plot_feature(row, data_dir, output_path)
        print(f"[{index}/{len(rows)}] {output_path.name}", flush=True)

    overview_path = output_dir / "healthy_vs_diseased_overview.png"
    plot_overview(rows, data_dir, overview_path, overview_count)
    print(f"Individual plots: {individual_dir}")
    print(f"Overview plot: {overview_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot extracted KAUH log-mel features.")
    parser.add_argument("--data-dir", type=Path, default=Path("kauh_mel_dataset"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("kauh_mel_dataset") / "manifest.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots") / "kauh_mel_features",
    )
    parser.add_argument("--overview-count", type=int, default=5)
    args = parser.parse_args()

    plot_dataset(args.manifest, args.data_dir, args.output_dir, args.overview_count)


if __name__ == "__main__":
    main()
