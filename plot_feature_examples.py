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


def select_examples(rows: list[dict[str, str]], diseased: bool, count: int) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    seen_patients: set[str] = set()
    originals = [row for row in rows if int(row.get("augmented") or "0") == 0 and is_diseased(row) == diseased]

    for row in originals:
        patient = row["patient"]
        if patient in seen_patients:
            continue
        selected.append(row)
        seen_patients.add(patient)
        if len(selected) == count:
            return selected

    for row in originals:
        if row in selected:
            continue
        selected.append(row)
        if len(selected) == count:
            return selected

    return selected


def plot_feature(row: dict[str, str], data_dir: Path, output_path: Path, label: str, index: int) -> None:
    feature_path = data_dir / row["feature_path"]
    feature = np.load(feature_path)
    if feature.ndim != 2:
        raise ValueError(f"Expected a 2D feature for {feature_path}, got shape {feature.shape}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    image = ax.imshow(
        feature,
        origin="lower",
        aspect="auto",
        interpolation="bilinear",
        cmap="magma",
    )
    ax.set_title(
        f"{label.title()} {index} | Patient {row['patient']} | "
        f"{row['class_name']} | {Path(row['feature_path']).name}"
    )
    ax.set_xlabel("Time frame")
    ax.set_ylabel("Mel bin")
    fig.colorbar(image, ax=ax, label="Normalized log-mel energy")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_index(rows: list[tuple[str, int, dict[str, str], Path]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "label",
                "example_index",
                "patient",
                "class_name",
                "source_file",
                "cycle_index",
                "feature_path",
                "image_path",
            ]
        )
        for label, index, row, image_path in rows:
            writer.writerow(
                [
                    label,
                    index,
                    row["patient"],
                    row["class_name"],
                    row["source_file"],
                    row["cycle_index"],
                    row["feature_path"],
                    image_path,
                ]
            )


def plot_examples(args: argparse.Namespace) -> None:
    rows = load_manifest_rows(args.manifest)
    healthy_rows = select_examples(rows, diseased=False, count=args.count)
    diseased_rows = select_examples(rows, diseased=True, count=args.count)
    if len(healthy_rows) < args.count:
        print(f"[warn] Only found {len(healthy_rows)} healthy examples.")
    if len(diseased_rows) < args.count:
        print(f"[warn] Only found {len(diseased_rows)} diseased examples.")

    index_rows: list[tuple[str, int, dict[str, str], Path]] = []
    for label, selected_rows in (("healthy", healthy_rows), ("diseased", diseased_rows)):
        for example_index, row in enumerate(selected_rows, start=1):
            image_path = args.output_dir / label / f"{label}_{example_index:02d}_patient_{row['patient']}.png"
            plot_feature(row, args.data_dir, image_path, label, example_index)
            index_rows.append((label, example_index, row, image_path))
            print(f"Saved {label} example {example_index}: {image_path}")

    write_index(index_rows, args.output_dir / "feature_plot_index.csv")
    print(f"Index saved to: {args.output_dir / 'feature_plot_index.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot healthy and diseased extracted mel feature examples.")
    parser.add_argument("--data-dir", type=Path, default=Path("mel_dataset"))
    parser.add_argument("--manifest", type=Path, default=Path("mel_dataset") / "manifest.csv")
    parser.add_argument("--output-dir", type=Path, default=Path("plots") / "feature_examples")
    parser.add_argument("--count", type=int, default=5)
    args = parser.parse_args()
    plot_examples(args)


if __name__ == "__main__":
    main()
