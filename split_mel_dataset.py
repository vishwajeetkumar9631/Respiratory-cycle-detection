from __future__ import annotations

import argparse
import csv
from pathlib import Path

from train_cnn_classifier import CLASS_NAMES, label_counts, load_manifest, split_by_patient


def binary_label(row: dict[str, str]) -> int:
    crackle = int(row["crackle"])
    wheeze = int(row["wheeze"])
    return int(crackle > 0 or wheeze > 0)


def read_manifest_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    return rows, fieldnames


def write_split_manifest(
    path: Path,
    rows: list[dict[str, str]],
    fieldnames: list[str],
    split_name: str,
) -> None:
    output_fields = list(fieldnames)
    for extra_field in ("binary_label", "binary_class_name", "split"):
        if extra_field not in output_fields:
            output_fields.append(extra_field)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=output_fields)
        writer.writeheader()
        for row in rows:
            label = binary_label(row)
            output_row = dict(row)
            output_row["binary_label"] = str(label)
            output_row["binary_class_name"] = CLASS_NAMES[label]
            output_row["split"] = split_name
            writer.writerow(output_row)


def select_rows(
    manifest_rows: list[dict[str, str]],
    selected_feature_paths: set[str],
) -> list[dict[str, str]]:
    return [row for row in manifest_rows if row["feature_path"] in selected_feature_paths]


def split_dataset(args: argparse.Namespace) -> None:
    manifest_rows, fieldnames = read_manifest_rows(args.manifest)
    rows = load_manifest(args.manifest)
    train_rows, val_rows = split_by_patient(
        rows,
        args.val_size,
        args.seed,
        args.include_augmented_val,
        args.split_candidates,
    )

    train_paths = {str(row.feature_path) for row in train_rows}
    val_paths = {str(row.feature_path) for row in val_rows}
    train_manifest_rows = select_rows(manifest_rows, train_paths)
    val_manifest_rows = select_rows(manifest_rows, val_paths)
    train_original_rows = [row for row in train_rows if not row.augmented]
    val_original_rows = [row for row in val_rows if not row.augmented]
    original_count = len(train_original_rows) + len(val_original_rows)

    train_path = args.output_dir / "train_manifest.csv"
    val_path = args.output_dir / "val_manifest.csv"
    write_split_manifest(train_path, train_manifest_rows, fieldnames, "train")
    write_split_manifest(val_path, val_manifest_rows, fieldnames, "val")

    print(f"Train manifest: {train_path}")
    print(f"Validation manifest: {val_path}")
    print(f"Train samples: {len(train_rows)}")
    print(f"Train label counts: {label_counts(train_rows)}")
    print(f"Validation samples: {len(val_rows)}")
    print(f"Validation label counts: {label_counts(val_rows)}")
    print(
        "Original-cycle split: "
        f"train={len(train_original_rows)} ({len(train_original_rows) / original_count:.2%}), "
        f"val={len(val_original_rows)} ({len(val_original_rows) / original_count:.2%})"
    )
    print(f"Validation size target: {args.val_size:.2%}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split a mel_dataset manifest into train/validation manifests."
    )
    parser.add_argument("--manifest", type=Path, default=Path("mel_dataset") / "manifest.csv")
    parser.add_argument("--output-dir", type=Path, default=Path("mel_dataset_split"))
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split-candidates",
        type=int,
        default=200,
        help="Number of patient-wise random splits to try before choosing the most class-balanced validation split.",
    )
    parser.add_argument(
        "--include-augmented-val",
        action="store_true",
        help="Include augmented rows in validation. By default validation uses original cycles only.",
    )
    args = parser.parse_args()
    split_dataset(args)


if __name__ == "__main__":
    main()
