from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np


def verify_dataset(data_dir: Path, expected_shape: tuple[int, int]) -> None:
    manifest_path = data_dir / "manifest.csv"
    label_counts: Counter[str] = Counter()
    bad_files: list[str] = []
    rows = 0

    with manifest_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            rows += 1
            diseased = int(row["crackle"]) > 0 or int(row["wheeze"]) > 0
            label_counts["diseased" if diseased else "healthy"] += 1

            feature_path = data_dir / row["feature_path"]
            try:
                feature = np.load(feature_path)
                if feature.shape != expected_shape:
                    bad_files.append(f"{feature_path}: shape {feature.shape}")
            except Exception as exc:
                bad_files.append(f"{feature_path}: {type(exc).__name__}: {exc}")

    print(f"manifest rows: {rows}")
    print(f"label counts: {dict(label_counts)}")
    print(f"bad feature files: {len(bad_files)}")
    for item in bad_files[:20]:
        print(item)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify mel dataset manifest and feature tensors.")
    parser.add_argument("--data-dir", type=Path, default=Path("mel_dataset"))
    parser.add_argument("--n-mels", type=int, default=64)
    parser.add_argument("--n-frames", type=int, default=128)
    args = parser.parse_args()
    verify_dataset(args.data_dir, (args.n_mels, args.n_frames))


if __name__ == "__main__":
    main()
