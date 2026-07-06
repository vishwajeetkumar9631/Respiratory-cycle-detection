from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from extract_mel_dataset import resize_time_axis


OUTPUT_FIELDS = [
    "feature_path",
    "source_file",
    "patient",
    "dataset",
    "original_patient",
    "diagnosis",
    "class_name",
    "binary_label",
    "binary_class_name",
    "crackle",
    "wheeze",
    "augmented",
    "duration_s",
    "n_mels",
    "n_frames",
    "feature_scope",
    "split",
]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def select_subject_rows(
    rows: list[dict[str, str]],
    subject_count: int,
    samples_per_subject: int,
) -> list[dict[str, str]]:
    by_patient: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_patient.setdefault(row["patient"], []).append(row)

    eligible = [
        patient
        for patient, patient_rows in by_patient.items()
        if len(patient_rows) >= samples_per_subject
    ]
    eligible.sort(key=int)
    selected_patients = eligible[:subject_count]
    if len(selected_patients) < subject_count:
        raise ValueError(
            f"Only {len(selected_patients)} subjects have at least "
            f"{samples_per_subject} samples; requested {subject_count}"
        )

    selected: list[dict[str, str]] = []
    for patient in selected_patients:
        patient_rows = sorted(
            by_patient[patient],
            key=lambda row: (row.get("source_file", ""), row["feature_path"]),
        )
        selected.extend(patient_rows[:samples_per_subject])
    return selected


def copy_feature(
    source_path: Path,
    output_path: Path,
    n_mels: int,
    n_frames: int,
) -> None:
    feature = np.load(source_path)
    if feature.ndim != 2:
        raise ValueError(f"Expected 2D feature at {source_path}, got {feature.shape}")
    if feature.shape[0] != n_mels:
        raise ValueError(f"Expected {n_mels} mel bins at {source_path}, got {feature.shape}")
    feature = resize_time_axis(feature, n_frames).astype(np.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, feature)


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def prepare_dataset(
    icbhi_manifest: Path,
    kauh_manifest: Path,
    output_dir: Path,
    subjects_per_class: int,
    samples_per_subject: int,
    test_size: float,
    seed: int,
    n_mels: int,
    n_frames: int,
) -> None:
    icbhi_root = icbhi_manifest.parent
    kauh_root = kauh_manifest.parent
    features_dir = output_dir / "features"

    icbhi_candidates = [
        row
        for row in read_rows(icbhi_manifest)
        if row["class_name"] == "normal" and int(row.get("augmented") or "0") == 0
    ]
    kauh_candidates = [
        row
        for row in read_rows(kauh_manifest)
        if "heart failure" in row["diagnosis"].casefold()
    ]

    normal_rows = select_subject_rows(
        icbhi_candidates, subjects_per_class, samples_per_subject
    )
    hf_rows = select_subject_rows(
        kauh_candidates, subjects_per_class, samples_per_subject
    )

    combined: list[dict[str, str]] = []
    for dataset, selected_rows, root, label in (
        ("ICBHI", normal_rows, icbhi_root, 0),
        ("KAUH", hf_rows, kauh_root, 1),
    ):
        for index, row in enumerate(selected_rows, start=1):
            patient = f"{dataset}_{int(row['patient']):03d}"
            output_name = f"{patient}_{index:03d}.npy"
            output_path = features_dir / output_name
            copy_feature(root / row["feature_path"], output_path, n_mels, n_frames)

            diagnosis = "Normal" if label == 0 else row["diagnosis"]
            combined.append(
                {
                    "feature_path": str(output_path.relative_to(output_dir)),
                    "source_file": row["source_file"],
                    "patient": patient,
                    "dataset": dataset,
                    "original_patient": row["patient"],
                    "diagnosis": diagnosis,
                    "class_name": "normal" if label == 0 else "heart_failure",
                    "binary_label": str(label),
                    "binary_class_name": "healthy" if label == 0 else "heart_failure",
                    "crackle": str(label),
                    "wheeze": "0",
                    "augmented": "0",
                    "duration_s": row.get("duration_s") or row.get("duration_seconds") or "",
                    "n_mels": str(n_mels),
                    "n_frames": str(n_frames),
                    "feature_scope": "recording_resized",
                    "split": "",
                }
            )

    subjects_by_label: dict[int, list[str]] = {0: [], 1: []}
    for patient in sorted({row["patient"] for row in combined}):
        label = int(
            next(row["binary_label"] for row in combined if row["patient"] == patient)
        )
        subjects_by_label[label].append(patient)

    train_subjects: list[str] = []
    test_subjects: list[str] = []
    for label, class_subjects in subjects_by_label.items():
        class_train, class_test = train_test_split(
            class_subjects,
            test_size=test_size,
            random_state=seed + label,
        )
        train_subjects.extend(class_train)
        test_subjects.extend(class_test)
    train_set = set(train_subjects)
    test_set = set(test_subjects)

    train_rows = []
    test_rows = []
    for row in combined:
        output_row = dict(row)
        if row["patient"] in train_set:
            output_row["split"] = "train"
            train_rows.append(output_row)
        elif row["patient"] in test_set:
            output_row["split"] = "test"
            test_rows.append(output_row)
        else:
            raise RuntimeError(f"Subject missing from split: {row['patient']}")

    if output_dir.exists():
        stale_split = output_dir / "split"
        if stale_split.exists():
            shutil.rmtree(stale_split)
    write_manifest(output_dir / "manifest.csv", train_rows + test_rows)
    write_manifest(output_dir / "split" / "train_manifest.csv", train_rows)
    write_manifest(output_dir / "split" / "test_manifest.csv", test_rows)

    print(f"Dataset: {output_dir}")
    print(f"Samples: train={len(train_rows)}, test={len(test_rows)}")
    print(f"Subjects: train={len(train_set)}, test={len(test_set)}")
    print(f"Subject overlap: {len(train_set & test_set)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a balanced subject-independent normal-vs-heart-failure dataset."
    )
    parser.add_argument(
        "--icbhi-manifest", type=Path, default=Path("mel_dataset") / "manifest.csv"
    )
    parser.add_argument(
        "--kauh-manifest", type=Path, default=Path("kauh_mel_dataset") / "manifest.csv"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("hf_subject_independent_dataset")
    )
    parser.add_argument("--subjects-per-class", type=int, default=21)
    parser.add_argument("--samples-per-subject", type=int, default=3)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-mels", type=int, default=64)
    parser.add_argument("--n-frames", type=int, default=128)
    args = parser.parse_args()

    prepare_dataset(
        args.icbhi_manifest,
        args.kauh_manifest,
        args.output_dir,
        args.subjects_per_class,
        args.samples_per_subject,
        args.test_size,
        args.seed,
        args.n_mels,
        args.n_frames,
    )


if __name__ == "__main__":
    main()
