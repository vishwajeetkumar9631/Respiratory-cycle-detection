from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from extract_kauh_mel_dataset import parse_kauh_name
from extract_mel_dataset import log_mel_spectrogram, preprocess_audio
from segment_icbhi_cycles import detect_cycles, read_wav_mono


DEFAULT_DATASET = Path(r"C:\Users\ankit\Downloads\jwyy9np4gv-3\Audio Files")
FIELDNAMES = [
    "feature_path",
    "source_file",
    "patient",
    "diagnosis",
    "device_prefix",
    "cycle_index",
    "start_time_s",
    "end_time_s",
    "duration_s",
    "binary_label",
    "binary_class_name",
    "crackle",
    "wheeze",
    "augmented",
    "vtlp_alpha",
    "sample_rate",
    "n_mels",
    "n_frames",
    "f_min",
    "f_max",
    "feature_scope",
    "split",
]


def classify_diagnosis(diagnosis: str) -> tuple[int, str] | None:
    normalized = diagnosis.casefold()
    if normalized == "n":
        return 0, "healthy"
    if "heart failure" in normalized:
        return 1, "heart_failure"
    return None


def group_recordings(dataset_dir: Path) -> dict[str, list[tuple[Path, dict[str, str]]]]:
    grouped: dict[str, list[tuple[Path, dict[str, str]]]] = {}
    for path in sorted(dataset_dir.glob("*.wav"), key=lambda item: item.name.casefold()):
        metadata = parse_kauh_name(path.name)
        if classify_diagnosis(metadata["diagnosis"]) is None:
            continue
        grouped.setdefault(metadata["patient"], []).append((path, metadata))
    return grouped


def split_subjects(
    grouped: dict[str, list[tuple[Path, dict[str, str]]]],
    test_per_class: int,
    train_per_class: int | None,
    seed: int,
) -> tuple[set[str], set[str]]:
    subjects_by_label: dict[int, list[str]] = {0: [], 1: []}
    for patient, recordings in grouped.items():
        label_info = classify_diagnosis(recordings[0][1]["diagnosis"])
        if label_info is not None:
            subjects_by_label[label_info[0]].append(patient)

    rng = np.random.default_rng(seed)
    train_subjects: set[str] = set()
    test_subjects: set[str] = set()
    for label in (0, 1):
        subjects = np.asarray(sorted(subjects_by_label[label], key=int))
        rng.shuffle(subjects)
        if train_per_class is not None:
            if subjects.size <= train_per_class:
                raise ValueError(
                    f"Class {label} has {subjects.size} subjects; requested "
                    f"{train_per_class} training subjects"
                )
            train_subjects.update(subjects[:train_per_class].tolist())
            test_subjects.update(subjects[train_per_class:].tolist())
            continue
        if subjects.size <= test_per_class:
            raise ValueError(
                f"Class {label} has {subjects.size} subjects; requested {test_per_class} test subjects"
            )
        test_subjects.update(subjects[:test_per_class].tolist())
        train_subjects.update(subjects[test_per_class:].tolist())
    return train_subjects, test_subjects


def write_subject_split(
    path: Path,
    grouped: dict[str, list[tuple[Path, dict[str, str]]]],
    train_subjects: set[str],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["patient", "diagnosis", "binary_label", "binary_class_name", "split"])
        for patient in sorted(grouped, key=int):
            metadata = grouped[patient][0][1]
            label, class_name = classify_diagnosis(metadata["diagnosis"]) or (-1, "excluded")
            writer.writerow(
                [
                    patient,
                    metadata["diagnosis"],
                    label,
                    class_name,
                    "train" if patient in train_subjects else "test",
                ]
            )


def extract_original_cycles(
    grouped: dict[str, list[tuple[Path, dict[str, str]]]],
    train_subjects: set[str],
    output_dir: Path,
    target_rate: int,
    segmentation_rate: int,
    n_mels: int,
    n_frames: int,
    n_fft: int,
    hop_length: int,
    f_min: float,
    f_max: float,
    min_cycle_s: float,
    max_cycle_s: float | None,
    overwrite: bool,
) -> tuple[list[dict[str, str]], dict[str, np.ndarray]]:
    feature_dir = output_dir / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    cycle_audio_by_key: dict[str, np.ndarray] = {}
    recordings = [
        (patient, path, metadata)
        for patient, patient_recordings in grouped.items()
        for path, metadata in patient_recordings
    ]

    for recording_index, (patient, path, metadata) in enumerate(recordings, start=1):
        input_rate, raw = read_wav_mono(path)
        cycles, _ = detect_cycles(
            raw,
            input_rate,
            target_rate=segmentation_rate,
            min_cycle_s=min_cycle_s,
        )
        processed = preprocess_audio(raw, input_rate, target_rate)
        label, class_name = classify_diagnosis(metadata["diagnosis"]) or (-1, "excluded")
        split = "train" if patient in train_subjects else "test"

        kept_cycles = 0
        for cycle_index, (start, end) in enumerate(cycles, start=1):
            duration_s = end - start
            if max_cycle_s is not None and duration_s > max_cycle_s:
                continue
            start_sample = max(int(round(start * target_rate)), 0)
            end_sample = min(int(round(end * target_rate)), processed.size)
            if end_sample <= start_sample:
                continue
            kept_cycles += 1
            cycle_audio = processed[start_sample:end_sample]
            key = f"{path.stem}_cycle_{cycle_index:03d}"
            feature_path = feature_dir / f"{key}.npy"
            if overwrite or not feature_path.exists():
                feature = log_mel_spectrogram(
                    cycle_audio,
                    target_rate,
                    n_mels,
                    n_frames,
                    n_fft,
                    hop_length,
                    f_min,
                    f_max,
                    smooth_freq_sigma=0.6,
                    smooth_time_sigma=0.8,
                )
                np.save(feature_path, feature)
            if split == "train":
                cycle_audio_by_key[key] = cycle_audio
            rows.append(
                {
                    "feature_path": str(feature_path.relative_to(output_dir)),
                    "source_file": path.name,
                    "patient": patient,
                    "diagnosis": metadata["diagnosis"],
                    "device_prefix": metadata["device_prefix"],
                    "cycle_index": str(cycle_index),
                    "start_time_s": f"{start:.3f}",
                    "end_time_s": f"{end:.3f}",
                    "duration_s": f"{duration_s:.3f}",
                    "binary_label": str(label),
                    "binary_class_name": class_name,
                    "crackle": str(label),
                    "wheeze": "0",
                    "augmented": "0",
                    "vtlp_alpha": "1.000000",
                    "sample_rate": str(target_rate),
                    "n_mels": str(n_mels),
                    "n_frames": str(n_frames),
                    "f_min": f"{f_min:.1f}",
                    "f_max": f"{f_max:.1f}",
                    "feature_scope": "cycle",
                    "split": split,
                }
            )
        print(
            f"[{recording_index}/{len(recordings)}] {path.name}: "
            f"{kept_cycles}/{len(cycles)} cycles kept ({split})",
            flush=True,
        )
    return rows, cycle_audio_by_key


def augment_training_to_targets(
    rows: list[dict[str, str]],
    cycle_audio_by_key: dict[str, np.ndarray],
    output_dir: Path,
    target_counts: dict[int, int],
    target_rate: int,
    n_mels: int,
    n_frames: int,
    n_fft: int,
    hop_length: int,
    f_min: float,
    f_max: float,
    vtlp_min: float,
    vtlp_max: float,
    seed: int,
    overwrite: bool,
) -> list[dict[str, str]]:
    feature_dir = output_dir / "features"
    rng = np.random.default_rng(seed)
    augmented_rows: list[dict[str, str]] = []

    for label in (0, 1):
        originals = [
            row
            for row in rows
            if row["split"] == "train" and int(row["binary_label"]) == label
        ]
        target = target_counts[label]
        if len(originals) > target:
            raise ValueError(
                f"Class {label} already has {len(originals)} original cycles, above target {target}"
            )
        needed = target - len(originals)
        originals_by_patient: dict[str, list[dict[str, str]]] = {}
        for row in originals:
            originals_by_patient.setdefault(row["patient"], []).append(row)
        patients = sorted(originals_by_patient, key=int)
        for augment_index in range(needed):
            patient = patients[augment_index % len(patients)]
            patient_rows = originals_by_patient[patient]
            source_row = patient_rows[int(rng.integers(0, len(patient_rows)))]
            source_key = Path(source_row["feature_path"]).stem
            alpha = float(rng.uniform(vtlp_min, vtlp_max))
            output_name = f"{source_key}_vtlp_{augment_index + 1:04d}.npy"
            output_path = feature_dir / output_name
            if overwrite or not output_path.exists():
                feature = log_mel_spectrogram(
                    cycle_audio_by_key[source_key],
                    target_rate,
                    n_mels,
                    n_frames,
                    n_fft,
                    hop_length,
                    f_min,
                    f_max,
                    vtlp_alpha=alpha,
                    smooth_freq_sigma=0.6,
                    smooth_time_sigma=0.8,
                )
                np.save(output_path, feature)
            augmented = dict(source_row)
            augmented["feature_path"] = str(output_path.relative_to(output_dir))
            augmented["augmented"] = "1"
            augmented["vtlp_alpha"] = f"{alpha:.6f}"
            augmented_rows.append(augmented)
    return augmented_rows


def augment_training_per_original(
    rows: list[dict[str, str]],
    cycle_audio_by_key: dict[str, np.ndarray],
    output_dir: Path,
    augmentations_per_original: int,
    target_rate: int,
    n_mels: int,
    n_frames: int,
    n_fft: int,
    hop_length: int,
    f_min: float,
    f_max: float,
    vtlp_min: float,
    vtlp_max: float,
    seed: int,
    overwrite: bool,
) -> list[dict[str, str]]:
    if augmentations_per_original <= 0:
        return []

    feature_dir = output_dir / "features"
    rng = np.random.default_rng(seed)
    augmented_rows: list[dict[str, str]] = []
    originals = [row for row in rows if row["split"] == "train"]
    for source_row in originals:
        source_key = Path(source_row["feature_path"]).stem
        for augment_index in range(1, augmentations_per_original + 1):
            alpha = float(rng.uniform(vtlp_min, vtlp_max))
            output_path = feature_dir / f"{source_key}_vtlp_{augment_index:02d}.npy"
            if overwrite or not output_path.exists():
                feature = log_mel_spectrogram(
                    cycle_audio_by_key[source_key],
                    target_rate,
                    n_mels,
                    n_frames,
                    n_fft,
                    hop_length,
                    f_min,
                    f_max,
                    vtlp_alpha=alpha,
                    smooth_freq_sigma=0.6,
                    smooth_time_sigma=0.8,
                )
                np.save(output_path, feature)
            augmented = dict(source_row)
            augmented["feature_path"] = str(output_path.relative_to(output_dir))
            augmented["augmented"] = "1"
            augmented["vtlp_alpha"] = f"{alpha:.6f}"
            augmented_rows.append(augmented)
    return augmented_rows


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def build_dataset(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    grouped = group_recordings(args.dataset_dir)
    train_subjects, test_subjects = split_subjects(
        grouped, args.test_subjects_per_class, args.train_subjects_per_class, args.seed
    )
    write_subject_split(
        args.output_dir / "subject_split.csv",
        grouped,
        train_subjects,
    )
    original_rows, cycle_audio_by_key = extract_original_cycles(
        grouped,
        train_subjects,
        args.output_dir,
        args.target_rate,
        args.segmentation_rate,
        args.n_mels,
        args.n_frames,
        args.n_fft,
        args.hop_length,
        args.f_min,
        args.f_max,
        args.min_cycle_s,
        args.max_cycle_s,
        args.overwrite,
    )
    if args.target_train_normal > 0 or args.target_train_hf > 0:
        if args.target_train_normal <= 0 or args.target_train_hf <= 0:
            raise ValueError("Use both target train counts, or set both to 0.")
        augmented_rows = augment_training_to_targets(
            original_rows,
            cycle_audio_by_key,
            args.output_dir,
            {0: args.target_train_normal, 1: args.target_train_hf},
            args.target_rate,
            args.n_mels,
            args.n_frames,
            args.n_fft,
            args.hop_length,
            args.f_min,
            args.f_max,
            args.vtlp_min,
            args.vtlp_max,
            args.seed,
            args.overwrite,
        )
    else:
        augmented_rows = augment_training_per_original(
            original_rows,
            cycle_audio_by_key,
            args.output_dir,
            args.augmentations_per_original,
            args.target_rate,
            args.n_mels,
            args.n_frames,
            args.n_fft,
            args.hop_length,
            args.f_min,
            args.f_max,
            args.vtlp_min,
            args.vtlp_max,
            args.seed,
            args.overwrite,
        )
    all_rows = original_rows + augmented_rows
    train_rows = [row for row in all_rows if row["split"] == "train"]
    test_rows = [row for row in original_rows if row["split"] == "test"]
    write_manifest(args.output_dir / "manifest.csv", train_rows + test_rows)
    write_manifest(args.output_dir / "train_manifest.csv", train_rows)
    write_manifest(args.output_dir / "test_manifest.csv", test_rows)

    print(f"Train subjects: {len(train_subjects)}")
    print(f"Test subjects: {len(test_subjects)}")
    print(f"Subject overlap: {len(train_subjects & test_subjects)}")
    print(f"Train features: {len(train_rows)}")
    print(f"Test features: {len(test_rows)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a KAUH-only study-style normal-vs-HF 2D mel dataset."
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=Path("kauh_hf_study_dataset"))
    parser.add_argument(
        "--test-subjects-per-class",
        type=int,
        default=7,
        help="Reserve this many subjects from each class for validation.",
    )
    parser.add_argument(
        "--train-subjects-per-class",
        type=int,
        default=None,
        help="Use exactly this many training subjects per class and hold out all remaining subjects.",
    )
    parser.add_argument(
        "--augmentations-per-original",
        type=int,
        default=0,
        help="Offline VTLP copies per original training cycle. Default 0 relies on SpecAugment.",
    )
    parser.add_argument(
        "--target-train-normal",
        type=int,
        default=0,
        help="Optional legacy fixed target. Set both target counts above 0 to enable.",
    )
    parser.add_argument("--target-train-hf", type=int, default=0)
    parser.add_argument("--target-rate", type=int, default=8000)
    parser.add_argument(
        "--segmentation-rate",
        type=int,
        default=8000,
        help="Analysis rate used only to detect cycle boundaries.",
    )
    parser.add_argument("--n-mels", type=int, default=64)
    parser.add_argument("--n-frames", type=int, default=128)
    parser.add_argument("--n-fft", type=int, default=512)
    parser.add_argument("--hop-length", type=int, default=128)
    parser.add_argument("--f-min", type=float, default=100.0)
    parser.add_argument("--f-max", type=float, default=2000.0)
    parser.add_argument("--min-cycle-s", type=float, default=0.7)
    parser.add_argument(
        "--max-cycle-s",
        type=float,
        default=6.0,
        help="Drop implausibly long detected cycles before feature extraction and augmentation.",
    )
    parser.add_argument("--vtlp-min", type=float, default=0.9)
    parser.add_argument("--vtlp-max", type=float, default=1.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.max_cycle_s is not None and args.max_cycle_s <= 0:
        args.max_cycle_s = None
    if not 0.0 <= args.f_min < args.f_max <= args.target_rate / 2.0:
        raise ValueError(
            "Expected 0 <= f-min < f-max <= target-rate / 2, got "
            f"{args.f_min}, {args.f_max}, target-rate={args.target_rate}"
        )
    build_dataset(args)


if __name__ == "__main__":
    main()
