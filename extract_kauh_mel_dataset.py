from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np

from extract_mel_dataset import log_mel_spectrogram
from segment_icbhi_cycles import read_wav_mono


def parse_kauh_name(filename: str) -> dict[str, str]:
    parts = Path(filename).stem.split(",", maxsplit=4)
    if len(parts) != 5:
        raise ValueError(f"Unexpected KAUH filename: {filename}")

    recording_and_diagnosis, sound_type, chest_location, age, sex = parts
    recording_id, diagnosis = recording_and_diagnosis.split("_", maxsplit=1)
    match = re.fullmatch(r"([A-Za-z]+)(\d+)", recording_id)
    if match is None:
        raise ValueError(f"Unexpected KAUH recording ID: {recording_id}")

    diagnosis = diagnosis.strip()
    return {
        "recording_id": recording_id,
        "device_prefix": match.group(1).upper(),
        "patient": match.group(2),
        "diagnosis": diagnosis,
        "sound_type": sound_type.strip(),
        "chest_location": chest_location.strip(),
        "age": age.strip(),
        "sex": sex.strip(),
        "binary_label": "0" if diagnosis.upper() == "N" else "1",
        "binary_class_name": "healthy" if diagnosis.upper() == "N" else "diseased",
    }


def extract_kauh_mels(
    input_manifest: Path,
    output_dir: Path,
    n_mels: int,
    n_frames: int | None,
    n_fft: int,
    hop_length: int,
    f_min: float,
    f_max: float | None,
    smooth_freq_sigma: float,
    smooth_time_sigma: float,
    overwrite: bool,
) -> None:
    input_root = input_manifest.parent
    feature_dir = output_dir / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)

    with input_manifest.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"No recordings found in {input_manifest}")

    fieldnames = [
        "feature_path",
        "source_file",
        "processed_file",
        "recording_id",
        "device_prefix",
        "patient",
        "diagnosis",
        "sound_type",
        "chest_location",
        "age",
        "sex",
        "binary_label",
        "binary_class_name",
        "sample_rate",
        "duration_seconds",
        "n_mels",
        "n_frames",
        "n_fft",
        "hop_length",
        "f_min",
        "f_max",
        "smooth_freq_sigma",
        "smooth_time_sigma",
    ]

    output_manifest = output_dir / "manifest.csv"
    with output_manifest.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for index, row in enumerate(rows, start=1):
            wav_path = input_root / row["processed_file"]
            sample_rate, audio = read_wav_mono(wav_path)
            metadata = parse_kauh_name(row["source_file"])
            feature_path = feature_dir / f"{Path(row['source_file']).stem}_mel.npy"
            effective_f_max = sample_rate / 2.0 if f_max is None else min(f_max, sample_rate / 2.0)

            if overwrite or not feature_path.exists():
                feature = log_mel_spectrogram(
                    audio=audio,
                    sample_rate=sample_rate,
                    n_mels=n_mels,
                    n_frames=n_frames,
                    n_fft=n_fft,
                    hop_length=hop_length,
                    f_min=f_min,
                    f_max=effective_f_max,
                    smooth_freq_sigma=smooth_freq_sigma,
                    smooth_time_sigma=smooth_time_sigma,
                )
                np.save(feature_path, feature)
            else:
                feature = np.load(feature_path, mmap_mode="r")

            writer.writerow(
                {
                    "feature_path": feature_path.relative_to(output_dir),
                    "source_file": row["source_file"],
                    "processed_file": row["processed_file"],
                    **metadata,
                    "sample_rate": sample_rate,
                    "duration_seconds": row["duration_seconds"],
                    "n_mels": n_mels,
                    "n_frames": feature.shape[1],
                    "n_fft": n_fft,
                    "hop_length": hop_length,
                    "f_min": f"{f_min:.1f}",
                    "f_max": f"{effective_f_max:.1f}",
                    "smooth_freq_sigma": f"{smooth_freq_sigma:.3f}",
                    "smooth_time_sigma": f"{smooth_time_sigma:.3f}",
                }
            )
            file.flush()
            print(f"[{index}/{len(rows)}] {row['source_file']}", flush=True)

    print(f"Mel features: {feature_dir}")
    print(f"Manifest: {output_manifest}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract full-range log-mel spectrograms from preprocessed KAUH audio."
    )
    parser.add_argument(
        "--input-manifest",
        type=Path,
        default=Path("kauh_preprocessed") / "manifest.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("kauh_mel_dataset"))
    parser.add_argument("--n-mels", type=int, default=64)
    parser.add_argument(
        "--n-frames",
        type=int,
        default=None,
        help="Optional fixed frame count. Default keeps the complete time axis.",
    )
    parser.add_argument("--n-fft", type=int, default=512)
    parser.add_argument("--hop-length", type=int, default=128)
    parser.add_argument("--f-min", type=float, default=0.0)
    parser.add_argument(
        "--f-max",
        type=float,
        default=None,
        help="Optional maximum frequency. Default uses Nyquist.",
    )
    parser.add_argument("--smooth-freq-sigma", type=float, default=0.6)
    parser.add_argument("--smooth-time-sigma", type=float, default=0.8)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    extract_kauh_mels(
        input_manifest=args.input_manifest,
        output_dir=args.output_dir,
        n_mels=args.n_mels,
        n_frames=args.n_frames,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        f_min=args.f_min,
        f_max=args.f_max,
        smooth_freq_sigma=args.smooth_freq_sigma,
        smooth_time_sigma=args.smooth_time_sigma,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
