from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from extract_mel_dataset import preprocess_audio
from segment_icbhi_cycles import DEFAULT_DATASET, read_wav_mono


def preprocess_dataset(
    dataset_dir: Path,
    output_dir: Path,
    target_rate: int,
    overwrite: bool,
) -> None:
    wav_paths = sorted(dataset_dir.glob("*.wav"), key=lambda path: path.name.casefold())
    if not wav_paths:
        raise FileNotFoundError(f"No WAV files found in {dataset_dir}")

    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.csv"

    with manifest_path.open("w", encoding="utf-8", newline="") as manifest_file:
        writer = csv.DictWriter(
            manifest_file,
            fieldnames=[
                "source_file",
                "processed_file",
                "input_sample_rate",
                "target_sample_rate",
                "duration_seconds",
            ],
        )
        writer.writeheader()

        for index, wav_path in enumerate(wav_paths, start=1):
            processed_path = audio_dir / wav_path.name
            input_rate, raw = read_wav_mono(wav_path)
            processed = preprocess_audio(raw, input_rate, target_rate)
            if overwrite or not processed_path.exists():
                wavfile.write(processed_path, target_rate, processed.astype(np.float32))

            writer.writerow(
                {
                    "source_file": wav_path.name,
                    "processed_file": processed_path.relative_to(output_dir),
                    "input_sample_rate": input_rate,
                    "target_sample_rate": target_rate,
                    "duration_seconds": f"{processed.size / target_rate:.3f}",
                }
            )
            if index == 1 or index % 100 == 0:
                print(f"[{index}/{len(wav_paths)}] {wav_path.name}", flush=True)

    print(f"Preprocessed audio: {audio_dir}")
    print(f"Manifest: {manifest_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess ICBHI WAV files to 8 kHz z-score/high-pass audio.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=Path("icbhi_preprocessed"))
    parser.add_argument("--target-rate", type=int, default=8000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    preprocess_dataset(args.dataset_dir, args.output_dir, args.target_rate, args.overwrite)


if __name__ == "__main__":
    main()
