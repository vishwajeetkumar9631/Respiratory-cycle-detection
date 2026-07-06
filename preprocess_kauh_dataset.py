from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

from extract_mel_dataset import preprocess_audio
from segment_icbhi_cycles import read_wav_mono


DEFAULT_DATASET = Path(r"C:\Users\ankit\Downloads\jwyy9np4gv-3\Audio Files")


def plot_signal(
    source_path: Path,
    raw: np.ndarray,
    input_rate: int,
    processed: np.ndarray,
    target_rate: int,
    output_path: Path,
) -> None:
    raw_time = np.arange(raw.size, dtype=np.float64) / input_rate
    processed_time = np.arange(processed.size, dtype=np.float64) / target_rate

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    fig.suptitle(source_path.name, fontsize=12)

    axes[0].plot(raw_time, raw, color="#2b6cb0", linewidth=0.7)
    axes[0].set_title(f"Raw mono signal ({input_rate} Hz)")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(alpha=0.25)

    axes[1].plot(processed_time, processed, color="#c05621", linewidth=0.7)
    axes[1].set_title(
        f"Preprocessed signal ({target_rate} Hz, z-score, 100 Hz high-pass)"
    )
    axes[1].set_xlabel("Time (seconds)")
    axes[1].set_ylabel("Amplitude (z-score)")
    axes[1].grid(alpha=0.25)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def process_kauh_dataset(
    dataset_dir: Path,
    output_dir: Path,
    skip_first: int,
    target_rate: int,
    overwrite: bool,
) -> None:
    wav_paths = sorted(
        dataset_dir.glob("*.wav"),
        key=lambda path: path.name.casefold(),
    )
    if not wav_paths:
        raise FileNotFoundError(f"No WAV files found in {dataset_dir}")
    if skip_first < 0:
        raise ValueError("--skip-first cannot be negative")

    skipped = wav_paths[:skip_first]
    selected = wav_paths[skip_first:]
    audio_dir = output_dir / "audio"
    plot_dir = output_dir / "plots"
    audio_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as manifest_file:
        writer = csv.DictWriter(
            manifest_file,
            fieldnames=[
                "source_file",
                "processed_file",
                "plot_file",
                "input_sample_rate",
                "target_sample_rate",
                "duration_seconds",
            ],
        )
        writer.writeheader()

        for index, wav_path in enumerate(selected, start=1):
            processed_path = audio_dir / wav_path.name
            plot_path = plot_dir / f"{wav_path.stem}_signal.png"

            input_rate, raw = read_wav_mono(wav_path)
            processed = preprocess_audio(raw, input_rate, target_rate)

            if overwrite or not processed_path.exists():
                wavfile.write(processed_path, target_rate, processed.astype(np.float32))
            if overwrite or not plot_path.exists():
                plot_signal(
                    wav_path,
                    raw,
                    input_rate,
                    processed,
                    target_rate,
                    plot_path,
                )

            writer.writerow(
                {
                    "source_file": wav_path.name,
                    "processed_file": processed_path.relative_to(output_dir),
                    "plot_file": plot_path.relative_to(output_dir),
                    "input_sample_rate": input_rate,
                    "target_sample_rate": target_rate,
                    "duration_seconds": f"{processed.size / target_rate:.3f}",
                }
            )
            manifest_file.flush()
            print(f"[{index}/{len(selected)}] {wav_path.name}", flush=True)

    print(f"Skipped first {len(skipped)} files:")
    for path in skipped:
        print(f"  {path.name}")
    print(f"Preprocessed audio: {audio_dir}")
    print(f"Signal plots: {plot_dir}")
    print(f"Manifest: {manifest_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess KAUH WAV files and plot raw/preprocessed signals."
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=Path("kauh_preprocessed"))
    parser.add_argument(
        "--skip-first",
        type=int,
        default=4,
        help="Exclude this many files from the start of filename-sorted input.",
    )
    parser.add_argument("--target-rate", type=int, default=8000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    process_kauh_dataset(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        skip_first=args.skip_first,
        target_rate=args.target_rate,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
