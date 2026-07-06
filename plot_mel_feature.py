from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from extract_mel_dataset import preprocess_audio, read_cycle_rows, resize_time_axis
from segment_icbhi_cycles import DEFAULT_DATASET, read_wav_mono, zscore


DEFAULT_FEATURE = Path("mel_dataset") / "features" / "211_1p3_Ar_mc_AKGC417L_cycle_004.npy"
FEATURE_PATTERN = re.compile(r"(?P<recording>.+)_cycle_(?P<cycle>\d{3})(?:_.+)?$")


def plot_mel_feature(feature_path: Path, output_path: Path) -> None:
    feature = np.load(feature_path)
    if feature.ndim != 2:
        raise ValueError(f"Expected a 2D mel feature, got shape {feature.shape}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    image = ax.imshow(
        feature,
        origin="lower",
        aspect="auto",
        interpolation="bilinear",
        cmap="magma",
    )
    ax.set_title(feature_path.name)
    ax.set_xlabel("Time frame")
    ax.set_ylabel("Mel bin")
    fig.colorbar(image, ax=ax, label="Normalized log-mel energy")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"Saved plot to: {output_path}")


def parse_feature_name(feature_path: Path) -> tuple[str, int]:
    match = FEATURE_PATTERN.match(feature_path.stem)
    if not match:
        raise ValueError(f"Could not parse recording/cycle from feature name: {feature_path.name}")
    return match.group("recording"), int(match.group("cycle"))


def morlet_scalogram(
    audio: np.ndarray,
    sample_rate: int,
    n_bins: int,
    n_frames: int,
    f_min: float,
    f_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    frequencies = np.geomspace(max(f_min, 1.0), f_max, n_bins)
    rows = []
    wavelet_cycles = 6.0
    for frequency in frequencies:
        sigma_seconds = wavelet_cycles / (2.0 * np.pi * frequency)
        radius = max(int(round(4.0 * sigma_seconds * sample_rate)), 4)
        times = np.arange(-radius, radius + 1, dtype=np.float64) / sample_rate
        wavelet = np.exp(2j * np.pi * frequency * times) * np.exp(
            -(times**2) / (2.0 * sigma_seconds**2)
        )
        wavelet /= np.sqrt(np.sum(np.abs(wavelet) ** 2) + 1e-12)
        coefficients = signal.fftconvolve(audio, np.conj(wavelet[::-1]), mode="same")
        rows.append(np.abs(coefficients) ** 2)

    scalogram = 10.0 * np.log10(np.maximum(np.asarray(rows), 1e-10))
    scalogram = resize_time_axis(scalogram, n_frames)
    return zscore(scalogram).astype(np.float32), frequencies


def plot_scalogram(
    feature_path: Path,
    output_path: Path,
    dataset_dir: Path,
    cycles_dir: Path,
    target_rate: int,
    n_bins: int,
    n_frames: int,
    f_min: float,
    f_max: float,
) -> None:
    recording_stem, cycle_index = parse_feature_name(feature_path)
    wav_path = dataset_dir / f"{recording_stem}.wav"
    cycle_path = cycles_dir / f"{recording_stem}_detected_cycles.txt"
    if not wav_path.exists():
        raise FileNotFoundError(f"Missing WAV file: {wav_path}")
    if not cycle_path.exists():
        raise FileNotFoundError(f"Missing detected cycle file: {cycle_path}")

    cycles = read_cycle_rows(cycle_path)
    if cycle_index < 1 or cycle_index > len(cycles):
        raise ValueError(f"Cycle {cycle_index} not found in {cycle_path}; only {len(cycles)} cycles available")
    start, end, _, _ = cycles[cycle_index - 1]

    input_rate, raw_audio = read_wav_mono(wav_path)
    processed = preprocess_audio(raw_audio, input_rate, target_rate)
    start_sample = max(int(round(start * target_rate)), 0)
    end_sample = min(int(round(end * target_rate)), processed.size)
    if end_sample <= start_sample:
        raise ValueError(f"Invalid cycle boundary: {start:.3f}s to {end:.3f}s")

    scalogram, frequencies = morlet_scalogram(
        processed[start_sample:end_sample],
        target_rate,
        n_bins,
        n_frames,
        f_min,
        f_max,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    image = ax.imshow(
        scalogram,
        origin="lower",
        aspect="auto",
        interpolation="bilinear",
        cmap="viridis",
        extent=[0, n_frames - 1, frequencies[0], frequencies[-1]],
    )
    ax.set_title(f"{recording_stem} cycle {cycle_index:03d} scalogram")
    ax.set_xlabel("Time frame")
    ax.set_ylabel("Frequency (Hz)")
    fig.colorbar(image, ax=ax, label="Normalized wavelet power")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"Saved scalogram plot to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot a saved feature or a wavelet scalogram for its audio cycle.")
    parser.add_argument(
        "--feature",
        type=Path,
        default=DEFAULT_FEATURE,
        help=f"Saved .npy feature path. Default: {DEFAULT_FEATURE}",
    )
    parser.add_argument("--mode", choices=["feature", "scalogram"], default="feature")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("plots") / f"{DEFAULT_FEATURE.stem}_mel.png",
        help="Output PNG path.",
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--cycles-dir", type=Path, default=Path("detected_cycles"))
    parser.add_argument("--target-rate", type=int, default=8000)
    parser.add_argument("--n-bins", type=int, default=64)
    parser.add_argument("--n-frames", type=int, default=128)
    parser.add_argument("--f-min", type=float, default=80.0)
    parser.add_argument("--f-max", type=float, default=2500.0)
    args = parser.parse_args()
    if args.mode == "feature":
        plot_mel_feature(args.feature, args.output)
    else:
        output = args.output
        if output == Path("plots") / f"{DEFAULT_FEATURE.stem}_mel.png":
            output = Path("plots") / f"{args.feature.stem}_scalogram.png"
        plot_scalogram(
            args.feature,
            output,
            args.dataset_dir,
            args.cycles_dir,
            args.target_rate,
            args.n_bins,
            args.n_frames,
            args.f_min,
            args.f_max,
        )


if __name__ == "__main__":
    main()
