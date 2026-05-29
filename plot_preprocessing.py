from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from segment_icbhi_cycles import DEFAULT_DATASET, highpass, read_wav_mono, zscore


DEFAULT_RECORDING = "107_2b5_Ar_mc_AKGC417L.wav"


def preprocess_for_plot(
    audio: np.ndarray,
    input_rate: int,
    target_rate: int = 8000,
) -> tuple[int, np.ndarray]:
    if input_rate != target_rate:
        processed = signal.resample_poly(audio, target_rate, input_rate)
    else:
        processed = audio.copy()
    processed = zscore(processed)
    processed = highpass(processed, target_rate)
    return target_rate, processed


def plot_before_after(
    wav_path: Path,
    output_path: Path,
    start_s: float | None = None,
    duration_s: float | None = None,
) -> None:
    input_rate, raw = read_wav_mono(wav_path)
    processed_rate, processed = preprocess_for_plot(raw, input_rate)

    raw_time = np.arange(raw.size, dtype=np.float64) / input_rate
    processed_time = np.arange(processed.size, dtype=np.float64) / processed_rate

    if start_s is not None or duration_s is not None:
        start = max(start_s or 0.0, 0.0)
        end = start + duration_s if duration_s is not None else raw_time[-1]
        raw_mask = (raw_time >= start) & (raw_time <= end)
        processed_mask = (processed_time >= start) & (processed_time <= end)
    else:
        raw_mask = np.ones(raw_time.shape, dtype=bool)
        processed_mask = np.ones(processed_time.shape, dtype=bool)

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    fig.suptitle(wav_path.name, fontsize=14)

    axes[0].plot(raw_time[raw_mask], raw[raw_mask], color="#2b6cb0", linewidth=0.8)
    axes[0].set_title(f"Before preprocessing: raw mono signal ({input_rate} Hz)")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(alpha=0.25)

    axes[1].plot(
        processed_time[processed_mask],
        processed[processed_mask],
        color="#c05621",
        linewidth=0.8,
    )
    axes[1].set_title(
        "After preprocessing: resampled to 8000 Hz, z-score normalized, 100 Hz high-pass"
    )
    axes[1].set_xlabel("Time (seconds)")
    axes[1].set_ylabel("Amplitude (z-score)")
    axes[1].grid(alpha=0.25)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"Saved plot to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot an ICBHI signal before and after preprocessing."
    )
    parser.add_argument(
        "--wav",
        type=Path,
        default=DEFAULT_DATASET / DEFAULT_RECORDING,
        help="Path to a .wav recording.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("plots") / f"{Path(DEFAULT_RECORDING).stem}_preprocessing.png",
        help="Output PNG path.",
    )
    parser.add_argument(
        "--start",
        type=float,
        default=None,
        help="Optional start time in seconds.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Optional duration in seconds.",
    )
    args = parser.parse_args()

    plot_before_after(args.wav, args.output, args.start, args.duration)


if __name__ == "__main__":
    main()
