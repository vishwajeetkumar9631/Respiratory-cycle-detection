from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import interpolate, signal
from scipy.io import wavfile


DEFAULT_DATASET = Path(
    r"C:\Users\ankit\Downloads\ICBHI_final_database\ICBHI_final_database"
)


@dataclass(frozen=True)
class RecordingName:
    patient: str
    recording_index: str
    chest_location: str
    acquisition_mode: str
    equipment: str


def parse_icbhi_name(path: Path) -> RecordingName:
    parts = path.stem.split("_")
    if len(parts) < 5:
        raise ValueError(f"Unexpected ICBHI filename: {path.name}")
    return RecordingName(
        patient=parts[0],
        recording_index=parts[1],
        chest_location=parts[2],
        acquisition_mode=parts[3],
        equipment="_".join(parts[4:]),
    )


def read_wav_mono(path: Path) -> tuple[int, np.ndarray]:
    sample_rate, audio = wavfile.read(path)
    audio = np.asarray(audio)
    original_dtype = audio.dtype
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float64)
    if np.issubdtype(original_dtype, np.integer):
        limit = np.iinfo(original_dtype).max
        audio = audio / max(limit, 1)
    return sample_rate, audio


def zscore(audio: np.ndarray) -> np.ndarray:
    std = float(np.std(audio))
    if std <= np.finfo(float).eps:
        return audio - float(np.mean(audio))
    return (audio - float(np.mean(audio))) / std


def highpass(audio: np.ndarray, sample_rate: int, cutoff_hz: float = 100.0) -> np.ndarray:
    nyquist = sample_rate / 2.0
    if cutoff_hz >= nyquist:
        raise ValueError(f"Cutoff {cutoff_hz} Hz is >= Nyquist {nyquist} Hz")
    sos = signal.butter(4, cutoff_hz / nyquist, btype="highpass", output="sos")
    return signal.sosfiltfilt(sos, audio)


def windowed_hilbert_envelope(
    audio: np.ndarray,
    sample_rate: int,
    window_s: float = 1.0,
    hop_s: float = 0.1,
) -> np.ndarray:
    win_len = max(int(round(window_s * sample_rate)), 8)
    hop = max(int(round(hop_s * sample_rate)), 1)
    window = signal.windows.hann(win_len, sym=False)

    envelope = np.zeros(audio.shape[0], dtype=np.float64)
    weights = np.zeros(audio.shape[0], dtype=np.float64)

    starts = list(range(0, max(audio.shape[0] - win_len + 1, 1), hop))
    if not starts or starts[-1] + win_len < audio.shape[0]:
        starts.append(max(audio.shape[0] - win_len, 0))

    for start in starts:
        stop = min(start + win_len, audio.shape[0])
        chunk = audio[start:stop]
        local_window = window[: chunk.shape[0]]
        analytic = signal.hilbert(chunk * local_window)
        envelope[start:stop] += np.abs(analytic) * local_window
        weights[start:stop] += local_window

    valid = weights > np.finfo(float).eps
    envelope[valid] /= weights[valid]
    envelope[~valid] = np.abs(signal.hilbert(audio))[~valid]
    return envelope


def moving_average(values: np.ndarray, sample_rate: int, window_s: float) -> np.ndarray:
    size = max(int(round(window_s * sample_rate)), 1)
    kernel = np.ones(size, dtype=np.float64) / size
    return np.convolve(values, kernel, mode="same")


def estimate_respiration_period(
    envelope: np.ndarray,
    sample_rate: int,
    min_bpm: float = 6.0,
    max_bpm: float = 40.0,
) -> float:
    centered = envelope - np.mean(envelope)
    autocorr = signal.correlate(centered, centered, mode="full", method="fft")
    autocorr = autocorr[autocorr.size // 2 :]

    min_lag = max(int(round(sample_rate * 60.0 / max_bpm)), 1)
    max_lag = min(int(round(sample_rate * 60.0 / min_bpm)), autocorr.size - 1)
    if max_lag <= min_lag:
        return 3.0

    search = autocorr[min_lag : max_lag + 1]
    peaks, _ = signal.find_peaks(search)
    if peaks.size:
        lag = min_lag + peaks[np.argmax(search[peaks])]
    else:
        lag = min_lag + int(np.argmax(search))
    return lag / sample_rate


def enhance_peaks(
    envelope: np.ndarray,
    sample_rate: int,
    interpolation_rate: int = 50,
) -> np.ndarray:
    if envelope.size < 4:
        return envelope

    down_factor = max(int(round(sample_rate / interpolation_rate)), 1)
    coarse = signal.decimate(envelope, down_factor, ftype="fir", zero_phase=True)
    coarse_rate = sample_rate / down_factor

    coarse_time = np.arange(coarse.size, dtype=np.float64) / coarse_rate
    full_time = np.arange(envelope.size, dtype=np.float64) / sample_rate
    spline = interpolate.CubicSpline(coarse_time, coarse, extrapolate=True)
    enhanced = spline(full_time)
    return np.maximum(enhanced, 0.0)


def detect_cycles(
    audio: np.ndarray,
    input_rate: int,
    target_rate: int = 8000,
    min_cycle_s: float = 0.7,
) -> tuple[list[tuple[float, float]], dict[str, float]]:
    if input_rate != target_rate:
        processed = signal.resample_poly(audio, target_rate, input_rate)
    else:
        processed = audio.copy()

    processed = zscore(processed)
    processed = highpass(processed, target_rate)

    envelope = windowed_hilbert_envelope(processed, target_rate)
    rough_envelope = moving_average(envelope, target_rate, 0.25)
    period_s = estimate_respiration_period(rough_envelope, target_rate)

    smooth_window_s = float(np.clip(period_s / 6.0, 0.15, 0.8))
    smoothed = moving_average(envelope, target_rate, smooth_window_s)
    enhanced = enhance_peaks(smoothed, target_rate)

    valley_distance = max(int(round(target_rate * max(min_cycle_s, period_s * 0.45))), 1)
    prominence = max(float(np.std(enhanced)) * 0.05, np.finfo(float).eps)
    valleys, _ = signal.find_peaks(
        -enhanced,
        distance=valley_distance,
        prominence=prominence,
    )

    duration_s = processed.size / target_rate
    valley_times = valleys / target_rate
    valley_times = valley_times[(valley_times >= 0.0) & (valley_times <= duration_s)]

    if valley_times.size < 2:
        valley_times = np.array([0.0, duration_s], dtype=np.float64)
    else:
        if valley_times[0] > min(0.5, period_s * 0.25):
            valley_times = np.insert(valley_times, 0, 0.0)
        if duration_s - valley_times[-1] > min(0.5, period_s * 0.25):
            valley_times = np.append(valley_times, duration_s)

    cycles: list[tuple[float, float]] = []
    for start, end in zip(valley_times[:-1], valley_times[1:]):
        if end - start >= min_cycle_s:
            cycles.append((float(start), float(end)))

    info = {
        "input_sample_rate": float(input_rate),
        "target_sample_rate": float(target_rate),
        "estimated_period_s": float(period_s),
        "estimated_bpm": float(60.0 / period_s) if period_s > 0 else math.nan,
        "smooth_window_s": smooth_window_s,
    }
    return cycles, info


def write_cycles(path: Path, cycles: list[tuple[float, float]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for start, end in cycles:
            file.write(f"{start:.3f}\t{end:.3f}\t0\t0\n")


def process_dataset(dataset_dir: Path, output_dir: Path, overwrite: bool = False) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "detected_cycles_metadata.csv"

    wav_paths = sorted(dataset_dir.glob("*.wav"))
    if not wav_paths:
        raise FileNotFoundError(f"No .wav files found in {dataset_dir}")

    with metadata_path.open("w", encoding="utf-8", newline="") as metadata_file:
        writer = csv.DictWriter(
            metadata_file,
            fieldnames=[
                "file",
                "patient",
                "recording_index",
                "chest_location",
                "acquisition_mode",
                "equipment",
                "detected_cycles",
                "input_sample_rate",
                "target_sample_rate",
                "estimated_period_s",
                "estimated_bpm",
                "smooth_window_s",
            ],
        )
        writer.writeheader()

        for index, wav_path in enumerate(wav_paths, start=1):
            out_path = output_dir / f"{wav_path.stem}_detected_cycles.txt"
            if out_path.exists() and not overwrite:
                continue

            name = parse_icbhi_name(wav_path)
            sample_rate, audio = read_wav_mono(wav_path)
            cycles, info = detect_cycles(audio, sample_rate)
            write_cycles(out_path, cycles)

            writer.writerow(
                {
                    "file": wav_path.name,
                    "patient": name.patient,
                    "recording_index": name.recording_index,
                    "chest_location": name.chest_location,
                    "acquisition_mode": name.acquisition_mode,
                    "equipment": name.equipment,
                    "detected_cycles": len(cycles),
                    **info,
                }
            )
            metadata_file.flush()
            print(
                f"[{index}/{len(wav_paths)}] {wav_path.name}: {len(cycles)} cycles",
                flush=True,
            )

    print(f"Metadata written to: {metadata_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect respiratory cycle boundaries for ICBHI .wav recordings."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET,
        help=f"ICBHI_final_database directory. Default: {DEFAULT_DATASET}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("detected_cycles"),
        help="Directory for detected cycle .txt files and metadata CSV.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing detected cycle files.",
    )
    args = parser.parse_args()

    process_dataset(args.dataset_dir, args.output_dir, args.overwrite)


if __name__ == "__main__":
    main()
