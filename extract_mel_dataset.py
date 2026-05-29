from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy import signal

from segment_icbhi_cycles import (
    DEFAULT_DATASET,
    RecordingName,
    highpass,
    parse_icbhi_name,
    read_wav_mono,
    zscore,
)


CLASS_NAMES = {
    (0, 0): "normal",
    (1, 0): "crackle",
    (0, 1): "wheeze",
    (1, 1): "both",
}


def hz_to_mel(frequency_hz: np.ndarray | float) -> np.ndarray | float:
    return 2595.0 * np.log10(1.0 + np.asarray(frequency_hz) / 700.0)


def mel_to_hz(mel: np.ndarray | float) -> np.ndarray | float:
    return 700.0 * (np.power(10.0, np.asarray(mel) / 2595.0) - 1.0)


def vtlp_warp_hz(
    frequency_hz: np.ndarray,
    alpha: float,
    f_high: float,
) -> np.ndarray:
    if alpha <= 0.0:
        raise ValueError("VTLP alpha must be positive")
    if np.isclose(alpha, 1.0):
        return frequency_hz

    boundary = min(f_high, f_high / alpha)
    warped = frequency_hz * alpha
    if boundary < f_high:
        scale = (f_high - boundary * alpha) / max(f_high - boundary, np.finfo(float).eps)
        warped = np.where(
            frequency_hz <= boundary,
            warped,
            f_high - (f_high - frequency_hz) * scale,
        )
    return np.clip(warped, 0.0, f_high)


def mel_filterbank(
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    f_min: float,
    f_max: float,
    vtlp_alpha: float = 1.0,
) -> np.ndarray:
    if f_max <= f_min:
        raise ValueError("f_max must be greater than f_min")

    fft_frequencies = np.linspace(0.0, sample_rate / 2.0, n_fft // 2 + 1)
    warped_frequencies = vtlp_warp_hz(fft_frequencies, vtlp_alpha, f_max)
    warped_mels = hz_to_mel(warped_frequencies)

    mel_edges = np.linspace(hz_to_mel(f_min), hz_to_mel(f_max), n_mels + 2)
    filters = np.zeros((n_mels, fft_frequencies.size), dtype=np.float64)

    for index in range(n_mels):
        left, center, right = mel_edges[index : index + 3]
        lower = (warped_mels - left) / max(center - left, np.finfo(float).eps)
        upper = (right - warped_mels) / max(right - center, np.finfo(float).eps)
        filters[index] = np.maximum(0.0, np.minimum(lower, upper))

    enorm = 2.0 / np.maximum(mel_to_hz(mel_edges[2 : n_mels + 2]) - mel_to_hz(mel_edges[:n_mels]), 1e-12)
    filters *= enorm[:, np.newaxis]
    return filters


def preprocess_audio(
    audio: np.ndarray,
    input_rate: int,
    target_rate: int,
) -> np.ndarray:
    if input_rate != target_rate:
        processed = signal.resample_poly(audio, target_rate, input_rate)
    else:
        processed = audio.copy()
    return highpass(zscore(processed), target_rate)


def resize_time_axis(spectrogram: np.ndarray, n_frames: int) -> np.ndarray:
    if spectrogram.shape[1] == n_frames:
        return spectrogram
    if spectrogram.shape[1] < 2:
        return np.repeat(spectrogram, n_frames, axis=1)
    return signal.resample(spectrogram, n_frames, axis=1)


def smooth_log_mel(
    log_mel: np.ndarray,
    freq_sigma: float,
    time_sigma: float,
) -> np.ndarray:
    if freq_sigma <= 0.0 and time_sigma <= 0.0:
        return log_mel
    return ndimage.gaussian_filter(
        log_mel,
        sigma=(max(freq_sigma, 0.0), max(time_sigma, 0.0)),
        mode="nearest",
    )


def log_mel_spectrogram(
    audio: np.ndarray,
    sample_rate: int,
    n_mels: int,
    n_frames: int,
    n_fft: int,
    hop_length: int,
    f_min: float,
    f_max: float,
    vtlp_alpha: float = 1.0,
    smooth_freq_sigma: float = 0.6,
    smooth_time_sigma: float = 0.8,
) -> np.ndarray:
    if audio.size < n_fft:
        audio = np.pad(audio, (0, n_fft - audio.size))

    _, _, stft = signal.stft(
        audio,
        fs=sample_rate,
        window="hann",
        nperseg=n_fft,
        noverlap=max(n_fft - hop_length, 0),
        nfft=n_fft,
        boundary="zeros",
        padded=True,
    )
    power = np.abs(stft) ** 2
    filters = mel_filterbank(sample_rate, n_fft, n_mels, f_min, f_max, vtlp_alpha)
    mel_power = filters @ power
    log_mel = 10.0 * np.log10(np.maximum(mel_power, 1e-10))
    log_mel = resize_time_axis(log_mel, n_frames)
    log_mel = smooth_log_mel(log_mel, smooth_freq_sigma, smooth_time_sigma)
    return zscore(log_mel).astype(np.float32)


def read_cycle_rows(path: Path) -> list[tuple[float, float, int, int]]:
    rows: list[tuple[float, float, int, int]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            parts = line.split()
            if not parts:
                continue
            if len(parts) < 2:
                raise ValueError(f"{path}:{line_number} has fewer than two columns")
            start = float(parts[0])
            end = float(parts[1])
            crackle = int(parts[2]) if len(parts) > 2 else 0
            wheeze = int(parts[3]) if len(parts) > 3 else 0
            rows.append((start, end, crackle, wheeze))
    return rows


def class_name(crackle: int, wheeze: int) -> str:
    return CLASS_NAMES.get((int(bool(crackle)), int(bool(wheeze))), "unknown")


def best_overlap_label(
    start: float,
    end: float,
    annotation_rows: list[tuple[float, float, int, int]],
    fallback_crackle: int,
    fallback_wheeze: int,
) -> tuple[int, int]:
    best_overlap = 0.0
    best_label = (fallback_crackle, fallback_wheeze)
    for ann_start, ann_end, ann_crackle, ann_wheeze in annotation_rows:
        overlap = max(0.0, min(end, ann_end) - max(start, ann_start))
        if overlap > best_overlap:
            best_overlap = overlap
            best_label = (ann_crackle, ann_wheeze)
    return best_label


def feature_filename(
    recording_stem: str,
    cycle_index: int,
    augmented: bool,
    augment_index: int,
) -> str:
    if augmented:
        return f"{recording_stem}_cycle_{cycle_index:03d}_vtlp_{augment_index:02d}.npy"
    return f"{recording_stem}_cycle_{cycle_index:03d}.npy"


def write_feature(
    feature_dir: Path,
    name: str,
    feature: np.ndarray,
    overwrite: bool,
) -> Path:
    path = feature_dir / name
    if path.exists() and not overwrite:
        return path
    np.save(path, feature)
    return path


def build_mel_dataset(
    dataset_dir: Path,
    annotation_dir: Path,
    cycles_dir: Path,
    output_dir: Path,
    target_rate: int,
    n_mels: int,
    n_frames: int,
    n_fft: int,
    hop_length: int,
    f_min: float,
    f_max: float,
    smooth_freq_sigma: float,
    smooth_time_sigma: float,
    augment_vtlp: int,
    vtlp_min: float,
    vtlp_max: float,
    seed: int,
    overwrite: bool,
    manifest_only: bool,
    limit: int | None,
) -> None:
    feature_dir = output_dir / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.csv"
    rng = np.random.default_rng(seed)

    cycle_paths = sorted(cycles_dir.glob("*_detected_cycles.txt"))
    if limit is not None:
        cycle_paths = cycle_paths[:limit]
    if not cycle_paths:
        raise FileNotFoundError(f"No detected cycle files found in {cycles_dir}")

    fieldnames = [
        "feature_path",
        "source_file",
        "patient",
        "recording_index",
        "chest_location",
        "acquisition_mode",
        "equipment",
        "cycle_index",
        "start_time_s",
        "end_time_s",
        "duration_s",
        "crackle",
        "wheeze",
        "class_name",
        "augmented",
        "vtlp_alpha",
        "sample_rate",
        "n_mels",
        "n_frames",
        "smooth_freq_sigma",
        "smooth_time_sigma",
    ]

    total_features = 0
    with manifest_path.open("w", encoding="utf-8", newline="") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=fieldnames)
        writer.writeheader()

        for file_index, cycle_path in enumerate(cycle_paths, start=1):
            recording_stem = cycle_path.name.removesuffix("_detected_cycles.txt")
            wav_path = dataset_dir / f"{recording_stem}.wav"
            if not wav_path.exists():
                print(f"[skip] Missing WAV for {cycle_path.name}: {wav_path}", flush=True)
                continue

            input_rate, raw_audio = read_wav_mono(wav_path)
            processed = preprocess_audio(raw_audio, input_rate, target_rate)
            cycles = read_cycle_rows(cycle_path)
            annotation_path = annotation_dir / f"{recording_stem}.txt"
            annotations = read_cycle_rows(annotation_path) if annotation_path.exists() else []
            recording = parse_icbhi_name(wav_path)

            for cycle_index, (start, end, crackle, wheeze) in enumerate(cycles, start=1):
                crackle, wheeze = best_overlap_label(start, end, annotations, crackle, wheeze)
                start_sample = max(int(round(start * target_rate)), 0)
                end_sample = min(int(round(end * target_rate)), processed.size)
                if end_sample <= start_sample:
                    continue

                cycle_audio = processed[start_sample:end_sample]
                alphas = [1.0]
                alphas.extend(float(rng.uniform(vtlp_min, vtlp_max)) for _ in range(augment_vtlp))

                for augment_index, alpha in enumerate(alphas):
                    augmented = augment_index > 0
                    out_name = feature_filename(recording_stem, cycle_index, augmented, augment_index)
                    if manifest_only:
                        feature_path = feature_dir / out_name
                    else:
                        feature = log_mel_spectrogram(
                            cycle_audio,
                            target_rate,
                            n_mels,
                            n_frames,
                            n_fft,
                            hop_length,
                            f_min,
                            f_max,
                            alpha,
                            smooth_freq_sigma,
                            smooth_time_sigma,
                        )
                        feature_path = write_feature(feature_dir, out_name, feature, overwrite)

                    metadata: RecordingName = recording
                    writer.writerow(
                        {
                            "feature_path": str(feature_path.relative_to(output_dir)),
                            "source_file": wav_path.name,
                            **asdict(metadata),
                            "cycle_index": cycle_index,
                            "start_time_s": f"{start:.3f}",
                            "end_time_s": f"{end:.3f}",
                            "duration_s": f"{end - start:.3f}",
                            "crackle": crackle,
                            "wheeze": wheeze,
                            "class_name": class_name(crackle, wheeze),
                            "augmented": int(augmented),
                            "vtlp_alpha": f"{alpha:.6f}",
                            "sample_rate": target_rate,
                            "n_mels": n_mels,
                            "n_frames": n_frames,
                            "smooth_freq_sigma": f"{smooth_freq_sigma:.3f}",
                            "smooth_time_sigma": f"{smooth_time_sigma:.3f}",
                        }
                    )
                    total_features += 1

            manifest_file.flush()
            print(
                f"[{file_index}/{len(cycle_paths)}] {wav_path.name}: "
                f"{len(cycles)} cycles",
                flush=True,
            )

    print(f"Features written to: {feature_dir}")
    print(f"Manifest written to: {manifest_path}")
    print(f"Total feature tensors: {total_features}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract cycle-level log-mel spectrogram tensors from detected cycles."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET,
        help=f"ICBHI_final_database directory. Default: {DEFAULT_DATASET}",
    )
    parser.add_argument(
        "--cycles-dir",
        type=Path,
        default=Path("detected_cycles"),
        help="Directory containing *_detected_cycles.txt files.",
    )
    parser.add_argument(
        "--annotation-dir",
        type=Path,
        default=None,
        help="Directory containing original ICBHI .txt annotations. Default: dataset dir.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("mel_dataset"),
        help="Output directory for .npy features and manifest.csv.",
    )
    parser.add_argument("--target-rate", type=int, default=8000)
    parser.add_argument("--n-mels", type=int, default=64)
    parser.add_argument("--n-frames", type=int, default=128)
    parser.add_argument("--n-fft", type=int, default=512)
    parser.add_argument("--hop-length", type=int, default=128)
    parser.add_argument("--f-min", type=float, default=100.0)
    parser.add_argument("--f-max", type=float, default=2000.0)
    parser.add_argument(
        "--smooth-freq-sigma",
        type=float,
        default=0.6,
        help="Gaussian smoothing sigma across mel bins. Use 0 to disable.",
    )
    parser.add_argument(
        "--smooth-time-sigma",
        type=float,
        default=0.8,
        help="Gaussian smoothing sigma across time frames. Use 0 to disable.",
    )
    parser.add_argument(
        "--augment-vtlp",
        type=int,
        default=0,
        help="Number of VTLP augmented copies to create per cycle.",
    )
    parser.add_argument("--vtlp-min", type=float, default=0.9)
    parser.add_argument("--vtlp-max", type=float, default=1.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Only rebuild manifest.csv labels/metadata; do not write .npy feature files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of recordings to process for a quick smoke test.",
    )
    args = parser.parse_args()

    build_mel_dataset(
        dataset_dir=args.dataset_dir,
        annotation_dir=args.annotation_dir or args.dataset_dir,
        cycles_dir=args.cycles_dir,
        output_dir=args.output_dir,
        target_rate=args.target_rate,
        n_mels=args.n_mels,
        n_frames=args.n_frames,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        f_min=args.f_min,
        f_max=args.f_max,
        smooth_freq_sigma=args.smooth_freq_sigma,
        smooth_time_sigma=args.smooth_time_sigma,
        augment_vtlp=args.augment_vtlp,
        vtlp_min=args.vtlp_min,
        vtlp_max=args.vtlp_max,
        seed=args.seed,
        overwrite=args.overwrite,
        manifest_only=args.manifest_only,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
