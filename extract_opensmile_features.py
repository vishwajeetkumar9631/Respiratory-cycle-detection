from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import wavfile


DEFAULT_MANIFEST = Path("kauh_hf_paper_dataset") / "manifest.csv"
DEFAULT_AUDIO_DIR = Path("kauh_preprocessed") / "audio"
DEFAULT_OUTPUT = Path("opensmile_features") / "segment_features.csv"
KEY_COLUMNS = ["source_file", "segment_index", "start_time_s", "end_time_s"]


def load_opensmile():
    try:
        import opensmile
    except ImportError as exc:
        raise SystemExit(
            "The Python package 'opensmile' is required.\n"
            "Install it with: python -m pip install opensmile"
        ) from exc
    return opensmile


def read_audio(path: Path) -> tuple[int, np.ndarray]:
    sample_rate, audio = wavfile.read(path)
    original_dtype = audio.dtype
    audio = np.asarray(audio)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32, copy=False)
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if np.issubdtype(original_dtype, np.integer):
        max_value = float(np.iinfo(original_dtype).max)
        if max_value > 0:
            audio = audio / max_value
    elif peak > 1.0:
        audio = audio / peak
    return int(sample_rate), np.clip(audio, -1.0, 1.0)


def make_smile(feature_set: str, feature_level: str):
    opensmile = load_opensmile()
    try:
        smile_feature_set = getattr(opensmile.FeatureSet, feature_set)
    except AttributeError as exc:
        valid = ", ".join(name for name in dir(opensmile.FeatureSet) if name.isupper())
        raise ValueError(f"Unknown OpenSMILE feature set '{feature_set}'. Valid examples: {valid}") from exc
    try:
        smile_feature_level = getattr(opensmile.FeatureLevel, feature_level)
    except AttributeError as exc:
        valid = ", ".join(name for name in dir(opensmile.FeatureLevel) if name.isupper())
        raise ValueError(f"Unknown OpenSMILE feature level '{feature_level}'. Valid examples: {valid}") from exc
    return opensmile.Smile(feature_set=smile_feature_set, feature_level=smile_feature_level)


def resolve_audio_path(row: dict[str, object], audio_dir: Path) -> Path:
    if "processed_file" in row and pd.notna(row["processed_file"]):
        candidate = Path(str(row["processed_file"]))
        if candidate.is_absolute():
            return candidate
        if (audio_dir / candidate.name).exists():
            return audio_dir / candidate.name
        return audio_dir / candidate

    source_file = Path(str(row["source_file"]))
    if source_file.is_absolute():
        return source_file
    return audio_dir / source_file.name


def infer_label(frame: pd.DataFrame) -> pd.Series:
    if "binary_label" in frame.columns:
        return frame["binary_label"].astype(int)
    if {"crackle", "wheeze"}.issubset(frame.columns):
        return (frame["crackle"].astype(int).gt(0) | frame["wheeze"].astype(int).gt(0)).astype(int)
    if "class_name" in frame.columns:
        return frame["class_name"].astype(str).str.casefold().ne("normal").astype(int)
    if "diagnosis" in frame.columns:
        normal = frame["diagnosis"].astype(str).str.casefold().isin(["n", "normal", "healthy"])
        return (~normal).astype(int)
    return pd.Series(np.zeros(len(frame), dtype=np.int64), index=frame.index)


def class_names(labels: pd.Series) -> pd.Series:
    return np.where(labels.astype(int).eq(1), "unhealthy", "healthy")


def cycle_segments(frame: pd.DataFrame, audio_dir: Path, min_duration: float) -> list[dict[str, object]]:
    required = {"source_file", "start_time_s", "end_time_s"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Cycle mode requires manifest columns: {missing}")

    labels = infer_label(frame)
    classes = frame["binary_class_name"] if "binary_class_name" in frame.columns else class_names(labels)
    rows: list[dict[str, object]] = []
    for row, label, class_name in zip(frame.to_dict("records"), labels, classes):
        start = float(row["start_time_s"])
        end = float(row["end_time_s"])
        duration = float(row.get("duration_s", end - start))
        if duration < min_duration:
            continue
        rows.append(
            {
                "source_file": row["source_file"],
                "audio_path": resolve_audio_path(row, audio_dir),
                "patient": str(row.get("patient", "")),
                "diagnosis": row.get("diagnosis", row.get("class_name", "")),
                "split": row.get("split", "all"),
                "label": int(label),
                "class_name": class_name,
                "segment_index": int(row.get("cycle_index", len(rows) + 1)),
                "start_time_s": start,
                "end_time_s": end,
                "duration_s": duration,
            }
        )
    return rows


def fixed_segments(frame: pd.DataFrame, audio_dir: Path, window_seconds: float) -> list[dict[str, object]]:
    labels = infer_label(frame)
    frame = frame.assign(_label=labels)
    rows: list[dict[str, object]] = []

    for source_file, group in frame.groupby("source_file", sort=True):
        recording = group.iloc[0].to_dict()
        audio_path = resolve_audio_path(recording, audio_dir)
        if not audio_path.exists():
            continue
        sample_rate, audio = read_audio(audio_path)
        total_seconds = audio.size / sample_rate
        count = int(math.floor(total_seconds / window_seconds))
        label = int(group["_label"].max())
        for index in range(count):
            start = index * window_seconds
            end = start + window_seconds
            rows.append(
                {
                    "source_file": source_file,
                    "audio_path": audio_path,
                    "patient": str(recording.get("patient", "")),
                    "diagnosis": recording.get("diagnosis", recording.get("class_name", "")),
                    "split": recording.get("split", "all"),
                    "label": label,
                    "class_name": "unhealthy" if label else "healthy",
                    "segment_index": index + 1,
                    "start_time_s": start,
                    "end_time_s": end,
                    "duration_s": window_seconds,
                }
            )
    return rows


def sanitize_segment(segment: np.ndarray) -> np.ndarray | None:
    if segment.size == 0:
        return None
    segment = np.asarray(segment, dtype=np.float32)
    segment = np.nan_to_num(segment, nan=0.0, posinf=0.0, neginf=0.0)
    if not np.any(segment):
        return None
    return np.clip(segment, -1.0, 1.0)


def segment_key(row: dict[str, object] | pd.Series) -> tuple[str, int, str, str]:
    return (
        str(row["source_file"]),
        int(row["segment_index"]),
        f"{float(row['start_time_s']):.6f}",
        f"{float(row['end_time_s']):.6f}",
    )


def load_previous_features(path: Path, overwrite: bool) -> tuple[pd.DataFrame | None, set[tuple[str, int, str, str]]]:
    if overwrite or not path.exists():
        return None, set()

    previous = pd.read_csv(path)
    missing = sorted(set(KEY_COLUMNS) - set(previous.columns))
    if missing:
        print(f"Existing output is missing resume key columns {missing}; rebuilding it.", flush=True)
        return None, set()

    keys = {segment_key(row) for _, row in previous.iterrows()}
    print(f"Loaded {len(previous)} previous OpenSMILE feature rows from: {path}", flush=True)
    return previous, keys


def extract_features(args: argparse.Namespace) -> None:
    manifest = pd.read_csv(args.manifest)
    if args.exclude_augmented and "augmented" in manifest.columns:
        manifest = manifest.loc[manifest["augmented"].astype(int).eq(0)].copy()

    if args.segment_mode == "cycle":
        rows = cycle_segments(manifest, args.audio_dir, args.min_duration_seconds)
    else:
        rows = fixed_segments(manifest, args.audio_dir, args.window_seconds)
    if not rows:
        raise ValueError("No segments were found. Check --manifest, --audio-dir, and --segment-mode.")

    smile = make_smile(args.feature_set, args.feature_level)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    previous_features, previous_keys = load_previous_features(args.output, args.overwrite)

    feature_rows: list[dict[str, object]] = []
    missing_audio: set[Path] = set()
    skipped_existing = 0
    cached_path: Path | None = None
    cached_rate: int | None = None
    cached_audio: np.ndarray | None = None

    for index, row in enumerate(rows, start=1):
        if segment_key(row) in previous_keys:
            skipped_existing += 1
            continue

        audio_path = Path(row["audio_path"])
        if not audio_path.exists():
            missing_audio.add(audio_path)
            continue
        if cached_path != audio_path:
            cached_rate, cached_audio = read_audio(audio_path)
            cached_path = audio_path

        sample_rate = int(cached_rate)
        start = max(int(round(float(row["start_time_s"]) * sample_rate)), 0)
        end = min(int(round(float(row["end_time_s"]) * sample_rate)), cached_audio.size)
        segment = sanitize_segment(cached_audio[start:end])
        if segment is None or segment.size < int(args.min_duration_seconds * sample_rate):
            continue

        features = smile.process_signal(segment, sample_rate).iloc[0]
        values = features.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        output_row = {
            "source_file": row["source_file"],
            "patient": row["patient"],
            "diagnosis": row["diagnosis"],
            "split": row["split"],
            "label": row["label"],
            "class_name": row["class_name"],
            "segment_index": row["segment_index"],
            "start_time_s": row["start_time_s"],
            "end_time_s": row["end_time_s"],
            "duration_s": row["duration_s"],
        }
        output_row.update({f"smile_{name}": float(value) for name, value in values.items()})
        feature_rows.append(output_row)

        if index == 1 or index % args.log_every == 0:
            print(f"[{index}/{len(rows)}] extracted {audio_path.name}", flush=True)

    if missing_audio:
        print(f"Skipped {len(missing_audio)} missing audio files.")
    if skipped_existing:
        print(f"Skipped {skipped_existing} previously extracted segment rows.")
    if not feature_rows and previous_features is None:
        raise ValueError("No OpenSMILE features were extracted.")

    new_features = pd.DataFrame(feature_rows)
    if previous_features is not None:
        output = pd.concat([previous_features, new_features], ignore_index=True) if feature_rows else previous_features
        output = output.drop_duplicates(subset=KEY_COLUMNS, keep="last")
    else:
        output = new_features

    output.to_csv(args.output, index=False)
    feature_count = sum(name.startswith("smile_") for name in output.columns)
    print(f"OpenSMILE features saved to: {args.output}")
    print(f"Rows: {len(output)}  New rows: {len(feature_rows)}  Feature columns: {feature_count}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract OpenSMILE features from respiratory audio segments.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--audio-dir", type=Path, default=DEFAULT_AUDIO_DIR)
    parser.add_argument("--output", "--output-features", dest="output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--feature-set", default="IS10")
    parser.add_argument("--feature-level", default="Functionals")
    parser.add_argument("--segment-mode", choices=["cycle", "fixed"], default="cycle")
    parser.add_argument("--window-seconds", type=float, default=1.0)
    parser.add_argument("--min-duration-seconds", type=float, default=0.25)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Ignore any previous output CSV and rebuild OpenSMILE features from scratch.",
    )
    parser.add_argument(
        "--include-augmented",
        dest="exclude_augmented",
        action="store_false",
        help="Include augmented rows from manifests that contain an augmented column.",
    )
    parser.set_defaults(exclude_augmented=True)
    args = parser.parse_args()
    extract_features(args)


if __name__ == "__main__":
    main()
