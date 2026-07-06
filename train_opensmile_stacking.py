from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.io import wavfile
from sklearn.base import clone
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


DEFAULT_MANIFEST = Path("kauh_hf_paper_dataset") / "manifest.csv"
DEFAULT_AUDIO_DIR = Path("kauh_preprocessed") / "audio"
DEFAULT_FEATURES = Path("opensmile_stacking_dataset") / "segment_features.csv"
DEFAULT_MODEL_DIR = Path("models") / "opensmile_stacking"
LABEL_COLUMN = "binary_label"
FULL_STACK_MODELS = ["j48_tree", "naive_bayes", "knn", "svm", "random_forest", "bagging", "boosting"]
OPTIMAL_STACK_MODELS = ["boosting", "svm", "knn", "naive_bayes", "j48_tree"]


def configure_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def load_opensmile():
    try:
        import opensmile
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "The Python package 'opensmile' is required for feature extraction.\n"
            "Install it with: python -m pip install opensmile\n"
            "Then rerun this script."
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
    audio = np.clip(audio, -1.0, 1.0)
    return int(sample_rate), audio


def sanitize_segment(segment: np.ndarray) -> np.ndarray | None:
    if segment.size == 0:
        return None
    segment = np.asarray(segment, dtype=np.float32)
    segment = np.nan_to_num(segment, nan=0.0, posinf=0.0, neginf=0.0)
    if not np.any(segment):
        return None
    return np.clip(segment, -1.0, 1.0)


def feature_extractor(feature_set: str, feature_level: str):
    opensmile = load_opensmile()
    try:
        smile_feature_set = getattr(opensmile.FeatureSet, feature_set)
    except AttributeError as exc:
        valid = ", ".join(name for name in dir(opensmile.FeatureSet) if name.isupper())
        raise ValueError(f"Unknown OpenSmile feature set '{feature_set}'. Valid examples: {valid}") from exc
    try:
        smile_feature_level = getattr(opensmile.FeatureLevel, feature_level)
    except AttributeError as exc:
        valid = ", ".join(name for name in dir(opensmile.FeatureLevel) if name.isupper())
        raise ValueError(f"Unknown OpenSmile feature level '{feature_level}'. Valid examples: {valid}") from exc
    return opensmile.Smile(feature_set=smile_feature_set, feature_level=smile_feature_level)


def segment_rows_from_manifest(
    manifest: pd.DataFrame,
    audio_dir: Path,
    segment_mode: str,
    window_seconds: float,
    min_duration_seconds: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if segment_mode == "cycle":
        for row in manifest.to_dict("records"):
            duration = float(row["duration_s"])
            if duration < min_duration_seconds:
                continue
            rows.append(
                {
                    "source_file": row["source_file"],
                    "audio_path": audio_dir / str(row["source_file"]),
                    "patient": str(row["patient"]),
                    "diagnosis": row.get("diagnosis", row.get("class_name", "")),
                    "split": row["split"],
                    "label": int(row[LABEL_COLUMN]),
                    "class_name": row["binary_class_name"],
                    "segment_index": int(row["cycle_index"]),
                    "start_time_s": float(row["start_time_s"]),
                    "end_time_s": float(row["end_time_s"]),
                    "duration_s": duration,
                }
            )
        return rows

    grouped = manifest.groupby("source_file", sort=True)
    for source_file, recording_rows in grouped:
        recording = recording_rows.iloc[0].to_dict()
        audio_path = audio_dir / str(recording["source_file"])
        if not audio_path.exists():
            continue
        sample_rate, audio = read_audio(audio_path)
        total_seconds = audio.size / sample_rate
        count = int(math.floor(total_seconds / window_seconds))
        recording_label = int(recording_rows[LABEL_COLUMN].max())
        recording_class_name = "unhealthy" if recording_label else "healthy"
        for index in range(count):
            start = index * window_seconds
            end = start + window_seconds
            rows.append(
                {
                    "source_file": source_file,
                    "audio_path": audio_path,
                    "patient": str(recording["patient"]),
                    "diagnosis": recording.get("diagnosis", recording.get("class_name", "")),
                    "split": recording["split"],
                    "label": recording_label,
                    "class_name": recording_class_name,
                    "segment_index": index + 1,
                    "start_time_s": start,
                    "end_time_s": end,
                    "duration_s": window_seconds,
                }
            )
    return rows


def prepare_manifest(args: argparse.Namespace) -> pd.DataFrame:
    manifest = pd.read_csv(args.manifest)
    if args.exclude_augmented and "augmented" in manifest.columns:
        manifest = manifest.loc[manifest["augmented"].astype(int).eq(0)].copy()

    if LABEL_COLUMN not in manifest.columns:
        if {"crackle", "wheeze"}.issubset(manifest.columns):
            manifest[LABEL_COLUMN] = (
                manifest["crackle"].astype(int).gt(0) | manifest["wheeze"].astype(int).gt(0)
            ).astype(int)
        elif "class_name" in manifest.columns:
            manifest[LABEL_COLUMN] = manifest["class_name"].astype(str).str.casefold().ne("normal").astype(int)
        else:
            raise ValueError("Manifest needs binary_label, crackle/wheeze columns, or class_name.")

    if "binary_class_name" not in manifest.columns:
        manifest["binary_class_name"] = np.where(manifest[LABEL_COLUMN].astype(int).eq(1), "unhealthy", "healthy")

    if "split" not in manifest.columns:
        manifest["split"] = "all"

    return manifest


def extract_features(args: argparse.Namespace) -> None:
    manifest = prepare_manifest(args)
    required = {LABEL_COLUMN, "source_file", "patient", "split", "start_time_s", "end_time_s", "duration_s"}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"Manifest is missing required columns: {missing}")

    rows = segment_rows_from_manifest(
        manifest,
        args.audio_dir,
        args.segment_mode,
        args.window_seconds,
        args.min_duration_seconds,
    )
    if not rows:
        raise ValueError("No segment rows were found. Check --manifest, --audio-dir, and --segment-mode.")

    smile = feature_extractor(args.feature_set, args.feature_level)
    args.output_features.parent.mkdir(parents=True, exist_ok=True)

    feature_rows: list[dict[str, object]] = []
    missing_audio: set[Path] = set()
    cached_audio_path: Path | None = None
    cached_sample_rate: int | None = None
    cached_audio: np.ndarray | None = None
    for index, row in enumerate(rows, start=1):
        audio_path = Path(row["audio_path"])
        if not audio_path.exists():
            missing_audio.add(audio_path)
            continue
        if cached_audio_path != audio_path:
            cached_sample_rate, cached_audio = read_audio(audio_path)
            cached_audio_path = audio_path
        sample_rate = int(cached_sample_rate)
        audio = cached_audio
        start_sample = max(int(round(float(row["start_time_s"]) * sample_rate)), 0)
        end_sample = min(int(round(float(row["end_time_s"]) * sample_rate)), audio.size)
        segment = audio[start_sample:end_sample]
        if segment.size < int(args.min_duration_seconds * sample_rate):
            continue
        segment = sanitize_segment(segment)
        if segment is None:
            continue

        features = smile.process_signal(segment, sample_rate)
        values = features.iloc[0].replace([np.inf, -np.inf], np.nan)
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
    if not feature_rows:
        raise ValueError("No OpenSmile features were extracted.")

    pd.DataFrame(feature_rows).to_csv(args.output_features, index=False)
    print(f"OpenSmile segment features saved to: {args.output_features}")
    print(f"Rows: {len(feature_rows)}  Feature columns: {sum(name.startswith('smile_') for name in feature_rows[0])}")


def make_bagging(seed: int) -> BaggingClassifier:
    tree = DecisionTreeClassifier(random_state=seed, class_weight="balanced")
    try:
        return BaggingClassifier(estimator=tree, n_estimators=80, random_state=seed, n_jobs=-1)
    except TypeError:  # pragma: no cover - older scikit-learn
        return BaggingClassifier(base_estimator=tree, n_estimators=80, random_state=seed, n_jobs=-1)


def base_models(seed: int) -> dict[str, Pipeline]:
    models = {
        "j48_tree": DecisionTreeClassifier(random_state=seed, class_weight="balanced", min_samples_leaf=3),
        "naive_bayes": GaussianNB(),
        "knn": KNeighborsClassifier(n_neighbors=5, weights="distance"),
        "svm": SVC(C=2.0, kernel="rbf", probability=True, class_weight="balanced", random_state=seed),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            random_state=seed,
            class_weight="balanced",
            n_jobs=-1,
            min_samples_leaf=2,
        ),
        "bagging": make_bagging(seed),
        "boosting": AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1, random_state=seed),
            n_estimators=150,
            learning_rate=0.5,
            random_state=seed,
        ),
    }
    return {
        name: Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", model),
            ]
        )
        for name, model in models.items()
    }


def select_base_models(seed: int, names: list[str]) -> dict[str, Pipeline]:
    models = base_models(seed)
    unknown = sorted(set(names) - set(models))
    if unknown:
        raise ValueError(f"Unknown base models: {unknown}. Available: {sorted(models)}")
    return {name: models[name] for name in names}


def meta_model(name: str, seed: int) -> Pipeline:
    if name == "random_forest":
        model = RandomForestClassifier(
            n_estimators=500,
            random_state=seed,
            class_weight="balanced",
            n_jobs=-1,
            min_samples_leaf=1,
        )
    elif name == "logistic_regression":
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed)
    else:
        raise ValueError("Unknown meta model. Use 'random_forest' or 'logistic_regression'.")
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )


def aggregation_functions(names: list[str]) -> dict[str, Callable[[np.ndarray], float]]:
    available: dict[str, Callable[[np.ndarray], float]] = {
        "mean": lambda values: float(np.mean(values)),
        "max": lambda values: float(np.max(values)),
        "min": lambda values: float(np.min(values)),
        "std": lambda values: float(np.std(values)),
        "median": lambda values: float(np.median(values)),
    }
    unknown = sorted(set(names) - set(available))
    if unknown:
        raise ValueError(f"Unknown aggregation functions: {unknown}")
    return {name: available[name] for name in names}


def positive_probabilities(model: Pipeline, x: np.ndarray) -> np.ndarray:
    probabilities = model.predict_proba(x)
    classes = list(model.named_steps["model"].classes_)
    positive_index = classes.index(1)
    return probabilities[:, positive_index]


def fit_loso_patient(
    patient: str,
    models: dict[str, Pipeline],
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
) -> tuple[str, np.ndarray, dict[str, np.ndarray]]:
    train_indices = np.flatnonzero(groups != patient)
    test_indices = np.flatnonzero(groups == patient)
    predictions: dict[str, np.ndarray] = {}
    if len(np.unique(y[train_indices])) < 2:
        return patient, test_indices, predictions
    for name, model in models.items():
        fitted = clone(model)
        fitted.fit(x[train_indices], y[train_indices])
        predictions[name] = positive_probabilities(fitted, x[test_indices])
    return patient, test_indices, predictions


def build_recording_frame(
    segment_frame: pd.DataFrame,
    prediction_columns: list[str],
    agg_names: list[str],
) -> pd.DataFrame:
    functions = aggregation_functions(agg_names)
    rows: list[dict[str, object]] = []
    for source_file, group in segment_frame.groupby("source_file", sort=True):
        row: dict[str, object] = {
            "source_file": source_file,
            "patient": str(group["patient"].iloc[0]),
            "split": group["split"].iloc[0],
            "label": int(group["label"].max()),
            "segments": int(len(group)),
        }
        for column in prediction_columns:
            values = group[column].to_numpy(dtype=np.float64)
            for agg_name, func in functions.items():
                row[f"{column}_{agg_name}"] = func(values)
        rows.append(row)
    return pd.DataFrame(rows)


def write_report(
    path: Path,
    title: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    extra: dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        file.write(f"{title}\n\n")
        for key, value in extra.items():
            file.write(f"{key}: {value}\n")
        file.write("\n")
        file.write(f"accuracy: {accuracy_score(y_true, y_pred):.4f}\n")
        file.write(f"balanced_accuracy: {balanced_accuracy_score(y_true, y_pred):.4f}\n\n")
        file.write("classification report:\n")
        file.write(classification_report(y_true, y_pred, target_names=["healthy", "heart_failure"], zero_division=0))
        file.write("\nconfusion matrix:\n")
        file.write(f"{confusion_matrix(y_true, y_pred)}\n")


def write_classification_graphs(
    output_prefix: Path,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: np.ndarray,
    class_names: list[str],
) -> None:
    plt = configure_matplotlib()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    matrix = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(np.arange(len(class_names)), labels=class_names)
    ax.set_yticks(np.arange(len(class_names)), labels=class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("OpenSmile Stacking Confusion Matrix")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            color = "white" if matrix[row, col] > matrix.max() / 2 else "#111827"
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center", color=color, fontsize=13)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    confusion_path = output_prefix.with_name(f"{output_prefix.name}_confusion_matrix.png")
    fig.savefig(confusion_path, dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, class_name, color in [(0, class_names[0], "#2563eb"), (1, class_names[1], "#dc2626")]:
        class_scores = scores[y_true == label]
        ax.hist(class_scores, bins=20, alpha=0.62, label=class_name, color=color, edgecolor="white")
    ax.axvline(0.5, color="#111827", linestyle="--", linewidth=1, label="threshold")
    ax.set_xlabel("Predicted unhealthy probability")
    ax.set_ylabel("Recordings")
    ax.set_title("OpenSmile Stacking Probability Distribution")
    ax.set_xlim(0.0, 1.0)
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    probability_path = output_prefix.with_name(f"{output_prefix.name}_probabilities.png")
    fig.savefig(probability_path, dpi=160)
    plt.close(fig)
    print(f"Saved graphs: {confusion_path}, {probability_path}")


def stack_feature_columns(recordings: pd.DataFrame, prediction_columns: list[str]) -> list[str]:
    return [
        column
        for column in recordings.columns
        if any(column.startswith(f"{prediction}_") for prediction in prediction_columns)
    ]


def load_feature_frame(features: Path) -> tuple[pd.DataFrame, list[str]]:
    frame = pd.read_csv(features)
    feature_columns = [column for column in frame.columns if column.startswith("smile_")]
    if not feature_columns:
        raise ValueError(f"No OpenSmile feature columns found in {features}")
    return frame, feature_columns


def train_stack(args: argparse.Namespace) -> None:
    frame, feature_columns = load_feature_frame(args.features)

    train_mask = frame["split"].eq("train")
    test_mask = frame["split"].isin(["test", "val"])
    if not train_mask.any() or not test_mask.any():
        recording_labels = frame.groupby("source_file", sort=True).agg(
            patient=("patient", "first"),
            label=("label", "max"),
        )
        patients = recording_labels["patient"].astype(str).drop_duplicates().to_numpy()
        patient_labels = (
            recording_labels.groupby(recording_labels["patient"].astype(str))["label"]
            .max()
            .reindex(patients)
            .to_numpy(dtype=np.int64)
        )
        train_patients, test_patients = train_test_split(
            patients,
            test_size=args.test_size,
            random_state=args.seed,
            stratify=patient_labels if len(np.unique(patient_labels)) == 2 else None,
        )
        train_mask = frame["patient"].astype(str).isin(set(train_patients))
        test_mask = frame["patient"].astype(str).isin(set(test_patients))
        print(
            f"Created patient-wise split: train_patients={len(train_patients)} "
            f"test_patients={len(test_patients)}",
            flush=True,
        )
    train_frame = frame.loc[train_mask].reset_index(drop=True)
    test_frame = frame.loc[test_mask].reset_index(drop=True)
    if train_frame.empty or test_frame.empty:
        raise ValueError("Both train and test/val splits are required for stacking.")

    x_train = train_frame[feature_columns].to_numpy(dtype=np.float64)
    y_train = train_frame["label"].to_numpy(dtype=np.int64)
    groups = train_frame["patient"].astype(str).to_numpy()
    x_test = test_frame[feature_columns].to_numpy(dtype=np.float64)
    y_test = test_frame["label"].to_numpy(dtype=np.int64)

    models = select_base_models(args.seed, args.base_models)
    oof_predictions = pd.DataFrame(index=train_frame.index)
    test_predictions = pd.DataFrame(index=test_frame.index)
    fitted_models: dict[str, Pipeline] = {}

    unique_groups = np.unique(groups)
    n_splits = min(args.oof_folds, len(unique_groups))
    if n_splits < 2:
        raise ValueError("Need at least two training patients for out-of-fold stacking.")
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=args.seed)

    for name, model in models.items():
        oof = np.zeros(len(train_frame), dtype=np.float64)
        for train_indices, val_indices in splitter.split(x_train, y_train, groups):
            fold_model = clone(model)
            fold_model.fit(x_train[train_indices], y_train[train_indices])
            oof[val_indices] = positive_probabilities(fold_model, x_train[val_indices])
        fitted = clone(model)
        fitted.fit(x_train, y_train)
        fitted_models[name] = fitted
        oof_predictions[f"{name}_prob"] = oof
        test_predictions[f"{name}_prob"] = positive_probabilities(fitted, x_test)

        segment_pred = (test_predictions[f"{name}_prob"].to_numpy() >= args.threshold).astype(int)
        print(
            f"{name}: segment balanced_acc={balanced_accuracy_score(y_test, segment_pred):.4f}",
            flush=True,
        )

    train_with_predictions = pd.concat([train_frame.reset_index(drop=True), oof_predictions], axis=1)
    test_with_predictions = pd.concat([test_frame.reset_index(drop=True), test_predictions], axis=1)
    prediction_columns = list(oof_predictions.columns)
    train_recordings = build_recording_frame(train_with_predictions, prediction_columns, args.agg)
    test_recordings = build_recording_frame(test_with_predictions, prediction_columns, args.agg)

    stack_columns = stack_feature_columns(train_recordings, prediction_columns)
    stacker = meta_model(args.meta_model, args.seed)
    stacker.fit(train_recordings[stack_columns], train_recordings["label"])
    recording_scores = positive_probabilities(stacker, test_recordings[stack_columns])
    recording_predictions = (recording_scores >= args.threshold).astype(int)

    args.model_dir.mkdir(parents=True, exist_ok=True)
    train_recordings.to_csv(args.model_dir / "recording_train_features.csv", index=False)
    test_recordings.assign(stacked_probability=recording_scores, prediction=recording_predictions).to_csv(
        args.model_dir / "recording_test_predictions.csv",
        index=False,
    )
    write_report(
        args.model_dir / "opensmile_stacking_evaluation.txt",
        "OpenSmile stacking evaluation",
        test_recordings["label"].to_numpy(dtype=np.int64),
        recording_predictions,
        {
            "features": args.features,
            "segment_rows_train": len(train_frame),
            "segment_rows_test": len(test_frame),
            "recordings_train": len(train_recordings),
            "recordings_test": len(test_recordings),
            "base_models": ", ".join(models),
            "meta_model": args.meta_model,
            "aggregations": ", ".join(args.agg),
            "threshold": args.threshold,
        },
    )
    write_classification_graphs(
        args.model_dir / "opensmile_stacking",
        test_recordings["label"].to_numpy(dtype=np.int64),
        recording_predictions,
        recording_scores,
        ["healthy", "unhealthy"],
    )
    joblib.dump(
        {
            "base_models": fitted_models,
            "stacker": stacker,
            "feature_columns": feature_columns,
            "stack_feature_columns": stack_columns,
            "aggregation_functions": args.agg,
            "threshold": args.threshold,
        },
        args.model_dir / "opensmile_stacking.joblib",
    )
    metadata = {
        "features": str(args.features),
        "model_dir": str(args.model_dir),
        "segment_feature_count": len(feature_columns),
        "stack_feature_count": len(stack_columns),
        "base_models": list(models),
        "meta_model": args.meta_model,
        "aggregations": args.agg,
    }
    (args.model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(classification_report(test_recordings["label"], recording_predictions, target_names=["healthy", "heart_failure"], zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(test_recordings["label"], recording_predictions))
    print(f"recording_acc={accuracy_score(test_recordings['label'], recording_predictions):.4f}")
    print(f"recording_balanced_acc={balanced_accuracy_score(test_recordings['label'], recording_predictions):.4f}")
    print(f"Saved model and reports to: {args.model_dir}")


def train_loso(args: argparse.Namespace) -> None:
    frame, feature_columns = load_feature_frame(args.features)
    models = select_base_models(args.seed, args.base_models)
    groups = frame["patient"].astype(str).to_numpy()
    unique_groups = np.array(sorted(np.unique(groups), key=lambda value: int(value) if value.isdigit() else value))
    x = frame[feature_columns].to_numpy(dtype=np.float64)
    y = frame["label"].to_numpy(dtype=np.int64)

    prediction_frame = frame[["source_file", "patient", "split", "label"]].copy()
    for name in models:
        prediction_frame[f"{name}_prob"] = np.nan

    cache_path = args.model_dir / "loso_segment_probabilities.csv"
    args.model_dir.mkdir(parents=True, exist_ok=True)
    if cache_path.exists() and not args.overwrite_loso_cache:
        cached = pd.read_csv(cache_path)
        if set(prediction_frame.columns).issubset(cached.columns):
            prediction_frame = cached[prediction_frame.columns].copy()
            print(f"Loaded cached LOSO segment probabilities from: {cache_path}")

    missing_patients = [
        patient
        for patient in unique_groups
        if prediction_frame.loc[prediction_frame["patient"].astype(str).eq(patient), f"{next(iter(models))}_prob"].isna().any()
    ]
    if args.max_loso_subjects is not None:
        missing_patients = missing_patients[: args.max_loso_subjects]
    if missing_patients:
        print(f"Computing LOSO segment probabilities for {len(missing_patients)} subjects...", flush=True)
        if args.loso_jobs == 1:
            results = []
            for group_index, patient in enumerate(missing_patients, start=1):
                patient_rows = int((groups == patient).sum())
                print(
                    f"[{group_index}/{len(missing_patients)}] fitting patient {patient} "
                    f"({patient_rows} held-out segments)",
                    flush=True,
                )
                result = fit_loso_patient(patient, models, x, y, groups)
                results.append(result)
                _, test_indices, predictions = result
                for name, values in predictions.items():
                    prediction_frame.loc[test_indices, f"{name}_prob"] = values
                prediction_frame.to_csv(cache_path, index=False)
                if group_index == 1 or group_index % args.log_every_loso == 0:
                    print(
                        f"[{group_index}/{len(missing_patients)}] cached LOSO probabilities for patient {patient}",
                        flush=True,
                    )
            print(f"Cached LOSO segment probabilities to: {cache_path}")
            results = []
        else:
            results = Parallel(n_jobs=args.loso_jobs, verbose=5)(
                delayed(fit_loso_patient)(patient, models, x, y, groups) for patient in missing_patients
            )
        for group_index, (patient, test_indices, predictions) in enumerate(results, start=1):
            for name, values in predictions.items():
                prediction_frame.loc[test_indices, f"{name}_prob"] = values
            if group_index == 1 or group_index % args.log_every_loso == 0:
                print(f"[{group_index}/{len(missing_patients)}] cached LOSO probabilities for patient {patient}", flush=True)
        if results:
            prediction_frame.to_csv(cache_path, index=False)
            print(f"Cached LOSO segment probabilities to: {cache_path}")

    if prediction_frame.filter(like="_prob").isna().any().any():
        if args.max_loso_subjects is not None:
            print(
                "Partial LOSO debug run complete; skipping recording-level report "
                "because not all subjects were evaluated.",
                flush=True,
            )
            return
        raise ValueError("Some LOSO probabilities were not generated.")

    prediction_columns = [f"{name}_prob" for name in models]
    recordings = build_recording_frame(prediction_frame, prediction_columns, args.agg)
    stack_columns = stack_feature_columns(recordings, prediction_columns)
    recording_groups = recordings["patient"].astype(str).to_numpy()
    recording_y = recordings["label"].to_numpy(dtype=np.int64)
    recording_scores = np.zeros(len(recordings), dtype=np.float64)
    recording_predictions = np.zeros(len(recordings), dtype=np.int64)

    for patient in unique_groups:
        train_mask = recording_groups != patient
        test_mask = recording_groups == patient
        if not test_mask.any() or len(np.unique(recording_y[train_mask])) < 2:
            continue
        stacker = meta_model(args.meta_model, args.seed)
        stacker.fit(recordings.loc[train_mask, stack_columns], recording_y[train_mask])
        scores = positive_probabilities(stacker, recordings.loc[test_mask, stack_columns])
        recording_scores[test_mask] = scores
        recording_predictions[test_mask] = (scores >= args.threshold).astype(int)

    args.model_dir.mkdir(parents=True, exist_ok=True)
    loso_predictions = recordings.assign(stacked_probability=recording_scores, prediction=recording_predictions)
    loso_predictions.to_csv(args.model_dir / "loso_recording_predictions.csv", index=False)
    write_report(
        args.model_dir / "opensmile_stacking_loso_evaluation.txt",
        "OpenSmile stacking LOSO evaluation",
        recording_y,
        recording_predictions,
        {
            "features": args.features,
            "recordings": len(recordings),
            "subjects": len(unique_groups),
            "base_models": ", ".join(models),
            "meta_model": args.meta_model,
            "aggregations": ", ".join(args.agg),
            "threshold": args.threshold,
        },
    )
    write_classification_graphs(
        args.model_dir / "opensmile_stacking_loso",
        recording_y,
        recording_predictions,
        recording_scores,
        ["healthy", "unhealthy"],
    )
    metadata = {
        "features": str(args.features),
        "model_dir": str(args.model_dir),
        "evaluation": "loso",
        "segment_feature_count": len(feature_columns),
        "stack_feature_count": len(stack_columns),
        "base_models": list(models),
        "meta_model": args.meta_model,
        "aggregations": args.agg,
    }
    (args.model_dir / "loso_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(classification_report(recording_y, recording_predictions, target_names=["healthy", "heart_failure"], zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(recording_y, recording_predictions))
    print(f"loso_recording_acc={accuracy_score(recording_y, recording_predictions):.4f}")
    print(f"loso_recording_balanced_acc={balanced_accuracy_score(recording_y, recording_predictions):.4f}")
    print(f"Saved LOSO report to: {args.model_dir / 'opensmile_stacking_loso_evaluation.txt'}")


def run(args: argparse.Namespace) -> None:
    if args.paper_preset == "full7":
        args.base_models = FULL_STACK_MODELS
        args.meta_model = "random_forest"
        args.agg = ["min", "max", "mean"]
    elif args.paper_preset == "optimal5":
        args.base_models = OPTIMAL_STACK_MODELS
        args.meta_model = "random_forest"
        args.agg = ["min", "max", "mean"]

    if args.extract or not args.features.exists():
        extract_features(args)
    if args.extract_only:
        return
    if args.evaluation == "loso":
        train_loso(args)
    else:
        train_stack(args)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the OpenSmile segment-model plus recording-level stacking pipeline."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--audio-dir", type=Path, default=DEFAULT_AUDIO_DIR)
    parser.add_argument("--features", "--output-features", dest="output_features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--feature-set", default="IS10")
    parser.add_argument("--feature-level", default="Functionals")
    parser.add_argument("--segment-mode", choices=["cycle", "fixed"], default="cycle")
    parser.add_argument("--window-seconds", type=float, default=1.0)
    parser.add_argument("--min-duration-seconds", type=float, default=0.25)
    parser.add_argument(
        "--base-models",
        nargs="+",
        default=OPTIMAL_STACK_MODELS,
        choices=["j48_tree", "naive_bayes", "knn", "svm", "random_forest", "bagging", "boosting"],
    )
    parser.add_argument("--meta-model", choices=["random_forest", "logistic_regression"], default="random_forest")
    parser.add_argument("--agg", nargs="+", default=["min", "max", "mean"], choices=["mean", "max", "min", "std", "median"])
    parser.add_argument("--paper-preset", choices=["none", "full7", "optimal5"], default="none")
    parser.add_argument("--evaluation", choices=["split", "loso"], default="split")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--oof-folds", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--log-every-loso", type=int, default=10)
    parser.add_argument("--loso-jobs", type=int, default=-1)
    parser.add_argument("--max-loso-subjects", type=int, default=None)
    parser.add_argument("--overwrite-loso-cache", action="store_true")
    parser.add_argument(
        "--include-augmented",
        dest="exclude_augmented",
        action="store_false",
        help="Include augmented manifest rows when extracting OpenSmile features.",
    )
    parser.set_defaults(exclude_augmented=True)
    parser.add_argument("--extract", action="store_true", help="Regenerate OpenSmile features even if the CSV exists.")
    parser.add_argument("--extract-only", action="store_true", help="Extract OpenSmile features and skip model training.")
    args = parser.parse_args()
    args.features = args.output_features
    run(args)


if __name__ == "__main__":
    main()
