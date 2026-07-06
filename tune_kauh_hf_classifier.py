from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

import train_cnn_classifier as trainer


TRIALS = [
    {
        "lr": 3e-4,
        "weight_decay": 5e-3,
        "label_smoothing": 0.10,
        "dropout": 0.65,
        "batch_size": 64,
        "freq_mask": 8,
        "time_mask": 16,
    },
    {
        "lr": 1e-4,
        "weight_decay": 5e-3,
        "label_smoothing": 0.10,
        "dropout": 0.65,
        "batch_size": 64,
        "freq_mask": 8,
        "time_mask": 16,
    },
    {
        "lr": 2e-4,
        "weight_decay": 1e-2,
        "label_smoothing": 0.15,
        "dropout": 0.70,
        "batch_size": 64,
        "freq_mask": 10,
        "time_mask": 20,
    },
    {
        "lr": 1e-4,
        "weight_decay": 1e-3,
        "label_smoothing": 0.05,
        "dropout": 0.55,
        "batch_size": 32,
        "freq_mask": 6,
        "time_mask": 12,
    },
    {
        "lr": 2e-4,
        "weight_decay": 5e-3,
        "label_smoothing": 0.05,
        "dropout": 0.60,
        "batch_size": 32,
        "freq_mask": 8,
        "time_mask": 16,
    },
]


def build_train_args(
    cli_args: argparse.Namespace,
    config: dict[str, float | int],
    output: Path,
    epochs: int,
    patience: int,
    seed: int,
) -> argparse.Namespace:
    return argparse.Namespace(
        data_dir=cli_args.data_dir,
        manifest=None,
        train_manifest=cli_args.train_manifest,
        val_manifest=cli_args.val_manifest,
        output=output,
        history_csv=output.with_name(f"{output.stem}_history.csv"),
        history_plot=output.with_name(f"{output.stem}_history.png"),
        eval_report=output.with_name(f"{output.stem}_evaluation.txt"),
        epochs=epochs,
        batch_size=int(config["batch_size"]),
        lr=float(config["lr"]),
        weight_decay=float(config["weight_decay"]),
        label_smoothing=float(config["label_smoothing"]),
        scheduler="plateau",
        lr_patience=3,
        lr_factor=0.5,
        min_lr=1e-6,
        grad_clip=1.0,
        dropout=float(config["dropout"]),
        specaugment=True,
        freq_mask=int(config["freq_mask"]),
        time_mask=int(config["time_mask"]),
        architecture="paper",
        save_metric="val-balanced-acc",
        class_weight_mode="balanced",
        diseased_threshold=0.5,
        auto_threshold=True,
        min_class_zero_recall=cli_args.min_healthy_recall,
        val_size=0.2,
        min_cycle_duration=0.35,
        max_cycle_duration=6.0,
        split_candidates=1,
        patience=patience,
        min_delta=1e-4,
        seed=seed,
        cpu=cli_args.cpu,
        include_augmented_val=False,
    )


def write_results(path: Path, rows: list[dict[str, float | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tune compact CNN hyperparameters for the KAUH HF dataset."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("kauh_hf_study_dataset_bandlimited"))
    parser.add_argument(
        "--train-manifest",
        type=Path,
        default=Path("kauh_hf_study_dataset_bandlimited") / "train_manifest.csv",
    )
    parser.add_argument(
        "--val-manifest",
        type=Path,
        default=Path("kauh_hf_study_dataset_bandlimited") / "test_manifest.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("models") / "kauh_hf_tuning")
    parser.add_argument("--trial-epochs", type=int, default=30)
    parser.add_argument("--final-epochs", type=int, default=100)
    parser.add_argument("--min-healthy-recall", type=float, default=0.55)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, float | int]] = []

    for trial_index, config in enumerate(TRIALS, start=1):
        print(f"\nHyperparameter trial {trial_index}/{len(TRIALS)}: {config}", flush=True)
        train_args = build_train_args(
            args,
            config,
            args.output_dir / f"trial_{trial_index:02d}.pt",
            args.trial_epochs,
            patience=10,
            seed=args.seed + trial_index,
        )
        torch.manual_seed(train_args.seed)
        np.random.seed(train_args.seed)
        metrics = trainer.train(train_args)
        ranking_score = (
            0.7 * metrics["val_balanced_acc"]
            + 0.3 * metrics["patient_balanced_acc"]
        )
        row: dict[str, float | int] = {
            "trial": trial_index,
            **config,
            **metrics,
            "ranking_score": ranking_score,
        }
        results.append(row)
        write_results(args.output_dir / "tuning_results.csv", results)

    best = max(results, key=lambda row: float(row["ranking_score"]))
    best_index = int(best["trial"])
    best_config = TRIALS[best_index - 1]
    print(f"\nBest trial: {best_index}, retraining for {args.final_epochs} epochs.", flush=True)
    final_args = build_train_args(
        args,
        best_config,
        args.output_dir / "best_model_100.pt",
        args.final_epochs,
        patience=20,
        seed=args.seed,
    )
    torch.manual_seed(final_args.seed)
    np.random.seed(final_args.seed)
    trainer.train(final_args)


if __name__ == "__main__":
    main()
