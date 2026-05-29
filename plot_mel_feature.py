from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_FEATURE = Path("mel_dataset") / "features" / "211_1p3_Ar_mc_AKGC417L_cycle_003.npy"


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot a saved .npy mel-spectrogram feature.")
    parser.add_argument(
        "--feature",
        type=Path,
        default=DEFAULT_FEATURE,
        help=f"Saved .npy feature path. Default: {DEFAULT_FEATURE}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("plots") / f"{DEFAULT_FEATURE.stem}_mel.png",
        help="Output PNG path.",
    )
    args = parser.parse_args()
    plot_mel_feature(args.feature, args.output)


if __name__ == "__main__":
    main()
