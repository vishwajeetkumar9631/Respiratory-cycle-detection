from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from train_cnn_classifier import create_model


def predict(model_path: Path, feature_path: Path, cpu: bool = False) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not cpu else "cpu")
    checkpoint = torch.load(model_path, map_location=device)
    class_names = checkpoint.get("class_names", ["healthy", "diseased"])
    architecture = checkpoint.get("architecture", "paper")
    input_channels = int(checkpoint.get("input_channels", checkpoint.get("input_shape", [1])[0]))
    diseased_threshold = float(checkpoint.get("diseased_threshold", 0.5))

    model = create_model(architecture, num_classes=len(class_names), input_channels=input_channels).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    feature = np.ascontiguousarray(np.load(feature_path), dtype=np.float32)
    if feature.ndim == 2:
        feature = feature[np.newaxis, :, :]
    elif feature.ndim != 3:
        raise ValueError(f"Expected 2D or 3D feature tensor, got shape {feature.shape}")
    tensor = torch.from_numpy(feature).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    if len(class_names) == 2:
        predicted_index = int(probabilities[1] >= diseased_threshold)
    else:
        predicted_index = int(np.argmax(probabilities))
    print(f"feature: {feature_path}")
    print(f"prediction: {class_names[predicted_index]}")
    if len(class_names) == 2:
        print(f"diseased_threshold: {diseased_threshold:.2f}")
    for class_name, probability in zip(class_names, probabilities):
        print(f"{class_name}: {probability:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict healthy/diseased for one mel feature.")
    parser.add_argument("--model", type=Path, default=Path("models") / "compact_cnn.pt")
    parser.add_argument(
        "--feature",
        type=Path,
        default=Path("mel_dataset") / "features" / "211_1p3_Ar_mc_AKGC417L_cycle_001.npy",
    )
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    predict(args.model, args.feature, args.cpu)


if __name__ == "__main__":
    main()
