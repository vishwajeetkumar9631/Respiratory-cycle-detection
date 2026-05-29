from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from train_cnn_classifier import CompactRespiratoryCNN


def predict(model_path: Path, feature_path: Path, cpu: bool = False) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not cpu else "cpu")
    checkpoint = torch.load(model_path, map_location=device)
    class_names = checkpoint.get("class_names", ["healthy", "diseased"])
    diseased_threshold = float(checkpoint.get("diseased_threshold", 0.5))

    model = CompactRespiratoryCNN(num_classes=len(class_names)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    feature = np.load(feature_path).astype(np.float32)
    tensor = torch.from_numpy(feature).unsqueeze(0).unsqueeze(0).to(device)
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
