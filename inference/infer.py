from pathlib import Path
import sys

import torch
from PIL import Image
import numpy as np


# Add project root to Python path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.kla_dataset import KLADataset
from models.baseline_sr import BaselineSR


# --------------------------------------------------
# Paths
# --------------------------------------------------

DATA_DIR = ROOT / "data" / "raw" / "train" / "train"
CHECKPOINT = ROOT / "checkpoints" / "baseline_day2_best.pth"
OUTPUT_DIR = ROOT / "outputs" / "day2_baseline"


def tensor_to_image(tensor):
    """
    Convert a [1, H, W] tensor into a grayscale PIL image.
    """

    tensor = tensor.detach().cpu().squeeze(0)

    tensor = tensor.clamp(0, 1)

    image = (tensor.numpy() * 255).astype(np.uint8)

    return Image.fromarray(image, mode="L")


def main():

    # --------------------------------------------------
    # Device
    # --------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    # --------------------------------------------------
    # Load dataset
    # --------------------------------------------------

    dataset = KLADataset(DATA_DIR)

    print("Dataset size:", len(dataset))

    # Use the first sample
    sample = dataset[0]

    noisy = sample["noisy"].unsqueeze(0).to(device)
    gt = sample["gt"]

    print("Sample name:", sample["name"])
    print("Input shape:", noisy.shape)
    print("GT shape:", gt.shape)

    # --------------------------------------------------
    # Load model
    # --------------------------------------------------

    model = BaselineSR().to(device)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print("Checkpoint loaded successfully")

    # --------------------------------------------------
    # Inference
    # --------------------------------------------------

    with torch.no_grad():

        prediction = model(noisy)

    prediction = prediction.squeeze(0)

    print("Prediction shape:", prediction.shape)

    # --------------------------------------------------
    # Save images
    # --------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    noisy_image = tensor_to_image(
        sample["noisy"]
    )

    gt_image = tensor_to_image(
        sample["gt"]
    )

    restored_image = tensor_to_image(
        prediction
    )

    noisy_path = OUTPUT_DIR / "noisy.png"
    gt_path = OUTPUT_DIR / "ground_truth.png"
    restored_path = OUTPUT_DIR / "restored.png"

    noisy_image.save(noisy_path)
    gt_image.save(gt_path)
    restored_image.save(restored_path)

    print()
    print("Images saved:")
    print("Noisy       :", noisy_path)
    print("Restored    :", restored_path)
    print("Ground Truth:", gt_path)


if __name__ == "__main__":
    main()