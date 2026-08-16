from pathlib import Path
import sys

import torch
from PIL import Image
import numpy as np


# --------------------------------------------------
# Add project root to Python path
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


from src.data.kla_dataset import KLADataset
from models.residual_sr import ResidualSR


# --------------------------------------------------
# Paths
# --------------------------------------------------

DATA_DIR = ROOT / "data" / "raw" / "train" / "train"

CHECKPOINT = (
    ROOT
    / "checkpoints"
    / "residual_day4_best.pth"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "day5_final"
)


# --------------------------------------------------
# Tensor → PIL image
# --------------------------------------------------

def tensor_to_image(tensor):
    """
    Convert [1, H, W] tensor to grayscale PIL image.
    """

    tensor = tensor.detach().cpu().squeeze(0)

    tensor = tensor.clamp(0, 1)

    image = (
        tensor.numpy() * 255
    ).astype(np.uint8)

    return Image.fromarray(
        image,
        mode="L"
    )


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    # --------------------------------------------------
    # Device
    # --------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    # --------------------------------------------------
    # Dataset
    # --------------------------------------------------

    dataset = KLADataset(DATA_DIR)

    print(
        "Dataset size:",
        len(dataset)
    )

    # Use first sample for final demonstration
    sample = dataset[0]

    noisy = (
        sample["noisy"]
        .unsqueeze(0)
        .to(device)
    )

    gt = sample["gt"]

    print(
        "Sample name:",
        sample["name"]
    )

    print(
        "Input shape:",
        noisy.shape
    )

    print(
        "Ground truth shape:",
        gt.shape
    )

    # --------------------------------------------------
    # Create ResidualSR model
    # --------------------------------------------------

    model = ResidualSR().to(device)

    # --------------------------------------------------
    # Load Day 4 checkpoint
    # --------------------------------------------------

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print(
        "ResidualSR checkpoint loaded successfully"
    )

    # --------------------------------------------------
    # Inference
    # --------------------------------------------------

    with torch.no_grad():

        prediction = model(noisy)

    prediction = prediction.squeeze(0)

    print(
        "Prediction shape:",
        prediction.shape
    )

    # --------------------------------------------------
    # Create output directory
    # --------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------
    # Convert images
    # --------------------------------------------------

    noisy_image = tensor_to_image(
        sample["noisy"]
    )

    restored_image = tensor_to_image(
        prediction
    )

    gt_image = tensor_to_image(
        gt
    )

    # --------------------------------------------------
    # Save images
    # --------------------------------------------------

    noisy_path = (
        OUTPUT_DIR / "noisy.png"
    )

    restored_path = (
        OUTPUT_DIR / "restored.png"
    )

    gt_path = (
        OUTPUT_DIR / "ground_truth.png"
    )

    noisy_image.save(noisy_path)
    restored_image.save(restored_path)
    gt_image.save(gt_path)

    # --------------------------------------------------
    # Results
    # --------------------------------------------------

    print()
    print("Final images saved:")
    print(
        "Noisy       :",
        noisy_path
    )

    print(
        "Restored    :",
        restored_path
    )

    print(
        "Ground Truth:",
        gt_path
    )


# --------------------------------------------------
# Entry point
# --------------------------------------------------

if __name__ == "__main__":
    main()