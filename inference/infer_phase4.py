from pathlib import Path
import sys

import torch
import numpy as np
from PIL import Image


# ============================================================
# PROJECT ROOT
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT))


from src.models.siv_ai_phase4 import SIVAI
from src.data.kla_dataset import KLADataset


# ============================================================
# PATHS
# ============================================================

DATA_DIR = ROOT / "data" / "raw" / "train" / "train"

CHECKPOINT = (
    ROOT
    / "phase4_source"
    / "siv_ai_phase4_weights.pth"
)

OUTPUT_DIR = ROOT / "outputs" / "phase4"


# ============================================================
# TENSOR → IMAGE
# ============================================================

def tensor_to_image(tensor):

    tensor = tensor.detach().cpu()

    tensor = tensor.squeeze()

    tensor = tensor.clamp(0, 1)

    image = (
        tensor.numpy() * 255.0
    ).astype(np.uint8)

    return Image.fromarray(
        image,
        mode="L"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("SIV-AI PHASE 4 INFERENCE")
    print("=" * 70)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset = KLADataset(DATA_DIR)

    print(
        "Dataset size:",
        len(dataset)
    )

    # First sample
    sample = dataset[0]

    print(
        "Sample:",
        sample["name"]
    )

    print(
        "Noisy shape:",
        sample["noisy"].shape
    )

    print(
        "GT shape:",
        sample["gt"].shape
    )

    # --------------------------------------------------------
    # Prepare input
    # --------------------------------------------------------

    noisy = (
        sample["noisy"]
        .unsqueeze(0)
        .to(device)
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = SIVAI().to(device)

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        "Parameters:",
        parameter_count
    )

    if parameter_count != 110049:

        raise RuntimeError(
            f"Unexpected parameter count: "
            f"{parameter_count}"
        )

    # --------------------------------------------------------
    # Load Phase-4 weights
    # --------------------------------------------------------

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
        weights_only=False
    )

    # Clean state_dict
    model.load_state_dict(
        checkpoint
    )

    model.eval()

    print(
        "Phase-4 checkpoint loaded successfully"
    )

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    with torch.no_grad():

        prediction = model(noisy)

    print(
        "Prediction shape:",
        prediction.shape
    )

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Convert images
    # --------------------------------------------------------

    noisy_image = tensor_to_image(
        sample["noisy"]
    )

    gt_image = tensor_to_image(
        sample["gt"]
    )

    restored_image = tensor_to_image(
        prediction
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    noisy_path = (
        OUTPUT_DIR / "noisy.png"
    )

    gt_path = (
        OUTPUT_DIR / "ground_truth.png"
    )

    restored_path = (
        OUTPUT_DIR / "restored.png"
    )

    noisy_image.save(noisy_path)

    gt_image.save(gt_path)

    restored_image.save(restored_path)

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print()
    print("Images saved:")
    print("Noisy       :", noisy_path)
    print("Restored    :", restored_path)
    print("Ground Truth:", gt_path)

    print("=" * 70)


if __name__ == "__main__":

    main()