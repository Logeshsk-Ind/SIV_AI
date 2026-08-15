from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity


# --------------------------------------------------
# Add project root to Python path
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


from src.data.kla_dataset import KLADataset
from models.baseline_sr import BaselineSR


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DATA_DIR = ROOT / "data" / "raw" / "train" / "train"

# IMPORTANT:
# This file evaluates the Day 2 checkpoint.
CHECKPOINT = ROOT / "checkpoints" / "baseline_day3_best.pth"

BATCH_SIZE = 1
SEED = 42
VAL_RATIO = 0.2


# --------------------------------------------------
# Main evaluation
# --------------------------------------------------

def main():

    # --------------------------------------------------
    # Device
    # --------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))


    # --------------------------------------------------
    # Load complete dataset
    # --------------------------------------------------

    dataset = KLADataset(DATA_DIR)

    print("Total samples:", len(dataset))


    # --------------------------------------------------
    # Reproduce the same validation split
    # --------------------------------------------------

    generator = torch.Generator().manual_seed(SEED)

    indices = torch.randperm(
        len(dataset),
        generator=generator
    ).tolist()

    val_size = int(len(dataset) * VAL_RATIO)

    val_indices = indices[-val_size:]

    validation_dataset = Subset(
        dataset,
        val_indices
    )

    print("Validation samples:", len(validation_dataset))


    # --------------------------------------------------
    # DataLoader
    # --------------------------------------------------

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )


    # --------------------------------------------------
    # Create model
    # --------------------------------------------------

    model = BaselineSR().to(device)


    # --------------------------------------------------
    # Load Day 2 checkpoint
    # --------------------------------------------------

    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT}"
        )

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
    # Evaluation
    # --------------------------------------------------

    total_psnr = 0.0
    total_ssim = 0.0

    processed = 0

    with torch.no_grad():

        for batch in validation_loader:

            noisy = batch["noisy"].to(device)
            gt = batch["gt"].to(device)

            prediction = model(noisy)

            # Remove batch/channel dimensions
            prediction = prediction.squeeze(0).squeeze(0)
            gt = gt.squeeze(0).squeeze(0)

            prediction = prediction.cpu().numpy()
            gt = gt.cpu().numpy()

            # Keep prediction in valid image range
            prediction = np.clip(
                prediction,
                0.0,
                1.0
            )

            # PSNR
            psnr = peak_signal_noise_ratio(
                gt,
                prediction,
                data_range=1.0
            )

            # SSIM
            ssim = structural_similarity(
                gt,
                prediction,
                data_range=1.0
            )

            total_psnr += psnr
            total_ssim += ssim

            processed += 1

            if processed % 200 == 0:
                print(
                    f"Processed {processed}/"
                    f"{len(validation_dataset)}"
                )


    # --------------------------------------------------
    # Calculate mean metrics
    # --------------------------------------------------

    mean_psnr = total_psnr / len(validation_dataset)
    mean_ssim = total_ssim / len(validation_dataset)


    # --------------------------------------------------
    # Final results
    # --------------------------------------------------

    print()
    print("Day 3 Baseline Dataset Evaluation")
    print("----------------------------------")
    print(f"Validation samples: {len(val_indices)}")
    print(f"Mean PSNR: {mean_psnr:.4f} dB")
    print(f"Mean SSIM: {mean_ssim:.4f}")


# --------------------------------------------------
# Entry point
# --------------------------------------------------

if __name__ == "__main__":
    main()