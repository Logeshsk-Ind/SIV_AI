from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity


# ============================================================
# Project root
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ============================================================
# Project imports
# ============================================================

from src.data.kla_dataset import KLADataset
from src.models.wavelet_sr import WaveletSR


# ============================================================
# Paths
# ============================================================

DATA_DIR = (
    ROOT
    / "data"
    / "raw"
    / "train"
    / "train"
)

CHECKPOINT = (
    ROOT
    / "checkpoints"
    / "wavelet_sr_best.pth"
)


# ============================================================
# Configuration
# ============================================================

BATCH_SIZE = 1
SEED = 42
VAL_RATIO = 0.20


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset = KLADataset(DATA_DIR)

    print(
        "Total samples:",
        len(dataset)
    )


    # --------------------------------------------------------
    # Reproduce exact validation split
    # --------------------------------------------------------

    generator = torch.Generator().manual_seed(SEED)

    indices = torch.randperm(
        len(dataset),
        generator=generator
    ).tolist()

    val_size = int(
        len(dataset) * VAL_RATIO
    )

    val_indices = indices[-val_size:]

    validation_dataset = Subset(
        dataset,
        val_indices
    )

    print(
        "Validation samples:",
        len(validation_dataset)
    )


    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )


    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = WaveletSR().to(device)


    # --------------------------------------------------------
    # Checkpoint
    # --------------------------------------------------------

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


    print(
        "Checkpoint loaded successfully"
    )

    print(
        "Checkpoint epoch:",
        checkpoint["epoch"]
    )

    print(
        "Checkpoint validation loss:",
        checkpoint["val_loss"]
    )


    # --------------------------------------------------------
    # Parameter count
    # --------------------------------------------------------

    parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        "Model parameters:",
        parameters
    )


    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    total_psnr = 0.0
    total_ssim = 0.0

    processed = 0


    with torch.no_grad():

        for batch in validation_loader:

            noisy = batch["noisy"].to(device)

            gt = batch["gt"].to(device)


            # Forward pass

            prediction = model(noisy)


            # Keep image range valid

            prediction = prediction.clamp(
                0.0,
                1.0
            )


            # Remove batch/channel dimensions

            prediction_np = (
                prediction[0, 0]
                .cpu()
                .numpy()
            )

            gt_np = (
                gt[0, 0]
                .cpu()
                .numpy()
            )


            # ------------------------------------------------
            # PSNR
            # ------------------------------------------------

            psnr = peak_signal_noise_ratio(
                gt_np,
                prediction_np,
                data_range=1.0
            )


            # ------------------------------------------------
            # SSIM
            # ------------------------------------------------

            ssim = structural_similarity(
                gt_np,
                prediction_np,
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


    # --------------------------------------------------------
    # Mean metrics
    # --------------------------------------------------------

    mean_psnr = (
        total_psnr
        / len(validation_dataset)
    )

    mean_ssim = (
        total_ssim
        / len(validation_dataset)
    )


    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    print()

    print(
        "SIV-AI WaveletSR Dataset Evaluation"
    )

    print(
        "------------------------------------"
    )

    print(
        f"Validation samples: "
        f"{len(validation_dataset)}"
    )

    print(
        f"Model parameters: "
        f"{parameters}"
    )

    print(
        f"Mean PSNR: "
        f"{mean_psnr:.4f} dB"
    )

    print(
        f"Mean SSIM: "
        f"{mean_ssim:.4f}"
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()