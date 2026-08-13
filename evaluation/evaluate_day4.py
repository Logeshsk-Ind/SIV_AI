from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity


# --------------------------------------------------
# Project root
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT)
)


from src.data.kla_dataset import KLADataset
from models.residual_sr import ResidualSR


# --------------------------------------------------
# Paths
# --------------------------------------------------

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
    / "residual_day4_best.pth"
)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

SEED = 42
VAL_RATIO = 0.2

BATCH_SIZE = 1


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
        "Total samples:",
        len(dataset)
    )

    # --------------------------------------------------
    # Same validation split as Day 2 / Day 3
    # --------------------------------------------------

    generator = torch.Generator().manual_seed(
        SEED
    )

    indices = torch.randperm(
        len(dataset),
        generator=generator
    ).tolist()

    val_size = int(
        len(dataset) * VAL_RATIO
    )

    val_indices = indices[-val_size:]

    print(
        "Validation samples:",
        len(val_indices)
    )

    val_dataset = Subset(
        dataset,
        val_indices
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    model = ResidualSR().to(device)

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

    # --------------------------------------------------
    # Metrics
    # --------------------------------------------------

    psnr_values = []
    ssim_values = []

    # --------------------------------------------------
    # Evaluation
    # --------------------------------------------------

    with torch.no_grad():

        for batch_idx, batch in enumerate(
            val_loader
        ):

            noisy = batch["noisy"].to(
                device
            )

            gt = batch["gt"].to(
                device
            )

            prediction = model(
                noisy
            )

            prediction = prediction.clamp(
                0,
                1
            )

            # Convert to NumPy
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

            # PSNR
            psnr = peak_signal_noise_ratio(
                gt_np,
                prediction_np,
                data_range=1.0
            )

            # SSIM
            ssim = structural_similarity(
                gt_np,
                prediction_np,
                data_range=1.0
            )

            psnr_values.append(psnr)
            ssim_values.append(ssim)

            if (
                (batch_idx + 1) % 200
                == 0
            ):

                print(
                    f"Processed "
                    f"{batch_idx + 1}/"
                    f"{len(val_loader)}"
                )

    # --------------------------------------------------
    # Mean metrics
    # --------------------------------------------------

    mean_psnr = float(
        np.mean(psnr_values)
    )

    mean_ssim = float(
        np.mean(ssim_values)
    )

    # --------------------------------------------------
    # Results
    # --------------------------------------------------

    print()

    print(
        "Day 4 Residual Dataset Evaluation"
    )

    print(
        "----------------------------------"
    )

    print(
        f"Validation samples: "
        f"{len(val_indices)}"
    )

    print(
        f"Mean PSNR: "
        f"{mean_psnr:.4f} dB"
    )

    print(
        f"Mean SSIM: "
        f"{mean_ssim:.4f}"
    )


# --------------------------------------------------
# Entry point
# --------------------------------------------------

if __name__ == "__main__":
    main()