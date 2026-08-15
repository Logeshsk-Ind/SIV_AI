from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity


# ============================================================
# PROJECT ROOT
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT)
)


# ============================================================
# IMPORTS
# ============================================================

from src.data.kla_dataset import KLADataset
from src.models.siv_ai_phase4 import SIVAI


# ============================================================
# CONFIGURATION
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
    / "phase4_source"
    / "siv_ai_phase4_weights.pth"
)

SEED = 42
VAL_RATIO = 0.20

BATCH_SIZE = 1


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("SIV-AI PHASE 4 EXACT EVALUATION")
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

    if device.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    print()
    print("Dataset:")
    print(DATA_DIR)

    dataset = KLADataset(DATA_DIR)

    print(
        "Total samples:",
        len(dataset)
    )

    # --------------------------------------------------------
    # EXACT SPLIT
    # --------------------------------------------------------

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

    train_indices = indices[:-val_size]

    val_indices = indices[-val_size:]

    print(
        "Training samples:",
        len(train_indices)
    )

    print(
        "Validation samples:",
        len(val_indices)
    )

    # --------------------------------------------------------
    # Validation dataset
    # --------------------------------------------------------

    val_dataset = Subset(
        dataset,
        val_indices
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print()
    print("Creating Phase-4 model...")

    model = SIVAI().to(device)

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        "Model parameters:",
        parameter_count
    )

    assert parameter_count == 110049

    # --------------------------------------------------------
    # Load weights
    # --------------------------------------------------------

    print()
    print("Loading checkpoint:")
    print(CHECKPOINT)

    state_dict = torch.load(
        CHECKPOINT,
        map_location=device
    )

    print(
        "Checkpoint tensors:",
        len(state_dict)
    )

    result = model.load_state_dict(
        state_dict,
        strict=True
    )

    print(
        "Checkpoint loaded successfully."
    )

    print(result)

    # --------------------------------------------------------
    # Evaluation mode
    # --------------------------------------------------------

    model.eval()

    # --------------------------------------------------------
    # Metric storage
    # --------------------------------------------------------

    psnr_values = []
    ssim_values = []

    total_squared_error = 0.0
    total_pixels = 0

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    with torch.no_grad():

        for batch_idx, batch in enumerate(
            val_loader
        ):

            noisy = batch["noisy"].to(
                device,
                non_blocking=True
            )

            gt = batch["gt"].to(
                device,
                non_blocking=True
            )

            # ------------------------------------------------
            # Forward
            # ------------------------------------------------

            prediction = model(
                noisy
            )

            # ------------------------------------------------
            # Clamp prediction
            # ------------------------------------------------

            prediction = prediction.clamp(
                0.0,
                1.0
            )

            # ------------------------------------------------
            # Move to CPU
            # ------------------------------------------------

            prediction_np = (
                prediction
                .cpu()
                .numpy()
            )

            gt_np = (
                gt
                .cpu()
                .numpy()
            )

            # ------------------------------------------------
            # Per-image metrics
            # ------------------------------------------------

            for pred, target in zip(
                prediction_np,
                gt_np
            ):

                pred_image = pred[0]

                target_image = target[0]

                # PSNR
                psnr = peak_signal_noise_ratio(
                    target_image,
                    pred_image,
                    data_range=1.0
                )

                # SSIM
                ssim = structural_similarity(
                    target_image,
                    pred_image,
                    data_range=1.0
                )

                psnr_values.append(
                    psnr
                )

                ssim_values.append(
                    ssim
                )

                # Global MSE
                diff = (
                    pred_image
                    - target_image
                )

                total_squared_error += float(
                    np.sum(
                        diff * diff
                    )
                )

                total_pixels += (
                    diff.size
                )

            # ------------------------------------------------
            # Progress
            # ------------------------------------------------

            processed = min(
                (batch_idx + 1) * BATCH_SIZE,
                len(val_dataset)
            )

            if processed % 100 == 0:

                print(
                    f"Processed "
                    f"{processed}/"
                    f"{len(val_dataset)}"
                )

    # ========================================================
    # FINAL METRICS
    # ========================================================

    mean_psnr = float(
        np.mean(psnr_values)
    )

    mean_ssim = float(
        np.mean(ssim_values)
    )

    global_mse = (
        total_squared_error
        / total_pixels
    )

    global_psnr = (
        -10.0
        * np.log10(global_mse)
    )

    # ========================================================
    # RESULTS
    # ========================================================

    print()
    print("=" * 70)
    print("SIV-AI PHASE 4 EXACT RESULTS")
    print("=" * 70)

    print(
        f"Validation samples : "
        f"{len(val_dataset)}"
    )

    print(
        f"Mean PSNR           : "
        f"{mean_psnr:.6f} dB"
    )

    print(
        f"Global PSNR         : "
        f"{global_psnr:.6f} dB"
    )

    print(
        f"Mean SSIM           : "
        f"{mean_ssim:.6f}"
    )

    print(
        f"Global MSE          : "
        f"{global_mse:.8f}"
    )

    print()
    print("=" * 70)
    print("COLAB REFERENCE")
    print("=" * 70)

    print(
        "Best Global PSNR : "
        "25.142849"
    )

    print(
        "Best Mean PSNR   : "
        "27.873514"
    )

    print(
        "Best SSIM        : "
        "0.761343"
    )

    print()
    print("=" * 70)
    print("DIFFERENCE")
    print("=" * 70)

    print(
        f"Mean PSNR difference : "
        f"{mean_psnr - 27.873514:+.6f} dB"
    )

    print(
        f"Global PSNR difference : "
        f"{global_psnr - 25.142849:+.6f} dB"
    )

    print(
        f"SSIM difference : "
        f"{mean_ssim - 0.761343:+.6f}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()