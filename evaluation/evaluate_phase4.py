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

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ============================================================
# IMPORTS
# ============================================================

from src.data.kla_dataset import KLADataset
from src.models.siv_ai_phase4 import SIVAI


# ============================================================
# CONFIG
# ============================================================

DATA_DIR = ROOT / "data" / "raw" / "train" / "train"

CHECKPOINT = (
    ROOT
    / "phase4_source"
    / "siv_ai_phase4_weights.pth"
)

BATCH_SIZE = 4

VAL_RATIO = 0.20

SEED = 42


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("SIV-AI PHASE 4 DATASET EVALUATION")
    print("=" * 70)

    # --------------------------------------------------------
    # DEVICE
    # --------------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    print()
    print("Dataset:")
    print(DATA_DIR)

    dataset = KLADataset(DATA_DIR)

    print("Total samples:", len(dataset))

    if len(dataset) == 0:
        raise RuntimeError(
            "Dataset is empty. Check DATA_DIR."
        )

    # --------------------------------------------------------
    # DETERMINISTIC VALIDATION SPLIT
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

    val_dataset = Subset(
        dataset,
        val_indices
    )

    print(
        "Validation samples:",
        len(val_dataset)
    )

    # --------------------------------------------------------
    # DATALOADER
    # --------------------------------------------------------

    loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    print()
    print("Creating Phase-4 model...")

    model = SIVAI().to(device)

    params = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        "Model parameters:",
        params
    )

    assert params == 110049, (
        f"Wrong architecture! "
        f"Expected 110049, got {params}"
    )

    # --------------------------------------------------------
    # CHECKPOINT
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

    model.load_state_dict(
        state_dict,
        strict=True
    )

    print(
        "Checkpoint loaded successfully."
    )

    # --------------------------------------------------------
    # EVALUATION
    # --------------------------------------------------------

    model.eval()

    psnr_values = []
    ssim_values = []

    global_mse_sum = 0.0
    global_pixel_count = 0

    # --------------------------------------------------------
    # INFERENCE
    # --------------------------------------------------------

    with torch.no_grad():

        for batch_idx, batch in enumerate(loader):

            noisy = batch["noisy"].to(
                device,
                non_blocking=True
            )

            gt = batch["gt"].to(
                device,
                non_blocking=True
            )

            # ------------------------------------------------
            # MODEL
            # ------------------------------------------------

            prediction = model(noisy)

            # ------------------------------------------------
            # CLAMP FOR METRICS
            # ------------------------------------------------

            prediction = prediction.clamp(
                0.0,
                1.0
            )

            # ------------------------------------------------
            # GLOBAL MSE
            # ------------------------------------------------

            diff = prediction - gt

            global_mse_sum += (
                diff.pow(2)
                .sum()
                .item()
            )

            global_pixel_count += (
                gt.numel()
            )

            # ------------------------------------------------
            # CPU FOR SSIM / PSNR
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
            # PER IMAGE METRICS
            # ------------------------------------------------

            for pred, target in zip(
                prediction_np,
                gt_np
            ):

                pred_image = pred[0]

                target_image = target[0]

                psnr = peak_signal_noise_ratio(
                    target_image,
                    pred_image,
                    data_range=1.0
                )

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

            # ------------------------------------------------
            # PROGRESS
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
    # RESULTS
    # ========================================================

    mean_psnr = float(
        np.mean(psnr_values)
    )

    mean_ssim = float(
        np.mean(ssim_values)
    )

    global_mse = (
        global_mse_sum
        / global_pixel_count
    )

    global_psnr = (
        10.0
        * np.log10(
            1.0 / global_mse
        )
    )

    # ========================================================
    # PRINT
    # ========================================================

    print()
    print("=" * 70)
    print("SIV-AI PHASE 4 RESULTS")
    print("=" * 70)

    print(
        f"Validation samples : {len(val_dataset)}"
    )

    print(
        f"Mean PSNR           : {mean_psnr:.4f} dB"
    )

    print(
        f"Global PSNR         : {global_psnr:.4f} dB"
    )

    print(
        f"Mean SSIM           : {mean_ssim:.4f}"
    )

    print(
        f"Global MSE          : {global_mse:.8f}"
    )

    print("=" * 70)

    # ========================================================
    # COMPARE WITH COLAB CHECKPOINT METADATA
    # ========================================================

    print()
    print("COLAB PHASE-4 REFERENCE")
    print("-" * 70)

    print(
        "Best Global PSNR : 25.142849"
    )

    print(
        "Best Mean PSNR   : 27.873514"
    )

    print(
        "Best SSIM        : 0.761343"
    )

    print()
    print("VS CODE EVALUATION")
    print("-" * 70)

    print(
        f"Mean PSNR        : {mean_psnr:.6f}"
    )

    print(
        f"Global PSNR      : {global_psnr:.6f}"
    )

    print(
        f"SSIM             : {mean_ssim:.6f}"
    )

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()