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

sys.path.insert(0, str(ROOT))


from src.data.kla_dataset import KLADataset
from src.models.siv_ai_phase4 import SIVAI


# ============================================================
# CONFIG
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

BATCH_SIZE = 1

SEED = 42
VAL_RATIO = 0.20


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("SIV-AI PHASE 4 PER-IMAGE DIAGNOSTIC")
    print("=" * 70)

    # --------------------------------------------------------
    # DEVICE
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
    # DATASET
    # --------------------------------------------------------

    dataset = KLADataset(DATA_DIR)

    print()
    print("Dataset:", DATA_DIR)
    print("Total samples:", len(dataset))

    # --------------------------------------------------------
    # SAME VALIDATION SPLIT
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

    print("Validation samples:", len(val_dataset))

    # --------------------------------------------------------
    # DATALOADER
    # --------------------------------------------------------

    loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
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

    print("Parameters:", params)

    assert params == 110049

    # --------------------------------------------------------
    # LOAD CHECKPOINT
    # --------------------------------------------------------

    print()
    print("Loading checkpoint:")
    print(CHECKPOINT)

    state_dict = torch.load(
        CHECKPOINT,
        map_location=device
    )

    model.load_state_dict(
        state_dict,
        strict=True
    )

    model.eval()

    print("Checkpoint loaded successfully.")

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    psnr_values = []
    ssim_values = []

    mse_values = []

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

            names = batch["name"]

            prediction = model(noisy)

            prediction = prediction.clamp(
                0,
                1
            )

            pred_np = (
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
            # MSE
            # ------------------------------------------------

            mse = np.mean(
                (gt_np - pred_np) ** 2
            )

            # ------------------------------------------------
            # PSNR
            # ------------------------------------------------

            psnr = peak_signal_noise_ratio(
                gt_np,
                pred_np,
                data_range=1.0
            )

            # ------------------------------------------------
            # SSIM
            # ------------------------------------------------

            ssim = structural_similarity(
                gt_np,
                pred_np,
                data_range=1.0
            )

            mse_values.append(mse)
            psnr_values.append(psnr)
            ssim_values.append(ssim)

            # ------------------------------------------------
            # PRINT EVERY 100
            # ------------------------------------------------

            if (batch_idx + 1) % 100 == 0:

                print(
                    f"Processed "
                    f"{batch_idx + 1}/"
                    f"{len(val_dataset)}"
                )

    # ========================================================
    # RESULTS
    # ========================================================

    psnr_values = np.asarray(
        psnr_values
    )

    ssim_values = np.asarray(
        ssim_values
    )

    mse_values = np.asarray(
        mse_values
    )

    print()
    print("=" * 70)
    print("PER-IMAGE RESULTS")
    print("=" * 70)

    print(
        "Mean PSNR:",
        np.mean(psnr_values)
    )

    print(
        "Median PSNR:",
        np.median(psnr_values)
    )

    print(
        "Minimum PSNR:",
        np.min(psnr_values)
    )

    print(
        "Maximum PSNR:",
        np.max(psnr_values)
    )

    print()

    print(
        "Mean SSIM:",
        np.mean(ssim_values)
    )

    print(
        "Median SSIM:",
        np.median(ssim_values)
    )

    print(
        "Minimum SSIM:",
        np.min(ssim_values)
    )

    print(
        "Maximum SSIM:",
        np.max(ssim_values)
    )

    print()

    print(
        "Mean MSE:",
        np.mean(mse_values)
    )

    # ========================================================
    # WORST IMAGES
    # ========================================================

    print()
    print("=" * 70)
    print("WORST 20 PSNR SAMPLES")
    print("=" * 70)

    worst_indices = np.argsort(
        psnr_values
    )[:20]

    for rank, i in enumerate(
        worst_indices,
        start=1
    ):

        dataset_index = val_indices[i]

        name = dataset[dataset_index]["name"]

        print(
            f"{rank:02d}. "
            f"{name} | "
            f"PSNR={psnr_values[i]:.4f} dB | "
            f"SSIM={ssim_values[i]:.4f} | "
            f"MSE={mse_values[i]:.8f}"
        )

    # ========================================================
    # BEST IMAGES
    # ========================================================

    print()
    print("=" * 70)
    print("BEST 20 PSNR SAMPLES")
    print("=" * 70)

    best_indices = np.argsort(
        psnr_values
    )[-20:][::-1]

    for rank, i in enumerate(
        best_indices,
        start=1
    ):

        dataset_index = val_indices[i]

        name = dataset[dataset_index]["name"]

        print(
            f"{rank:02d}. "
            f"{name} | "
            f"PSNR={psnr_values[i]:.4f} dB | "
            f"SSIM={ssim_values[i]:.4f} | "
            f"MSE={mse_values[i]:.8f}"
        )

    print()
    print("=" * 70)
    print("DIAGNOSTIC COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()