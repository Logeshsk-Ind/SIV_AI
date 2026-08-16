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
BATCH_SIZE = 4


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("SIV-AI PHASE 4 FAILURE ANALYSIS")
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
    # Same validation split
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
    # DataLoader
    # --------------------------------------------------------

    loader = DataLoader(
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

    print(
        "Parameters:",
        sum(
            p.numel()
            for p in model.parameters()
        )
    )

    # --------------------------------------------------------
    # Load checkpoint
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

    print(
        "Checkpoint loaded successfully."
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    records = []

    # --------------------------------------------------------
    # Evaluation
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

            prediction = model(noisy)

            prediction = prediction.clamp(
                0,
                1
            )

            # ------------------------------------------------
            # Individual samples
            # ------------------------------------------------

            for i in range(noisy.shape[0]):

                noisy_np = (
                    noisy[i, 0]
                    .cpu()
                    .numpy()
                )

                prediction_np = (
                    prediction[i, 0]
                    .cpu()
                    .numpy()
                )

                gt_np = (
                    gt[i, 0]
                    .cpu()
                    .numpy()
                )

                # ------------------------------------------------
                # Metrics
                # ------------------------------------------------

                psnr = peak_signal_noise_ratio(
                    gt_np,
                    prediction_np,
                    data_range=1.0
                )

                ssim = structural_similarity(
                    gt_np,
                    prediction_np,
                    data_range=1.0
                )

                mse = float(
                    np.mean(
                        (
                            gt_np
                            - prediction_np
                        ) ** 2
                    )
                )

                # ------------------------------------------------
                # Statistics
                # ------------------------------------------------

                noisy_std = float(
                    np.std(noisy_np)
                )

                prediction_std = float(
                    np.std(prediction_np)
                )

                gt_std = float(
                    np.std(gt_np)
                )

                noisy_min = float(
                    np.min(noisy_np)
                )

                noisy_max = float(
                    np.max(noisy_np)
                )

                noisy_range = (
                    noisy_max
                    - noisy_min
                )

                if gt_std > 1e-8:

                    std_ratio = (
                        prediction_std
                        / gt_std
                    )

                else:

                    std_ratio = 0.0

                records.append(
                    {
                        "name": batch["name"][i],
                        "psnr": psnr,
                        "ssim": ssim,
                        "mse": mse,
                        "noisy_std": noisy_std,
                        "prediction_std": prediction_std,
                        "gt_std": gt_std,
                        "std_ratio": std_ratio,
                        "noisy_min": noisy_min,
                        "noisy_max": noisy_max,
                        "noisy_range": noisy_range,
                    }
                )

            if (batch_idx + 1) % 100 == 0:

                print(
                    f"Processed "
                    f"{(batch_idx + 1) * BATCH_SIZE}/"
                    f"{len(val_dataset)}"
                )

    # ========================================================
    # SORT BY PSNR
    # ========================================================

    records.sort(
        key=lambda x: x["psnr"]
    )

    # ========================================================
    # WORST 20
    # ========================================================

    print()
    print("=" * 70)
    print("WORST 20")
    print("=" * 70)

    for index, r in enumerate(
        records[:20],
        start=1
    ):

        print(
            f"{index:02d}. "
            f"{r['name']} | "
            f"PSNR={r['psnr']:.4f} | "
            f"SSIM={r['ssim']:.4f} | "
            f"InputSTD={r['noisy_std']:.4f} | "
            f"PredSTD={r['prediction_std']:.4f} | "
            f"GTSTD={r['gt_std']:.4f} | "
            f"Ratio={r['std_ratio']:.3f} | "
            f"Range={r['noisy_range']:.4f}"
        )

    # ========================================================
    # BEST 20
    # ========================================================

    print()
    print("=" * 70)
    print("BEST 20")
    print("=" * 70)

    for index, r in enumerate(
        records[-20:][::-1],
        start=1
    ):

        print(
            f"{index:02d}. "
            f"{r['name']} | "
            f"PSNR={r['psnr']:.4f} | "
            f"SSIM={r['ssim']:.4f} | "
            f"InputSTD={r['noisy_std']:.4f} | "
            f"PredSTD={r['prediction_std']:.4f} | "
            f"GTSTD={r['gt_std']:.4f} | "
            f"Ratio={r['std_ratio']:.3f} | "
            f"Range={r['noisy_range']:.4f}"
        )

    # ========================================================
    # GLOBAL STATISTICS
    # ========================================================

    psnr_values = np.array(
        [r["psnr"] for r in records]
    )

    ssim_values = np.array(
        [r["ssim"] for r in records]
    )

    mse_values = np.array(
        [r["mse"] for r in records]
    )

    ratio_values = np.array(
        [r["std_ratio"] for r in records]
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("GLOBAL FAILURE ANALYSIS")
    print("=" * 70)

    print(
        f"Mean PSNR       : "
        f"{np.mean(psnr_values):.6f} dB"
    )

    print(
        f"Mean SSIM       : "
        f"{np.mean(ssim_values):.6f}"
    )

    print(
        f"Mean MSE        : "
        f"{np.mean(mse_values):.8f}"
    )

    print(
        f"Mean STD ratio  : "
        f"{np.mean(ratio_values):.6f}"
    )

    print(
        f"Median STD ratio: "
        f"{np.median(ratio_values):.6f}"
    )

    # ========================================================
    # BAD IMAGES
    # ========================================================

    bad = [
        r
        for r in records
        if r["psnr"] < 20
    ]

    print()
    print("=" * 70)
    print("VERY BAD IMAGES: PSNR < 20 dB")
    print("=" * 70)

    print(
        f"Count: {len(bad)} / {len(records)}"
    )

    if bad:

        print(
            f"Mean Input STD: "
            f"{np.mean([r['noisy_std'] for r in bad]):.6f}"
        )

        print(
            f"Mean Prediction STD: "
            f"{np.mean([r['prediction_std'] for r in bad]):.6f}"
        )

        print(
            f"Mean GT STD: "
            f"{np.mean([r['gt_std'] for r in bad]):.6f}"
        )

        print(
            f"Mean STD Ratio: "
            f"{np.mean([r['std_ratio'] for r in bad]):.6f}"
        )

        print(
            f"Mean Input Range: "
            f"{np.mean([r['noisy_range'] for r in bad]):.6f}"
        )

    # ========================================================
    # GOOD IMAGES
    # ========================================================

    good = [
        r
        for r in records
        if r["psnr"] >= 30
    ]

    print()
    print("=" * 70)
    print("GOOD IMAGES: PSNR >= 30 dB")
    print("=" * 70)

    print(
        f"Count: {len(good)} / {len(records)}"
    )

    if good:

        print(
            f"Mean Input STD: "
            f"{np.mean([r['noisy_std'] for r in good]):.6f}"
        )

        print(
            f"Mean Prediction STD: "
            f"{np.mean([r['prediction_std'] for r in good]):.6f}"
        )

        print(
            f"Mean GT STD: "
            f"{np.mean([r['gt_std'] for r in good]):.6f}"
        )

        print(
            f"Mean STD Ratio: "
            f"{np.mean([r['std_ratio'] for r in good]):.6f}"
        )

        print(
            f"Mean Input Range: "
            f"{np.mean([r['noisy_range'] for r in good]):.6f}"
        )

    # ========================================================
    # TARGET CHECK
    # ========================================================

    print()
    print("=" * 70)
    print("CURRENT TARGET STATUS")
    print("=" * 70)

    mean_psnr = np.mean(psnr_values)
    mean_ssim = np.mean(ssim_values)

    print(
        f"PSNR target  : 27–28+ dB"
    )

    print(
        f"Current PSNR : {mean_psnr:.4f} dB"
    )

    print()

    print(
        f"SSIM target  : 0.85–0.95"
    )

    print(
        f"Current SSIM : {mean_ssim:.4f}"
    )

    print()

    if mean_psnr >= 27:

        print("PSNR STATUS: PASS")

    else:

        print("PSNR STATUS: NEEDS IMPROVEMENT")

    if mean_ssim >= 0.85:

        print("SSIM STATUS: TARGET REACHED")

    else:

        print("SSIM STATUS: NEEDS IMPROVEMENT")

    # ========================================================
    # END
    # ========================================================

    print()
    print("=" * 70)
    print("FAILURE ANALYSIS COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()