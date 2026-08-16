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

from models.baseline_sr import BaselineSR
from models.residual_sr import ResidualSR

from src.models.wavelet_sr import WaveletSR


# ============================================================
# Configuration
# ============================================================

DATA_DIR = (
    ROOT
    / "data"
    / "raw"
    / "train"
    / "train"
)

BATCH_SIZE = 1

SEED = 42

VAL_RATIO = 0.20


# ============================================================
# Checkpoints
# ============================================================

CHECKPOINTS = {

    "Day 2 Baseline":
        ROOT / "checkpoints" / "baseline_day2_best.pth",

    "Day 3 Baseline":
        ROOT / "checkpoints" / "baseline_day3_best.pth",

    "Day 4 Residual":
        ROOT / "checkpoints" / "residual_day4_best.pth",

    "WaveletSR":
        ROOT / "checkpoints" / "wavelet_sr_best.pth",
}


# ============================================================
# Model constructors
# ============================================================

def create_model(name):

    if name == "Day 2 Baseline":
        return BaselineSR()

    if name == "Day 3 Baseline":
        return BaselineSR()

    if name == "Day 4 Residual":
        return ResidualSR()

    if name == "WaveletSR":
        return WaveletSR()

    raise ValueError(
        f"Unknown model: {name}"
    )


# ============================================================
# Parameter count
# ============================================================

def count_parameters(model):

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


# ============================================================
# Main
# ============================================================

def main():

    print()
    print("=" * 70)
    print("SIV-AI FINAL MODEL COMPARISON")
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
    # Store results
    # --------------------------------------------------------

    results = {}


    # ========================================================
    # Evaluate every model
    # ========================================================

    for model_name, checkpoint_path in CHECKPOINTS.items():

        print()
        print("-" * 70)
        print(
            f"Evaluating: {model_name}"
        )
        print("-" * 70)


        # ----------------------------------------------------
        # Check checkpoint
        # ----------------------------------------------------

        if not checkpoint_path.exists():

            raise FileNotFoundError(
                f"Checkpoint not found: "
                f"{checkpoint_path}"
            )


        # ----------------------------------------------------
        # Create model
        # ----------------------------------------------------

        model = create_model(
            model_name
        ).to(device)


        # ----------------------------------------------------
        # Load checkpoint
        # ----------------------------------------------------

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device
        )


        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        model.eval()


        # ----------------------------------------------------
        # Parameter count
        # ----------------------------------------------------

        parameters = count_parameters(
            model
        )


        print(
            "Checkpoint epoch:",
            checkpoint.get("epoch")
        )

        print(
            "Validation loss:",
            checkpoint.get("val_loss")
        )

        print(
            "Parameters:",
            parameters
        )


        # ----------------------------------------------------
        # Metric accumulators
        # ----------------------------------------------------

        psnr_values = []

        ssim_values = []


        # ----------------------------------------------------
        # Evaluation
        # ----------------------------------------------------

        with torch.no_grad():

            for batch_idx, batch in enumerate(
                validation_loader,
                start=1
            ):

                noisy = batch["noisy"].to(
                    device,
                    non_blocking=True
                )

                gt = batch["gt"].to(
                    device,
                    non_blocking=True
                )


                # Forward pass

                prediction = model(
                    noisy
                )


                # Valid image range

                prediction = prediction.clamp(
                    0.0,
                    1.0
                )


                # Convert first image to NumPy

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


                psnr_values.append(
                    psnr
                )

                ssim_values.append(
                    ssim
                )


                # ------------------------------------------------
                # Progress
                # ------------------------------------------------

                if batch_idx % 200 == 0:

                    print(
                        f"Processed "
                        f"{batch_idx}/"
                        f"{len(validation_loader)}"
                    )


        # ----------------------------------------------------
        # Mean metrics
        # ----------------------------------------------------

        mean_psnr = float(
            np.mean(psnr_values)
        )

        mean_ssim = float(
            np.mean(ssim_values)
        )


        # ----------------------------------------------------
        # Save results
        # ----------------------------------------------------

        results[model_name] = {

            "psnr":
                mean_psnr,

            "ssim":
                mean_ssim,

            "parameters":
                parameters,

            "epoch":
                checkpoint.get("epoch"),

            "val_loss":
                checkpoint.get("val_loss"),
        }


        print()
        print(
            f"{model_name} Results"
        )

        print(
            f"PSNR: {mean_psnr:.4f} dB"
        )

        print(
            f"SSIM: {mean_ssim:.4f}"
        )


    # ========================================================
    # Final comparison
    # ========================================================

    baseline_psnr = results[
        "Day 2 Baseline"
    ]["psnr"]

    baseline_ssim = results[
        "Day 2 Baseline"
    ]["ssim"]


    print()
    print()
    print("=" * 90)
    print("FINAL SIV-AI MODEL COMPARISON")
    print("=" * 90)

    print()

    print(
        f"{'Model':<20}"
        f"{'PSNR (dB)':>14}"
        f"{'SSIM':>12}"
        f"{'Parameters':>16}"
        f"{'Δ PSNR':>14}"
        f"{'Δ SSIM':>14}"
    )

    print("-" * 90)


    for model_name in CHECKPOINTS:

        result = results[
            model_name
        ]

        delta_psnr = (
            result["psnr"]
            - baseline_psnr
        )

        delta_ssim = (
            result["ssim"]
            - baseline_ssim
        )


        print(
            f"{model_name:<20}"
            f"{result['psnr']:>14.4f}"
            f"{result['ssim']:>12.4f}"
            f"{result['parameters']:>16,}"
            f"{delta_psnr:>+14.4f}"
            f"{delta_ssim:>+14.4f}"
        )


    # ========================================================
    # Best models
    # ========================================================

    best_psnr_model = max(
        results,
        key=lambda name:
            results[name]["psnr"]
    )

    best_ssim_model = max(
        results,
        key=lambda name:
            results[name]["ssim"]
    )


    print()
    print("=" * 90)

    print(
        "Best PSNR:",
        best_psnr_model,
        f"({results[best_psnr_model]['psnr']:.4f} dB)"
    )

    print(
        "Best SSIM:",
        best_ssim_model,
        f"({results[best_ssim_model]['ssim']:.4f})"
    )

    print("=" * 90)


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()