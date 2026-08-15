from pathlib import Path
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt


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
# Paths
# ============================================================

DATA_DIR = (
    ROOT
    / "data"
    / "raw"
    / "train"
    / "train"
)

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


OUTPUT_DIR = (
    ROOT
    / "results"
    / "visual_comparison"
)


# ============================================================
# Configuration
# ============================================================

SEED = 42

VAL_RATIO = 0.20

NUM_SAMPLES = 5


# ============================================================
# Model creation
# ============================================================

def create_model(name):

    if name in [
        "Day 2 Baseline",
        "Day 3 Baseline",
    ]:
        return BaselineSR()

    if name == "Day 4 Residual":
        return ResidualSR()

    if name == "WaveletSR":
        return WaveletSR()

    raise ValueError(
        f"Unknown model: {name}"
    )


# ============================================================
# Main
# ============================================================

def main():

    print()
    print("=" * 70)
    print("SIV-AI VISUAL COMPARISON")
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
    # Output directory
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
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
    # Reproduce validation split
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


    # --------------------------------------------------------
    # Select evenly spaced validation samples
    # --------------------------------------------------------

    positions = np.linspace(
        0,
        len(val_indices) - 1,
        NUM_SAMPLES,
        dtype=int
    )

    selected_indices = [
        val_indices[position]
        for position in positions
    ]


    print(
        "Selected validation indices:",
        selected_indices
    )


    # --------------------------------------------------------
    # Load models
    # --------------------------------------------------------

    models = {}


    for model_name, checkpoint_path in CHECKPOINTS.items():

        print(
            f"Loading {model_name}..."
        )


        model = create_model(
            model_name
        ).to(device)


        if not checkpoint_path.exists():

            raise FileNotFoundError(
                f"Checkpoint not found: "
                f"{checkpoint_path}"
            )


        checkpoint = torch.load(
            checkpoint_path,
            map_location=device
        )


        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        model.eval()

        models[model_name] = model


    print(
        "All models loaded successfully."
    )


    # ========================================================
    # Generate visual comparisons
    # ========================================================

    for sample_number, dataset_index in enumerate(
        selected_indices,
        start=1
    ):

        sample = dataset[dataset_index]


        noisy = sample["noisy"].unsqueeze(0).to(
            device
        )

        gt = sample["gt"].squeeze(0).cpu().numpy()

        sample_name = sample["name"]


        print()
        print(
            f"Generating sample "
            f"{sample_number}/{NUM_SAMPLES}: "
            f"{sample_name}"
        )


        # ----------------------------------------------------
        # Predictions
        # ----------------------------------------------------

        predictions = {}


        with torch.no_grad():

            for model_name, model in models.items():

                prediction = model(
                    noisy
                )

                prediction = prediction.clamp(
                    0.0,
                    1.0
                )

                prediction = (
                    prediction[0, 0]
                    .cpu()
                    .numpy()
                )

                predictions[model_name] = prediction


        # ----------------------------------------------------
        # Input
        # ----------------------------------------------------

        noisy_np = (
            sample["noisy"]
            .squeeze(0)
            .cpu()
            .numpy()
        )


        # ----------------------------------------------------
        # Plot
        # ----------------------------------------------------

        fig, axes = plt.subplots(
            2,
            3,
            figsize=(14, 9)
        )


        # Noisy input

        axes[0, 0].imshow(
            noisy_np,
            cmap="gray",
            vmin=0,
            vmax=1
        )

        axes[0, 0].set_title(
            "Noisy Input\n128×128"
        )

        axes[0, 0].axis("off")


        # Ground truth

        axes[0, 1].imshow(
            gt,
            cmap="gray",
            vmin=0,
            vmax=1
        )

        axes[0, 1].set_title(
            "Ground Truth\n256×256"
        )

        axes[0, 1].axis("off")


        # Day 2

        axes[0, 2].imshow(
            predictions["Day 2 Baseline"],
            cmap="gray",
            vmin=0,
            vmax=1
        )

        axes[0, 2].set_title(
            "Day 2 Baseline"
        )

        axes[0, 2].axis("off")


        # Day 3

        axes[1, 0].imshow(
            predictions["Day 3 Baseline"],
            cmap="gray",
            vmin=0,
            vmax=1
        )

        axes[1, 0].set_title(
            "Day 3 Baseline"
        )

        axes[1, 0].axis("off")


        # Day 4

        axes[1, 1].imshow(
            predictions["Day 4 Residual"],
            cmap="gray",
            vmin=0,
            vmax=1
        )

        axes[1, 1].set_title(
            "Day 4 Residual"
        )

        axes[1, 1].axis("off")


        # WaveletSR

        axes[1, 2].imshow(
            predictions["WaveletSR"],
            cmap="gray",
            vmin=0,
            vmax=1
        )

        axes[1, 2].set_title(
            "WaveletSR"
        )

        axes[1, 2].axis("off")


        # ----------------------------------------------------
        # Figure title
        # ----------------------------------------------------

        fig.suptitle(
            f"SIV-AI Visual Comparison — Sample {sample_name}",
            fontsize=16
        )


        plt.tight_layout()


        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        output_path = (
            OUTPUT_DIR
            / f"sample_{sample_name}.png"
        )


        plt.savefig(
            output_path,
            dpi=200,
            bbox_inches="tight"
        )

        plt.close()


        print(
            "Saved:",
            output_path
        )


    # ========================================================
    # Complete
    # ========================================================

    print()
    print("=" * 70)
    print("VISUAL COMPARISON COMPLETED")
    print("=" * 70)

    print(
        "Output directory:",
        OUTPUT_DIR
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()