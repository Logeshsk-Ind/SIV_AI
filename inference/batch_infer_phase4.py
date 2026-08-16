from pathlib import Path
import sys
import time

import torch
from PIL import Image
import numpy as np


# ============================================================
# PROJECT ROOT
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT))


# ============================================================
# IMPORT MODEL
# ============================================================

from src.models.siv_ai_phase4 import SIVAI


# ============================================================
# PATHS
# ============================================================

CHECKPOINT = ROOT / "siv_ai_phase4_best.pth"

INPUT_DIR = ROOT / "test_data" / "NoisyLR"

OUTPUT_ROOT = ROOT / "outputs" / "phase4_test"

OUTPUT_NPY_DIR = OUTPUT_ROOT / "npy"

OUTPUT_PNG_DIR = OUTPUT_ROOT / "png"


# ============================================================
# SETTINGS
# ============================================================

INPUT_SIZE = (128, 128)

OUTPUT_SIZE = (256, 256)

EXPECTED_PARAMETERS = 110049


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(device):

    print()
    print("=" * 70)
    print("LOADING SIV-AI PHASE 4 MODEL")
    print("=" * 70)

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = SIVAI().to(device)

    # --------------------------------------------------------
    # Parameter count
    # --------------------------------------------------------

    total_parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        "Parameters:",
        total_parameters
    )

    if total_parameters != EXPECTED_PARAMETERS:

        raise RuntimeError(
            "\nERROR: Phase-4 architecture mismatch!\n"
            f"Expected parameters: {EXPECTED_PARAMETERS}\n"
            f"Found parameters   : {total_parameters}\n"
        )

    print(
        "Parameter count: PASS"
    )

    # --------------------------------------------------------
    # Check checkpoint
    # --------------------------------------------------------

    if not CHECKPOINT.exists():

        raise FileNotFoundError(
            "\nPhase-4 checkpoint not found:\n"
            f"{CHECKPOINT}\n"
        )

    print(
        "Checkpoint:",
        CHECKPOINT
    )

    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
        weights_only=False
    )

    # --------------------------------------------------------
    # Extract state dict
    # --------------------------------------------------------

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:

            state_dict = checkpoint[
                "model_state_dict"
            ]

        else:

            state_dict = checkpoint

    else:

        raise RuntimeError(
            "Unsupported checkpoint format."
        )

    # --------------------------------------------------------
    # Load weights
    # --------------------------------------------------------

    model.load_state_dict(
        state_dict,
        strict=True
    )

    # --------------------------------------------------------
    # Evaluation mode
    # --------------------------------------------------------

    model.eval()

    print(
        "Phase-4 checkpoint loaded successfully."
    )

    print("=" * 70)

    return model


# ============================================================
# LOAD NPY INPUT
# ============================================================

def load_npy(
    path,
    device
):

    array = np.load(path)

    # --------------------------------------------------------
    # Convert float32
    # --------------------------------------------------------

    array = array.astype(
        np.float32,
        copy=False
    )

    # --------------------------------------------------------
    # Remove singleton dimensions
    # --------------------------------------------------------

    array = np.squeeze(array)

    # --------------------------------------------------------
    # Verify dimensions
    # --------------------------------------------------------

    if array.ndim != 2:

        raise ValueError(
            f"Invalid input dimensions for {path.name}: "
            f"{array.shape}"
        )

    # --------------------------------------------------------
    # Verify size
    # --------------------------------------------------------

    expected_shape = (
        INPUT_SIZE[1],
        INPUT_SIZE[0]
    )

    if tuple(array.shape) != expected_shape:

        raise ValueError(
            f"Invalid input size for {path.name}: "
            f"expected {expected_shape}, "
            f"got {array.shape}"
        )

    # --------------------------------------------------------
    # Check numerical validity
    # --------------------------------------------------------

    if not np.isfinite(array).all():

        raise ValueError(
            f"NaN or Inf detected in {path.name}"
        )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # No /255 normalization.
    # Preserve original floating-point values.
    # --------------------------------------------------------

    tensor = torch.from_numpy(array)

    tensor = tensor.unsqueeze(0)

    tensor = tensor.unsqueeze(0)

    tensor = tensor.to(
        device=device,
        dtype=torch.float32
    )

    return tensor


# ============================================================
# SAVE FLOAT32 PREDICTION
# ============================================================

def save_prediction_npy(
    prediction,
    output_path
):

    # --------------------------------------------------------
    # Convert:
    #
    # [1,1,H,W]
    #       ↓
    # [H,W]
    # --------------------------------------------------------

    array = (
        prediction
        .detach()
        .cpu()
        .squeeze()
        .numpy()
        .astype(np.float32)
    )

    # --------------------------------------------------------
    # Save WITHOUT clipping
    # --------------------------------------------------------

    np.save(
        output_path,
        array
    )


# ============================================================
# SAVE PNG PREVIEW
# ============================================================

def save_prediction_png(
    prediction,
    output_path
):

    # --------------------------------------------------------
    # Convert to CPU
    # --------------------------------------------------------

    tensor = (
        prediction
        .detach()
        .cpu()
        .squeeze()
    )

    # --------------------------------------------------------
    # Clamp ONLY for visualization
    # --------------------------------------------------------

    tensor = tensor.clamp(
        0.0,
        1.0
    )

    # --------------------------------------------------------
    # Convert to NumPy
    # --------------------------------------------------------

    array = tensor.numpy()

    # --------------------------------------------------------
    # Convert [0,1] → [0,255]
    # --------------------------------------------------------

    array = (
        array * 255.0
    ).round().astype(
        np.uint8
    )

    # --------------------------------------------------------
    # Create image
    # --------------------------------------------------------

    image = Image.fromarray(
        array,
        mode="L"
    )

    image.save(
        output_path
    )


# ============================================================
# PROCESS ONE FILE
# ============================================================

def process_file(
    model,
    input_path,
    output_npy,
    output_png,
    device
):

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    noisy = load_npy(
        input_path,
        device
    )

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    with torch.no_grad():

        prediction = model(
            noisy
        )

    # --------------------------------------------------------
    # Verify output
    # --------------------------------------------------------

    expected_shape = (
        1,
        1,
        OUTPUT_SIZE[1],
        OUTPUT_SIZE[0]
    )

    if tuple(prediction.shape) != expected_shape:

        raise RuntimeError(
            f"Unexpected output shape for "
            f"{input_path.name}: "
            f"expected {expected_shape}, "
            f"got {tuple(prediction.shape)}"
        )

    # --------------------------------------------------------
    # Save floating-point prediction
    # --------------------------------------------------------

    save_prediction_npy(
        prediction,
        output_npy
    )

    # --------------------------------------------------------
    # Save PNG preview
    # --------------------------------------------------------

    save_prediction_png(
        prediction,
        output_png
    )

    return prediction


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("SIV-AI PHASE 4 BATCH INFERENCE")
    print("=" * 70)

    # ========================================================
    # DEVICE
    # ========================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device:",
        device
    )

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    # ========================================================
    # CHECK INPUT DIRECTORY
    # ========================================================

    if not INPUT_DIR.exists():

        raise FileNotFoundError(
            "\nNoisyLR directory not found:\n"
            f"{INPUT_DIR}"
        )

    # ========================================================
    # FIND TEST FILES
    # ========================================================

    input_files = sorted(
        INPUT_DIR.glob("*.npy")
    )

    if not input_files:

        raise FileNotFoundError(
            "\nNo .npy files found in:\n"
            f"{INPUT_DIR}"
        )

    print()
    print(
        "Input directory:"
    )

    print(
        f"  {INPUT_DIR}"
    )

    print()
    print(
        "Test files found:",
        len(input_files)
    )

    # ========================================================
    # CREATE OUTPUT DIRECTORIES
    # ========================================================

    OUTPUT_NPY_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    OUTPUT_PNG_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print()
    print(
        "NPY output:"
    )

    print(
        f"  {OUTPUT_NPY_DIR}"
    )

    print()
    print(
        "PNG preview:"
    )

    print(
        f"  {OUTPUT_PNG_DIR}"
    )

    # ========================================================
    # LOAD MODEL ONCE
    # ========================================================

    model = load_model(
        device
    )

    # ========================================================
    # PROCESS DATASET
    # ========================================================

    print()
    print("=" * 70)
    print("STARTING BATCH INFERENCE")
    print("=" * 70)

    total_start = time.perf_counter()

    successful = 0

    failed = 0

    total_inference_time = 0.0

    for index, input_path in enumerate(
        input_files,
        start=1
    ):

        print()
        print(
            f"[{index}/{len(input_files)}] "
            f"{input_path.name}"
        )

        output_npy = (
            OUTPUT_NPY_DIR
            / input_path.name
        )

        output_png = (
            OUTPUT_PNG_DIR
            / (
                input_path.stem
                + "_restored.png"
            )
        )

        try:

            start = time.perf_counter()

            prediction = process_file(
                model=model,
                input_path=input_path,
                output_npy=output_npy,
                output_png=output_png,
                device=device
            )

            # ------------------------------------------------
            # Synchronize CUDA before timing
            # ------------------------------------------------

            if device.type == "cuda":

                torch.cuda.synchronize()

            elapsed = (
                time.perf_counter()
                - start
            )

            total_inference_time += elapsed

            successful += 1

            # ------------------------------------------------
            # Statistics
            # ------------------------------------------------

            print(
                "  Input :",
                tuple(
                    (128, 128)
                )
            )

            print(
                "  Output:",
                tuple(
                    prediction.shape
                )
            )

            print(
                "  Range :",
                f"{float(prediction.min()):.6f}",
                "to",
                f"{float(prediction.max()):.6f}"
            )

            print(
                "  Time  :",
                f"{elapsed:.3f} sec"
            )

            print(
                "  Status: PASS"
            )

        except Exception as error:

            failed += 1

            print(
                "  Status: FAILED"
            )

            print(
                "  Error :",
                str(error)
            )

    # ========================================================
    # TOTAL TIME
    # ========================================================

    total_elapsed = (
        time.perf_counter()
        - total_start
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("BATCH INFERENCE COMPLETE")
    print("=" * 70)

    print()
    print(
        "Total files   :",
        len(input_files)
    )

    print(
        "Successful     :",
        successful
    )

    print(
        "Failed         :",
        failed
    )

    print()
    print(
        "Total time     :",
        f"{total_elapsed:.3f} sec"
    )

    if successful > 0:

        print(
            "Average/file   :",
            f"{total_inference_time / successful:.3f} sec"
        )

    print()
    print(
        "Floating-point predictions:"
    )

    print(
        f"  {OUTPUT_NPY_DIR}"
    )

    print()
    print(
        "PNG previews:"
    )

    print(
        f"  {OUTPUT_PNG_DIR}"
    )

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()