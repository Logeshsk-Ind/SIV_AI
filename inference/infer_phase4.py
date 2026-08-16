from pathlib import Path
import sys
import argparse

import torch
from PIL import Image
import numpy as np


# ============================================================
# PROJECT ROOT
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

# Add project root to Python path
sys.path.insert(0, str(ROOT))


# ============================================================
# IMPORT MODEL
# ============================================================

from src.models.siv_ai_phase4 import SIVAI


# ============================================================
# PATHS
# ============================================================

CHECKPOINT = ROOT / "siv_ai_phase4_best.pth"

INPUT_DIR = ROOT / "inputs"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "phase4"


# ============================================================
# IMAGE SETTINGS
# ============================================================

INPUT_SIZE = (128, 128)
OUTPUT_SIZE = (256, 256)


# ============================================================
# SUPPORTED INPUT FORMATS
# ============================================================

SUPPORTED_EXTENSIONS = {
    ".npy",
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


# ============================================================
# LOAD IMAGE / NPY
# ============================================================

def load_image(image_path, device):
    """
    Load an input image or NumPy .npy array.

    Supported:
        PNG, JPG, JPEG, BMP, TIFF, WEBP
        NPY

    Model input:
        [1, 1, 128, 128]

    Standard image files:
        uint8 [0,255] -> float32 [0,1]

    NPY files:
        Values are preserved as float32.
        NO /255 normalization is applied.
    """

    image_path = Path(image_path)

    # --------------------------------------------------------
    # Check file exists
    # --------------------------------------------------------

    if not image_path.exists():
        raise FileNotFoundError(
            f"Input image not found:\n{image_path}"
        )

    # --------------------------------------------------------
    # Determine extension
    # --------------------------------------------------------

    suffix = image_path.suffix.lower()

    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported image format: {suffix}\n"
            f"Supported formats: "
            f"{', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    print()
    print("Loading input:")
    print(" ", image_path)

    # ========================================================
    # NPY INPUT
    # ========================================================

    if suffix == ".npy":

        print()
        print("Detected NumPy test input (.npy)")

        # ----------------------------------------------------
        # Load NPY
        # ----------------------------------------------------

        image_array = np.load(image_path)

        # ----------------------------------------------------
        # Display original NPY information
        # ----------------------------------------------------

        print()
        print("NPY information:")

        print(
            "  Shape :",
            image_array.shape
        )

        print(
            "  Dtype :",
            image_array.dtype
        )

        print(
            "  Min   :",
            float(np.min(image_array))
        )

        print(
            "  Max   :",
            float(np.max(image_array))
        )

        print(
            "  Mean  :",
            float(np.mean(image_array))
        )

        print(
            "  Std   :",
            float(np.std(image_array))
        )

        # ----------------------------------------------------
        # Convert to float32
        # ----------------------------------------------------

        image_array = image_array.astype(
            np.float32,
            copy=False
        )

        # ----------------------------------------------------
        # Remove unnecessary dimensions
        # ----------------------------------------------------

        image_array = np.squeeze(image_array)

        # ----------------------------------------------------
        # Accept grayscale layouts:
        #
        # [128,128]
        # [1,128,128]
        # [128,128,1]
        # ----------------------------------------------------

        if image_array.ndim == 2:

            pass

        elif image_array.ndim == 3:

            if image_array.shape[0] == 1:

                image_array = image_array[0]

            elif image_array.shape[-1] == 1:

                image_array = image_array[:, :, 0]

            else:

                raise ValueError(
                    "\nUnsupported NPY shape.\n"
                    f"Expected grayscale image, "
                    f"got: {image_array.shape}"
                )

        else:

            raise ValueError(
                "\nUnsupported NPY dimensions.\n"
                f"Expected 2D grayscale array, "
                f"got: shape={image_array.shape}"
            )

        # ----------------------------------------------------
        # Display final array shape
        # ----------------------------------------------------

        print()
        print(
            "After dimension processing:",
            image_array.shape
        )

        # ----------------------------------------------------
        # Verify spatial size
        # ----------------------------------------------------

        expected_hw = (
            INPUT_SIZE[1],
            INPUT_SIZE[0]
        )

        if tuple(image_array.shape) != expected_hw:

            raise ValueError(
                "\nUnexpected NPY input size!\n"
                f"Expected: {expected_hw}\n"
                f"Got     : {image_array.shape}\n"
                "\n"
                "The Phase-4 model expects "
                "a 128x128 NoisyLR input."
            )

        # ----------------------------------------------------
        # Check NaN / Inf
        # ----------------------------------------------------

        if not np.isfinite(image_array).all():

            raise ValueError(
                "\nNPY input contains NaN or Inf values."
            )

        # ----------------------------------------------------
        # Preserve original floating-point values
        # ----------------------------------------------------

        print()
        print("NPY values preserved.")
        print("No /255 normalization applied.")

        # ----------------------------------------------------
        # Convert NumPy → PyTorch
        #
        # [H,W]
        #    ↓
        # [1,H,W]
        #    ↓
        # [1,1,H,W]
        # ----------------------------------------------------

        tensor = torch.from_numpy(image_array)

        tensor = tensor.unsqueeze(0)
        tensor = tensor.unsqueeze(0)

        # ----------------------------------------------------
        # Move to device
        # ----------------------------------------------------

        tensor = tensor.to(
            device=device,
            dtype=torch.float32
        )

        # ----------------------------------------------------
        # Print model input
        # ----------------------------------------------------

        print()
        print("Model input:")
        print(" ", tuple(tensor.shape))

        print(
            "Input range:",
            float(tensor.min()),
            "to",
            float(tensor.max())
        )

        return tensor

    # ========================================================
    # STANDARD IMAGE INPUT
    # ========================================================

    print()
    print("Detected standard image input.")

    # --------------------------------------------------------
    # Open image
    # --------------------------------------------------------

    image = Image.open(image_path)

    print()
    print("Original image:")

    print(
        "  Mode :",
        image.mode
    )

    print(
        "  Size :",
        image.size
    )

    # --------------------------------------------------------
    # Convert to grayscale
    # --------------------------------------------------------

    image = image.convert("L")

    # --------------------------------------------------------
    # Resize to Phase-4 input size
    # --------------------------------------------------------

    image = image.resize(
        INPUT_SIZE,
        Image.Resampling.BICUBIC
    )

    # --------------------------------------------------------
    # Convert to NumPy
    # --------------------------------------------------------

    image_array = np.asarray(
        image,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Normalize standard images to [0,1]
    # --------------------------------------------------------

    image_array = image_array / 255.0

    # --------------------------------------------------------
    # Convert:
    #
    # [H,W]
    #      ↓
    # [1,H,W]
    #      ↓
    # [1,1,H,W]
    # --------------------------------------------------------

    tensor = torch.from_numpy(image_array)

    tensor = tensor.unsqueeze(0)
    tensor = tensor.unsqueeze(0)

    # --------------------------------------------------------
    # Move to device
    # --------------------------------------------------------

    tensor = tensor.to(
        device=device,
        dtype=torch.float32
    )

    # --------------------------------------------------------
    # Print model input
    # --------------------------------------------------------

    print()
    print("Model input:")

    print(
        " ",
        tuple(tensor.shape)
    )

    print(
        "Input range:",
        float(tensor.min()),
        "to",
        float(tensor.max())
    )

    return tensor


# ============================================================
# SAVE IMAGE
# ============================================================

def save_tensor_as_image(
    tensor,
    output_path
):
    """
    Convert model output tensor into
    an 8-bit grayscale PNG.
    """

    # --------------------------------------------------------
    # Remove batch/channel dimensions
    # --------------------------------------------------------

    tensor = tensor.detach().cpu()
    tensor = tensor.squeeze()

    # --------------------------------------------------------
    # Verify output dimensions
    # --------------------------------------------------------

    if tuple(tensor.shape) != OUTPUT_SIZE[::-1]:

        raise RuntimeError(
            "\nCannot save output.\n"
            f"Expected image size: {OUTPUT_SIZE[::-1]}\n"
            f"Got                : {tuple(tensor.shape)}"
        )

    # --------------------------------------------------------
    # Clamp only for PNG visualization
    # --------------------------------------------------------

    tensor = tensor.clamp(0.0, 1.0)

    # --------------------------------------------------------
    # Convert to NumPy
    # --------------------------------------------------------

    image_array = tensor.numpy()

    # --------------------------------------------------------
    # Convert [0,1] → [0,255]
    # --------------------------------------------------------

    image_array = (
        image_array * 255.0
    ).round().astype(np.uint8)

    # --------------------------------------------------------
    # Create PIL image
    # --------------------------------------------------------

    image = Image.fromarray(
        image_array,
        mode="L"
    )

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    image.save(output_path)

    print()
    print("Saved:")
    print(" ", output_path)

    print(
        "Output size:",
        image.size
    )


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(device):
    """
    Load exact Phase-4 SIV-AI model.
    """

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

    # --------------------------------------------------------
    # Verify exact architecture
    # --------------------------------------------------------

    if total_parameters != 110049:

        raise RuntimeError(
            "\nERROR: Phase-4 architecture mismatch!\n"
            f"Expected parameters: 110049\n"
            f"Found parameters   : {total_parameters}\n"
        )

    print("Parameter count: PASS")

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
    # Support full checkpoint
    # --------------------------------------------------------

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:

            state_dict = checkpoint["model_state_dict"]

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
# RUN INFERENCE
# ============================================================

def run_inference(
    model,
    image_path,
    output_path,
    device
):
    """
    Run SIV-AI Phase-4 inference.
    """

    # --------------------------------------------------------
    # Load input
    # --------------------------------------------------------

    noisy = load_image(
        image_path,
        device
    )

    # --------------------------------------------------------
    # Verify input shape
    # --------------------------------------------------------

    expected_input_shape = (
        1,
        1,
        INPUT_SIZE[1],
        INPUT_SIZE[0]
    )

    if tuple(noisy.shape) != expected_input_shape:

        raise RuntimeError(
            "\nUnexpected model input shape!\n"
            f"Expected: {expected_input_shape}\n"
            f"Got     : {tuple(noisy.shape)}"
        )

    print()
    print("Input shape: PASS")

    # ========================================================
    # INFERENCE
    # ========================================================

    print()
    print("=" * 70)
    print("RUNNING SIV-AI PHASE 4")
    print("=" * 70)

    with torch.no_grad():

        prediction = model(noisy)

    # --------------------------------------------------------
    # Verify output shape
    # --------------------------------------------------------

    print()
    print(
        "Prediction shape:",
        tuple(prediction.shape)
    )

    expected_shape = (
        1,
        1,
        OUTPUT_SIZE[1],
        OUTPUT_SIZE[0]
    )

    if tuple(prediction.shape) != expected_shape:

        raise RuntimeError(
            "\nUnexpected model output shape!\n"
            f"Expected: {expected_shape}\n"
            f"Got     : {tuple(prediction.shape)}"
        )

    print("Output shape: PASS")

    # --------------------------------------------------------
    # Output statistics
    # --------------------------------------------------------

    print()
    print("Output statistics:")

    print(
        "  Min   :",
        float(prediction.min())
    )

    print(
        "  Max   :",
        float(prediction.max())
    )

    print(
        "  Mean  :",
        float(prediction.mean())
    )

    print(
        "  Std   :",
        float(prediction.std())
    )

    # --------------------------------------------------------
    # Save output
    # --------------------------------------------------------

    save_tensor_as_image(
        prediction,
        output_path
    )

    print("=" * 70)

    return prediction


# ============================================================
# FIND DEFAULT INPUT
# ============================================================

def find_default_input():

    INPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    images = sorted(
        [
            p
            for p in INPUT_DIR.iterdir()
            if p.is_file()
            and p.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
    )

    if not images:
        return None

    return images[0]


# ============================================================
# ARGUMENT PARSING
# ============================================================

def parse_arguments():

    parser = argparse.ArgumentParser(
        description="SIV-AI Phase-4 Image Restoration"
    )

    parser.add_argument(
        "input",
        nargs="?",
        default=None,
        help="Input image or .npy file"
    )

    parser.add_argument(
        "--output_dir",
        default=None,
        help="Directory where restored PNG will be saved"
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("SIV-AI PHASE 4 IMAGE RESTORATION")
    print("=" * 70)

    # ========================================================
    # DEVICE
    # ========================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    # ========================================================
    # ARGUMENTS
    # ========================================================

    args = parse_arguments()

    # ========================================================
    # GET INPUT
    # ========================================================

    if args.input is not None:

        image_path = Path(args.input)

    else:

        print()
        print("No input path supplied.")

        image_path = find_default_input()

        if image_path is None:

            print()
            print("No test inputs found.")

            print()
            print("Put an input inside:")
            print(f"  {INPUT_DIR}")

            print()
            print("Supported formats:")
            print(
                "  NPY, PNG, JPG, JPEG, "
                "BMP, TIFF, WEBP"
            )

            print()
            print("Example:")
            print(
                "  python "
                "inference/infer_phase4.py "
                "test_data/NoisyLR/000000.npy"
            )

            return

        print()
        print(
            "Using first input from inputs folder:"
        )

        print(" ", image_path)

    # ========================================================
    # RESOLVE RELATIVE INPUT PATH
    # ========================================================

    if not image_path.is_absolute():

        image_path = (
            ROOT / image_path
        ).resolve()

    # ========================================================
    # OUTPUT DIRECTORY
    # ========================================================

    if args.output_dir:

        output_dir = Path(
            args.output_dir
        )

        if not output_dir.is_absolute():

            output_dir = (
                ROOT / output_dir
            ).resolve()

    else:

        output_dir = DEFAULT_OUTPUT_DIR

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # CREATE OUTPUT FILENAME
    # ========================================================

    output_name = (
        image_path.stem
        + "_restored.png"
    )

    output_path = (
        output_dir / output_name
    )

    # ========================================================
    # LOAD MODEL
    # ========================================================

    model = load_model(device)

    # ========================================================
    # RUN INFERENCE
    # ========================================================

    run_inference(
        model=model,
        image_path=image_path,
        output_path=output_path,
        device=device
    )

    # ========================================================
    # FINAL MESSAGE
    # ========================================================

    print()
    print("=" * 70)
    print("INFERENCE COMPLETE")
    print("=" * 70)

    print()
    print("Input:")
    print(
        f"  {image_path}"
    )

    print()
    print("Restored:")
    print(
        f"  {output_path}"
    )

    print()
    print("Input  : 128 × 128")
    print("Output : 256 × 256")

    print()
    print(
        "Phase-4 inference completed successfully."
    )

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()