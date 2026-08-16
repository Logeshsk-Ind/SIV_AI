from pathlib import Path
import sys
import io
import time

import numpy as np
import torch

from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware


# ============================================================
# PROJECT ROOT
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT))


# ============================================================
# IMPORT SIV-AI MODEL
# ============================================================

from src.models.siv_ai_phase4 import SIVAI


# ============================================================
# CONFIGURATION
# ============================================================

CHECKPOINT = ROOT / "siv_ai_phase4_best.pth"

EXPECTED_PARAMETERS = 110049

INPUT_SIZE = (128, 128)
OUTPUT_SIZE = (256, 256)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# SUPPORTED FILE TYPES
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
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="SIV-AI Phase 4",
    description=(
        "Semiconductor Inspection and Verification AI "
        "for NoisyLR image restoration."
    ),
    version="1.0.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# STARTUP INFORMATION
# ============================================================

print()
print("=" * 70)
print("SIV-AI WEB BACKEND")
print("=" * 70)

print("Device:", DEVICE)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# LOAD MODEL ONCE
# ============================================================

print()
print("Loading Phase-4 model...")


model = SIVAI().to(DEVICE)


# ============================================================
# VERIFY PARAMETER COUNT
# ============================================================

parameter_count = sum(
    p.numel()
    for p in model.parameters()
)

print(
    "Parameters:",
    parameter_count
)


if parameter_count != EXPECTED_PARAMETERS:

    raise RuntimeError(
        "\nPhase-4 architecture mismatch!\n"
        f"Expected: {EXPECTED_PARAMETERS}\n"
        f"Found   : {parameter_count}\n"
    )


print("Parameter count: PASS")


# ============================================================
# CHECK CHECKPOINT
# ============================================================

if not CHECKPOINT.exists():

    raise FileNotFoundError(
        "\nPhase-4 checkpoint not found:\n"
        f"{CHECKPOINT}"
    )


print(
    "Checkpoint:",
    CHECKPOINT
)


# ============================================================
# LOAD CHECKPOINT
# ============================================================

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE,
    weights_only=False,
)


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


model.load_state_dict(
    state_dict,
    strict=True,
)

model.eval()


print(
    "Phase-4 model loaded successfully."
)

print("=" * 70)


# ============================================================
# NPY PREPROCESSING
# ============================================================

def prepare_npy(array):
    """
    Prepare KLA NoisyLR .npy input.

    Expected:
        128 x 128

    Values:
        float32

    IMPORTANT:
        NPY values are preserved.

        No /255 normalization.

        No clipping before inference.
    """

    # --------------------------------------------------------
    # Convert to float32
    # --------------------------------------------------------

    array = np.asarray(
        array,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Remove dimensions of size 1
    # --------------------------------------------------------

    array = np.squeeze(array)

    # --------------------------------------------------------
    # Accept grayscale layouts
    #
    # (128,128)
    # (1,128,128)
    # (128,128,1)
    # --------------------------------------------------------

    if array.ndim == 2:

        pass

    elif array.ndim == 3:

        if array.shape[0] == 1:

            array = array[0]

        elif array.shape[-1] == 1:

            array = array[:, :, 0]

        else:

            raise ValueError(
                "Unsupported NPY shape. "
                "Expected grayscale array, "
                f"got {array.shape}"
            )

    else:

        raise ValueError(
            "Unsupported NPY dimensions. "
            f"Got shape {array.shape}"
        )

    # --------------------------------------------------------
    # Verify 128x128
    # --------------------------------------------------------

    if tuple(array.shape) != (128, 128):

        raise ValueError(
            "Invalid NPY dimensions.\n"
            "Expected: (128, 128)\n"
            f"Received: {array.shape}"
        )

    # --------------------------------------------------------
    # Check NaN / Inf
    # --------------------------------------------------------

    if not np.isfinite(array).all():

        raise ValueError(
            "NPY file contains NaN or Inf values."
        )

    # --------------------------------------------------------
    # Preserve original values
    #
    # NO:
    #   /255
    #
    # NO:
    #   clipping
    # --------------------------------------------------------

    tensor = torch.from_numpy(array)

    # (128,128)
    #     ↓
    # (1,128,128)
    #     ↓
    # (1,1,128,128)

    tensor = tensor.unsqueeze(0)
    tensor = tensor.unsqueeze(0)

    tensor = tensor.to(
        DEVICE,
        dtype=torch.float32
    )

    return tensor


# ============================================================
# STANDARD IMAGE PREPROCESSING
# ============================================================

def prepare_standard_image(contents):
    """
    Prepare PNG/JPG/etc.

    Grayscale
    Resize to 128x128
    Normalize uint8 image to [0,1]
    """

    image = Image.open(
        io.BytesIO(contents)
    )

    image = image.convert("L")

    image = image.resize(
        INPUT_SIZE,
        Image.Resampling.BICUBIC
    )

    array = np.asarray(
        image,
        dtype=np.float32
    )

    array = array / 255.0

    tensor = torch.from_numpy(array)

    tensor = tensor.unsqueeze(0)
    tensor = tensor.unsqueeze(0)

    tensor = tensor.to(
        DEVICE,
        dtype=torch.float32
    )

    return tensor


# ============================================================
# OUTPUT → PNG
# ============================================================

def prediction_to_png(prediction):
    """
    Convert Phase-4 prediction to 256x256 PNG.

    Clipping is ONLY for PNG visualization.
    """

    output = prediction.detach().cpu()

    output = output.squeeze()

    # --------------------------------------------------------
    # Verify output size
    # --------------------------------------------------------

    if tuple(output.shape) != (256, 256):

        raise RuntimeError(
            "Unexpected model output size. "
            f"Expected (256,256), got {tuple(output.shape)}"
        )

    # --------------------------------------------------------
    # Clamp only for image visualization
    # --------------------------------------------------------

    output = output.clamp(
        0.0,
        1.0
    )

    # --------------------------------------------------------
    # Convert [0,1] → [0,255]
    # --------------------------------------------------------

    output_array = (
        output.numpy() * 255.0
    ).round().astype(np.uint8)

    # --------------------------------------------------------
    # PIL image
    # --------------------------------------------------------

    image = Image.fromarray(
        output_array,
        mode="L"
    )

    # --------------------------------------------------------
    # Encode PNG into memory
    # --------------------------------------------------------

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="PNG"
    )

    buffer.seek(0)

    return buffer, image


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():

    return {
        "project": "SIV-AI",
        "model": "Phase-4",
        "status": "online",
        "device": str(DEVICE),
        "gpu": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None
        ),
        "parameters": parameter_count,
        "input_size": "128x128",
        "output_size": "256x256",
        "npy_supported": True,
        "supported_formats": [
            "NPY",
            "PNG",
            "JPG",
            "JPEG",
            "BMP",
            "TIF",
            "TIFF",
            "WEBP",
        ],
        "endpoint": "/restore",
    }


# ============================================================
# HEALTH ENDPOINT
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "project": "SIV-AI",
        "model": "Phase-4",
        "device": str(DEVICE),
        "gpu": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None
        ),
        "parameters": parameter_count,
        "npy_supported": True,
    }


# ============================================================
# RESTORE ENDPOINT
# ============================================================

@app.post("/restore")
async def restore(
    file: UploadFile = File(...)
):

    # ========================================================
    # FILE INFORMATION
    # ========================================================

    filename = file.filename or ""

    suffix = Path(
        filename
    ).suffix.lower()

    print()
    print("=" * 70)
    print("NEW RESTORATION REQUEST")
    print("=" * 70)

    print(
        "Filename:",
        filename
    )

    print(
        "Extension:",
        suffix
    )


    # ========================================================
    # VALIDATE EXTENSION
    # ========================================================

    if suffix not in SUPPORTED_EXTENSIONS:

        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported file format. "
                "Use NPY, PNG, JPG, JPEG, BMP, "
                "TIF, TIFF or WEBP."
            ),
        )


    # ========================================================
    # READ UPLOADED FILE
    # ========================================================

    try:

        contents = await file.read()

    except Exception as exc:

        raise HTTPException(
            status_code=400,
            detail=(
                f"Unable to read uploaded file: {exc}"
            ),
        )


    if not contents:

        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty.",
        )


    print(
        "Upload size:",
        len(contents),
        "bytes"
    )


    # ========================================================
    # PREPARE INPUT
    # ========================================================

    try:

        # ====================================================
        # NPY
        # ====================================================

        if suffix == ".npy":

            print()
            print("-" * 70)
            print("NPY INPUT DETECTED")
            print("-" * 70)

            # -----------------------------------------------
            # Load directly from memory
            # -----------------------------------------------

            array = np.load(
                io.BytesIO(contents),
                allow_pickle=False,
            )

            # -----------------------------------------------
            # Information
            # -----------------------------------------------

            print(
                "Shape:",
                array.shape
            )

            print(
                "Dtype:",
                array.dtype
            )

            print(
                "Min:",
                float(np.min(array))
            )

            print(
                "Max:",
                float(np.max(array))
            )

            print(
                "Mean:",
                float(np.mean(array))
            )

            print(
                "Std:",
                float(np.std(array))
            )

            # -----------------------------------------------
            # Prepare
            # -----------------------------------------------

            tensor = prepare_npy(
                array
            )

            print(
                "NPY values preserved."
            )

            print(
                "No /255 normalization applied."
            )

        # ====================================================
        # STANDARD IMAGE
        # ====================================================

        else:

            print()
            print("-" * 70)
            print("STANDARD IMAGE INPUT DETECTED")
            print("-" * 70)

            tensor = prepare_standard_image(
                contents
            )

            print(
                "Converted to grayscale."
            )

            print(
                "Resized to 128x128."
            )

            print(
                "Normalized to [0,1]."
            )


    except Exception as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )


    # ========================================================
    # VERIFY INPUT SHAPE
    # ========================================================

    expected_input = (
        1,
        1,
        128,
        128,
    )

    if tuple(tensor.shape) != expected_input:

        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid model input shape. "
                f"Expected {expected_input}, "
                f"got {tuple(tensor.shape)}"
            ),
        )


    print()
    print(
        "Model input:",
        tuple(tensor.shape)
    )

    print(
        "Input range:",
        float(tensor.min()),
        "to",
        float(tensor.max())
    )

    print(
        "Input shape: PASS"
    )


    # ========================================================
    # MODEL INFERENCE
    # ========================================================

    print()
    print("-" * 70)
    print("RUNNING SIV-AI PHASE 4")
    print("-" * 70)

    try:

        if DEVICE.type == "cuda":

            torch.cuda.synchronize()

        start_time = time.perf_counter()

        with torch.no_grad():

            prediction = model(
                tensor
            )

        if DEVICE.type == "cuda":

            torch.cuda.synchronize()

        inference_time = (
            time.perf_counter()
            - start_time
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Model inference failed: {exc}"
            ),
        )


    # ========================================================
    # VERIFY OUTPUT
    # ========================================================

    expected_output = (
        1,
        1,
        256,
        256,
    )

    if tuple(prediction.shape) != expected_output:

        raise HTTPException(
            status_code=500,
            detail=(
                "Invalid model output shape. "
                f"Expected {expected_output}, "
                f"got {tuple(prediction.shape)}"
            ),
        )


    print()
    print(
        "Prediction shape:",
        tuple(prediction.shape)
    )

    print(
        "Output shape: PASS"
    )


    # ========================================================
    # OUTPUT STATISTICS
    # ========================================================

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


    # ========================================================
    # CONVERT TO PNG
    # ========================================================

    try:

        buffer, image = prediction_to_png(
            prediction
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Unable to create PNG output: {exc}"
            ),
        )


    # ========================================================
    # FINAL LOG
    # ========================================================

    print()
    print(
        f"Inference completed in "
        f"{inference_time:.4f} seconds"
    )

    print(
        "Output size:",
        image.size
    )

    print(
        "Output format: PNG"
    )

    print("=" * 70)


    # ========================================================
    # RETURN RESTORED IMAGE
    # ========================================================

    return StreamingResponse(
        buffer,
        media_type="image/png",
        headers={
            "Content-Disposition": (
                'inline; filename="siv_ai_restored.png"'
            ),
            "X-SIV-AI-Input-Format": suffix,
            "X-SIV-AI-Input-Size": "128x128",
            "X-SIV-AI-Output-Size": "256x256",
            "X-SIV-AI-Inference-Time": (
                f"{inference_time:.4f}"
            ),
            "X-SIV-AI-Parameters": str(
                parameter_count
            ),
            "X-SIV-AI-NPY-Supported": "true",
        },
    )


# ============================================================
# OPTIONAL API INFORMATION
# ============================================================

@app.get("/info")
def info():

    return {
        "name": "SIV-AI",
        "full_name": (
            "Semiconductor Inspection and "
            "Verification AI"
        ),
        "model": "Phase-4",
        "parameters": parameter_count,
        "input": {
            "size": "128x128",
            "format": [
                "NPY",
                "PNG",
                "JPG",
                "JPEG",
                "BMP",
                "TIF",
                "TIFF",
                "WEBP",
            ],
        },
        "output": {
            "size": "256x256",
            "format": "PNG",
        },
        "npy_processing": {
            "normalization": "none",
            "clipping_before_inference": False,
            "values_preserved": True,
        },
        "device": str(DEVICE),
        "gpu": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None
        ),
    }