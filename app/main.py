from pathlib import Path
import sys, time, io

import torch
import numpy as np

from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "siv_ai_phase4_best.pth"
STATIC_DIR = Path(__file__).resolve().parent / "static"

sys.path.insert(0, str(ROOT))
from src.models.siv_ai_phase4 import SIVAI

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("SIV-AI WEB BACKEND")
print("=" * 60)
print("Device:", DEVICE)

model = SIVAI().to(DEVICE)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False
)

model.load_state_dict(
    checkpoint["model_state_dict"]
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
    else checkpoint
)

model.eval()

print("Phase-4 checkpoint loaded")
print("=" * 60)

app = FastAPI(
    title="SIV-AI",
    description="Semiconductor Image Restoration and Verification AI",
    version="1.0"
)

app.mount(
    "/static",
    StaticFiles(directory=STATIC_DIR),
    name="static"
)

IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg",
    ".bmp", ".tif", ".tiff", ".webp"
}

NPY_EXTENSIONS = {".npy"}


# ============================================================
# NPY LOADING
# ============================================================

def load_npy(data: bytes):
    try:
        array = np.load(
            io.BytesIO(data),
            allow_pickle=False
        )
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Could not read NPY file: {e}"
        )

    array = np.asarray(array)

    if not np.issubdtype(array.dtype, np.number):
        raise HTTPException(
            status_code=400,
            detail="NPY must contain numeric image data."
        )

    if array.ndim == 3:
        if array.shape[-1] == 1:
            array = array[:, :, 0]
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported NPY shape: {array.shape}"
            )

    if array.ndim != 2:
        raise HTTPException(
            status_code=400,
            detail=f"Expected 2D grayscale NPY, got {array.shape}"
        )

    if not np.isfinite(array).all():
        raise HTTPException(
            status_code=400,
            detail="NPY contains NaN or Inf values."
        )

    return array.astype(np.float32)


# ============================================================
# NPY -> PNG FOR BROWSER PREVIEW ONLY
# ============================================================

def npy_to_png(array):
    lo = float(np.percentile(array, 1))
    hi = float(np.percentile(array, 99))

    if hi <= lo:
        lo = float(array.min())
        hi = float(array.max())

    if hi <= lo:
        preview = np.zeros_like(array, dtype=np.uint8)
    else:
        preview = np.clip(
            (array - lo) / (hi - lo),
            0,
            1
        )

        preview = (
            preview * 255
        ).astype(np.uint8)

    image = Image.fromarray(preview, mode="L")

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)

    return buffer.getvalue()


# ============================================================
# NORMAL IMAGE -> MODEL TENSOR
# ============================================================

def image_to_tensor(image):
    image = image.convert("L")

    image = image.resize(
        (128, 128),
        Image.Resampling.BICUBIC
    )

    array = np.asarray(
        image,
        dtype=np.float32
    ) / 255.0

    tensor = torch.from_numpy(array)[None, None]

    return tensor.to(DEVICE)


# ============================================================
# NPY -> MODEL TENSOR
# ============================================================

def npy_to_tensor(array):

    if array.shape != (128, 128):
        raise HTTPException(
            status_code=400,
            detail=(
                "Phase-4 expects a 128 × 128 NoisyLR input. "
                f"Received {array.shape}."
            )
        )

    # IMPORTANT:
    # Preserve original NPY values.
    tensor = torch.from_numpy(
        array.astype(np.float32)
    )[None, None]

    return tensor.to(DEVICE)


# ============================================================
# OUTPUT -> PNG
# ============================================================

def tensor_to_png(output):

    output = output.detach().cpu().squeeze()

    output = output.clamp(0, 1)

    array = (
        output.numpy() * 255
    ).round().astype(np.uint8)

    image = Image.fromarray(
        array,
        mode="L"
    )

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)

    return buffer.getvalue()


# ============================================================
# HEALTH
# ============================================================

@app.get("/api/health")
def health():

    return {
        "status": "ok",
        "model": "SIV-AI Phase 4",
        "device": str(DEVICE),
        "parameters": sum(
            p.numel() for p in model.parameters()
        )
    }


# ============================================================
# PREVIEW
# ============================================================

@app.post("/api/preview")
async def preview(file: UploadFile = File(...)):

    name = file.filename or ""
    suffix = Path(name).suffix.lower()
    data = await file.read()

    if suffix == ".npy":

        array = load_npy(data)

        png = npy_to_png(array)

        headers = {
            "X-Input-Size":
                f"{array.shape[1]}x{array.shape[0]}"
        }

        return StreamingResponse(
            io.BytesIO(png),
            media_type="image/png",
            headers=headers
        )

    if suffix in IMAGE_EXTENSIONS:

        try:
            image = Image.open(
                io.BytesIO(data)
            )
            image.load()
        except Exception:
            raise HTTPException(
                status_code=400,
                detail="Could not read image."
            )

        buffer = io.BytesIO()

        image.convert("L").save(
            buffer,
            format="PNG"
        )

        buffer.seek(0)

        headers = {
            "X-Input-Size":
                f"{image.width}x{image.height}"
        }

        return StreamingResponse(
            buffer,
            media_type="image/png",
            headers=headers
        )

    raise HTTPException(
        status_code=400,
        detail="Unsupported file format."
    )


# ============================================================
# RESTORE
# ============================================================

@app.post("/api/restore")
async def restore(file: UploadFile = File(...)):

    name = file.filename or ""
    suffix = Path(name).suffix.lower()
    data = await file.read()

    # ---------------- NPY ----------------

    if suffix == ".npy":

        array = load_npy(data)

        tensor = npy_to_tensor(array)

        width = array.shape[1]
        height = array.shape[0]

    # ---------------- IMAGE ----------------

    elif suffix in IMAGE_EXTENSIONS:

        try:
            image = Image.open(
                io.BytesIO(data)
            )
            image.load()
        except Exception:
            raise HTTPException(
                status_code=400,
                detail="Could not read image."
            )

        width, height = image.size

        tensor = image_to_tensor(image)

    else:

        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported file format. "
                "Use NPY, PNG, JPG, JPEG, BMP, TIFF or WEBP."
            )
        )

    # ---------------- INFERENCE ----------------

    start = time.perf_counter()

    with torch.inference_mode():
        prediction = model(tensor)

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    output = tensor_to_png(prediction)

    return StreamingResponse(
        io.BytesIO(output),
        media_type="image/png",
        headers={
            "X-Inference-Time": f"{elapsed:.4f}",
            "X-Input-Size": f"{width}x{height}",
            "X-Output-Size": "256x256"
        }
    )


# ============================================================
# HOME
# ============================================================

@app.get("/")
def root():

    return StreamingResponse(
        open(
            STATIC_DIR / "index.html",
            "rb"
        ),
        media_type="text/html"
    )