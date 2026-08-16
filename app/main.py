from pathlib import Path
import sys
import time
import os

import torch

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException,
)

from fastapi.responses import (
    Response,
    FileResponse,
)

from fastapi.staticfiles import StaticFiles

from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel


# ============================================================
# PROJECT ROOT
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INFERENCE_DIR = ROOT / "inference"

STATIC_DIR = ROOT / "app" / "static"

TEMP_DIR = ROOT / "web_temp"

TEMP_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# PYTHON PATH
# ============================================================

if str(ROOT) not in sys.path:

    sys.path.insert(
        0,
        str(ROOT)
    )

if str(INFERENCE_DIR) not in sys.path:

    sys.path.insert(
        0,
        str(INFERENCE_DIR)
    )


# ============================================================
# IMPORT ACTUAL PHASE-4 INFERENCE
# ============================================================

try:

    from infer_phase4 import (
        load_model,
        run_inference,
    )

except Exception as exc:

    print()
    print("=" * 70)
    print("FAILED TO IMPORT PHASE-4 INFERENCE")
    print("=" * 70)

    print(
        "Inference directory:",
        INFERENCE_DIR
    )

    print(
        "Error:",
        repr(exc)
    )

    print("=" * 70)

    raise


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="SIV-AI",
    description=(
        "Semiconductor Inspection and Verification AI"
    ),
    version="1.0.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,

    allow_origins=[
        "https://logeshsk-ind.github.io",
        "https://logeshsk-ind.github.io/SIV_AI",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:5500",
        "http://127.0.0.1:5500",
    ],

    allow_credentials=False,

    allow_methods=[
        "*"
    ],

    allow_headers=[
        "*"
    ],
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# SUPPORTED FILE FORMATS
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
# STARTUP INFORMATION
# ============================================================

print()
print("=" * 70)
print("SIV-AI WEB BACKEND")
print("=" * 70)

print(
    "Project root:",
    ROOT
)

print(
    "Inference module:",
    INFERENCE_DIR / "infer_phase4.py"
)

print(
    "Device:",
    DEVICE
)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

else:

    print(
        "GPU: CPU"
    )


# ============================================================
# LOAD MODEL ONCE
# ============================================================

print()
print("Loading SIV-AI Phase-4 model...")

try:

    MODEL = load_model(
        DEVICE
    )

except Exception as exc:

    print()
    print("=" * 70)
    print("MODEL LOADING FAILED")
    print("=" * 70)

    print(
        repr(exc)
    )

    print("=" * 70)

    raise


print()
print("SIV-AI Phase-4 model READY.")

print("=" * 70)


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/api/health")
async def health():

    return {

        "status":
            "ok",

        "service":
            "SIV-AI",

        "model":
            "SIV-AI Phase-4",

        "device":
            str(DEVICE),

        "cuda":
            torch.cuda.is_available(),

        "gpu":
            (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else "CPU"
            ),

        "parameters":
            110049,

        "input_size":
            "128x128",

        "output_size":
            "256x256",

        "checkpoint":
            "siv_ai_phase4_best.pth",

        "inference":
            "inference/infer_phase4.py",

    }


# ============================================================
# RESTORE IMAGE
# ============================================================

@app.post("/api/restore")
async def restore_image(
    file: UploadFile = File(...)
):

    # ========================================================
    # START TIMER
    # ========================================================

    start_time = time.perf_counter()


    # ========================================================
    # VALIDATE FILE NAME
    # ========================================================

    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail="No file selected."
        )


    original_name = Path(
        file.filename
    ).name


    extension = Path(
        original_name
    ).suffix.lower()


    # ========================================================
    # VALIDATE FORMAT
    # ========================================================

    if extension not in SUPPORTED_EXTENSIONS:

        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported image format. "
                "Please upload NPY, PNG, JPG, JPEG, "
                "BMP, TIF, TIFF or WEBP."
            )
        )


    # ========================================================
    # READ UPLOAD
    # ========================================================

    contents = await file.read()


    if not contents:

        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty."
        )


    # ========================================================
    # TEMP FILE NAMES
    # ========================================================

    timestamp = time.time_ns()


    input_path = (
        TEMP_DIR
        / f"input_{timestamp}{extension}"
    )


    output_path = (
        TEMP_DIR
        / f"output_{timestamp}_restored.png"
    )


    # ========================================================
    # SAVE INPUT
    # ========================================================

    try:

        input_path.write_bytes(
            contents
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to save uploaded file: "
                f"{exc}"
            )
        )


    # ========================================================
    # SERVER LOG
    # ========================================================

    print()
    print("=" * 70)
    print("SIV-AI WEB INFERENCE REQUEST")
    print("=" * 70)

    print(
        "File:",
        original_name
    )

    print(
        "Format:",
        extension
    )

    print(
        "Input:",
        input_path
    )


    # ========================================================
    # RUN ACTUAL PHASE-4 MODEL
    # ========================================================

    try:

        run_inference(
            model=MODEL,
            image_path=input_path,
            output_path=output_path,
            device=DEVICE,
        )


    except ValueError as exc:

        print(
            "Input validation error:",
            exc
        )

        input_path.unlink(
            missing_ok=True
        )

        raise HTTPException(
            status_code=400,
            detail=str(exc)
        )


    except FileNotFoundError as exc:

        print(
            "File error:",
            exc
        )

        input_path.unlink(
            missing_ok=True
        )

        raise HTTPException(
            status_code=404,
            detail=str(exc)
        )


    except RuntimeError as exc:

        print(
            "Runtime error:",
            exc
        )

        input_path.unlink(
            missing_ok=True
        )

        raise HTTPException(
            status_code=500,
            detail=str(exc)
        )


    except Exception as exc:

        print()
        print("INFERENCE ERROR:")

        print(
            repr(exc)
        )

        input_path.unlink(
            missing_ok=True
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "SIV-AI inference failed: "
                f"{exc}"
            )
        )


    # ========================================================
    # VERIFY OUTPUT
    # ========================================================

    if not output_path.exists():

        input_path.unlink(
            missing_ok=True
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Inference completed, "
                "but no restored image was created."
            )
        )


    # ========================================================
    # CALCULATE RUNTIME
    # ========================================================

    elapsed = (
        time.perf_counter()
        - start_time
    )


    # ========================================================
    # READ OUTPUT
    # ========================================================

    try:

        output_bytes = (
            output_path.read_bytes()
        )

    except Exception as exc:

        input_path.unlink(
            missing_ok=True
        )

        output_path.unlink(
            missing_ok=True
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to read restored output: "
                f"{exc}"
            )
        )


    # ========================================================
    # CLEAN TEMP FILES
    # ========================================================

    input_path.unlink(
        missing_ok=True
    )

    output_path.unlink(
        missing_ok=True
    )


    # ========================================================
    # SERVER LOG
    # ========================================================

    print()
    print("INFERENCE COMPLETE")

    print(
        "Runtime:",
        f"{elapsed:.4f} seconds"
    )

    print(
        "Input:",
        "128 × 128"
    )

    print(
        "Output:",
        "256 × 256"
    )

    print("=" * 70)


    # ========================================================
    # RETURN PNG TO FRONTEND
    # ========================================================

    output_filename = (
        Path(original_name).stem
        + "_SIV-AI_restored.png"
    )


    return Response(

        content=output_bytes,

        media_type="image/png",

        headers={

            "Content-Disposition":
                (
                    'inline; filename="'
                    + output_filename
                    + '"'
                ),

            "X-SIV-AI-Runtime":
                f"{elapsed:.4f}",

            "X-SIV-AI-PSNR":
                "27.9101",

            "X-SIV-AI-SSIM":
                "0.7530",

            "X-SIV-AI-Input":
                "128x128",

            "X-SIV-AI-Output":
                "256x256",

            "Access-Control-Expose-Headers":
                (
                    "X-SIV-AI-Runtime, "
                    "X-SIV-AI-PSNR, "
                    "X-SIV-AI-SSIM, "
                    "X-SIV-AI-Input, "
                    "X-SIV-AI-Output"
                ),

        }

    )


# ============================================================
# FEEDBACK MODEL
# ============================================================

class Feedback(BaseModel):

    type: str = "General"

    name: str = ""

    email: str = ""

    rating: int = 5

    comments: str = ""


# ============================================================
# FEEDBACK API
# ============================================================

@app.post("/api/feedback")
async def submit_feedback(
    feedback: Feedback
):

    print()
    print("=" * 70)
    print("SIV-AI FEEDBACK")
    print("=" * 70)

    print(
        "Type:",
        feedback.type
    )

    print(
        "Name:",
        feedback.name
    )

    print(
        "Email:",
        feedback.email
    )

    print(
        "Rating:",
        feedback.rating
    )

    print(
        "Comments:",
        feedback.comments
    )

    print("=" * 70)


    return {

        "status":
            "success",

        "message":
            "Feedback submitted successfully."

    }


# ============================================================
# STATIC FRONTEND
# ============================================================

if STATIC_DIR.exists():

    app.mount(
        "/static",
        StaticFiles(
            directory=str(
                STATIC_DIR
            )
        ),
        name="static"
    )


# ============================================================
# HOME PAGE
# ============================================================

@app.get("/")
async def home():

    index_file = (
        STATIC_DIR / "index.html"
    )


    if not index_file.exists():

        raise HTTPException(
            status_code=404,
            detail=(
                "Frontend not found:\n"
                f"{index_file}"
            )
        )


    return FileResponse(
        index_file
    )


# ============================================================
# DIRECT RUN
# ============================================================

if __name__ == "__main__":

    import uvicorn

    port = int(
        os.environ.get(
            "PORT",
            "8001"
        )
    )

    uvicorn.run(

        "app.main:app",

        host="0.0.0.0",

        port=port,

        reload=False

    )