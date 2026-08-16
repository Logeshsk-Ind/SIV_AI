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
# IMPORT PHASE-4 INFERENCE
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
# FASTAPI APPLICATION
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

# IMPORTANT:
#
# Browser Origin for GitHub Pages is:
#
# https://logeshsk-ind.github.io
#
# The /SIV_AI repository path is NOT part of the origin.
#
# Therefore DO NOT use:
#
# https://logeshsk-ind.github.io/SIV_AI
#
# in allow_origins.


ALLOWED_ORIGINS = [

    # --------------------------------------------------------
    # GitHub Pages
    # --------------------------------------------------------

    "https://logeshsk-ind.github.io",

    # --------------------------------------------------------
    # Local development
    # --------------------------------------------------------

    "http://localhost:8000",

    "http://127.0.0.1:8000",

    "http://localhost:5500",

    "http://127.0.0.1:5500",

    "http://localhost:3000",

    "http://127.0.0.1:3000",

]


app.add_middleware(

    CORSMiddleware,

    allow_origins=ALLOWED_ORIGINS,

    allow_credentials=False,

    allow_methods=[
        "*"
    ],

    allow_headers=[
        "*"
    ],

    expose_headers=[

        "X-SIV-AI-Runtime",

        "X-SIV-AI-PSNR",

        "X-SIV-AI-SSIM",

        "X-SIV-AI-Input",

        "X-SIV-AI-Output",

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
# MODEL INFORMATION
# ============================================================

MODEL_NAME = (
    "SIV-AI Phase-4"
)


MODEL_PARAMETERS = (
    110049
)


MODEL_INPUT_SIZE = (
    "128x128"
)


MODEL_OUTPUT_SIZE = (
    "256x256"
)


MODEL_CHECKPOINT = (
    "siv_ai_phase4_best.pth"
)


MODEL_INFERENCE = (
    "inference/infer_phase4.py"
)


# ============================================================
# OFFICIAL MODEL METRICS
# ============================================================

MODEL_PSNR = (
    "27.9101"
)


MODEL_SSIM = (
    "0.7530"
)


# ============================================================
# STARTUP INFORMATION
# ============================================================

print()

print("=" * 70)

print(
    "SIV-AI WEB BACKEND"
)

print("=" * 70)

print()

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

print(
    "Checkpoint:",
    ROOT / MODEL_CHECKPOINT
)

print(
    "Input size:",
    MODEL_INPUT_SIZE
)

print(
    "Output size:",
    MODEL_OUTPUT_SIZE
)

print()


if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

else:

    print(
        "GPU: CPU"
    )


print()


# ============================================================
# LOAD MODEL ONCE
# ============================================================

print(
    "Loading SIV-AI Phase-4 model..."
)

try:

    MODEL = load_model(
        DEVICE
    )

except Exception as exc:

    print()

    print("=" * 70)

    print(
        "MODEL LOADING FAILED"
    )

    print("=" * 70)

    print(
        repr(exc)
    )

    print("=" * 70)

    raise


print()

print(
    "SIV-AI Phase-4 model READY."
)

print("=" * 70)

print()


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
async def root():

    return {

        "service":
            "SIV-AI",

        "status":
            "online",

        "model":
            MODEL_NAME,

        "version":
            "1.0.0",

        "docs":
            "/docs",

        "health":
            "/api/health",

        "restore":
            "/api/restore",

    }


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
            MODEL_NAME,

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
            MODEL_PARAMETERS,

        "input_size":
            MODEL_INPUT_SIZE,

        "output_size":
            MODEL_OUTPUT_SIZE,

        "checkpoint":
            MODEL_CHECKPOINT,

        "inference":
            MODEL_INFERENCE,

        "psnr":
            MODEL_PSNR,

        "ssim":
            MODEL_SSIM,

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

    start_time = (
        time.perf_counter()
    )


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
    # VALIDATE FILE FORMAT
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

    try:

        contents = await file.read()

    except Exception as exc:

        raise HTTPException(

            status_code=400,

            detail=(

                "Unable to read uploaded file: "

                f"{exc}"

            )

        )


    if not contents:

        raise HTTPException(

            status_code=400,

            detail="Uploaded file is empty."

        )


    # ========================================================
    # CREATE UNIQUE TEMP FILE NAMES
    # ========================================================

    timestamp = (
        time.time_ns()
    )


    input_path = (

        TEMP_DIR

        / (
            f"input_{timestamp}"
            f"{extension}"
        )

    )


    output_path = (

        TEMP_DIR

        / (
            f"output_{timestamp}"
            "_restored.png"
        )

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

    print(
        "SIV-AI WEB INFERENCE REQUEST"
    )

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

    print()


    # ========================================================
    # RUN PHASE-4 MODEL
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

        print(
            "INFERENCE ERROR:"
        )

        print(
            repr(exc)
        )

        print()

        input_path.unlink(
            missing_ok=True
        )

        output_path.unlink(
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

    print(
        "INFERENCE COMPLETE"
    )

    print()

    print(
        "Runtime:",
        f"{elapsed:.4f} seconds"
    )

    print(
        "Input:",
        MODEL_INPUT_SIZE
    )

    print(
        "Output:",
        MODEL_OUTPUT_SIZE
    )

    print(
        "PSNR:",
        f"{MODEL_PSNR} dB"
    )

    print(
        "SSIM:",
        MODEL_SSIM
    )

    print("=" * 70)

    print()


    # ========================================================
    # OUTPUT FILENAME
    # ========================================================

    output_filename = (

        Path(
            original_name
        ).stem

        + "_SIV-AI_restored.png"

    )


    # ========================================================
    # RETURN RESTORED PNG
    # ========================================================

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
                MODEL_PSNR,

            "X-SIV-AI-SSIM":
                MODEL_SSIM,

            "X-SIV-AI-Input":
                MODEL_INPUT_SIZE,

            "X-SIV-AI-Output":
                MODEL_OUTPUT_SIZE,

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

    print(
        "SIV-AI FEEDBACK"
    )

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

    print()


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
# OPTIONAL LOCAL FRONTEND
# ============================================================

@app.get("/app")
async def app_home():

    index_file = (

        STATIC_DIR
        / "index.html"

    )


    if not index_file.exists():

        raise HTTPException(

            status_code=404,

            detail=(

                "Frontend not found: "

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


    print()

    print("=" * 70)

    print(
        "STARTING SIV-AI SERVER"
    )

    print(
        f"PORT: {port}"
    )

    print("=" * 70)

    print()


    uvicorn.run(

        "app.main:app",

        host="0.0.0.0",

        port=port,

        reload=False

    )