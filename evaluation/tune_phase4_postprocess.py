from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader, Subset
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio


# ============================================================
# ROOT
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT))

from src.data.kla_dataset import KLADataset
from src.models.siv_ai_phase4 import SIVAI


# ============================================================
# CONFIG
# ============================================================

DATA_DIR = (
    ROOT / "data" / "raw" / "train" / "train"
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
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("SIV-AI PHASE 4 POST-PROCESSING SWEEP")
print("=" * 70)

print("Device:", device)

if device.type == "cuda":
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# DATASET
# ============================================================

dataset = KLADataset(DATA_DIR)

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

loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=(device.type == "cuda")
)

print("Dataset:", len(dataset))
print("Validation:", len(val_dataset))


# ============================================================
# MODEL
# ============================================================

model = SIVAI().to(device)

state = torch.load(
    CHECKPOINT,
    map_location=device
)

model.load_state_dict(
    state,
    strict=True
)

model.eval()

print("Checkpoint loaded.")


# ============================================================
# CACHE
# ============================================================

predictions = []
targets = []
inputs = []


with torch.no_grad():

    for batch in loader:

        noisy = batch["noisy"].to(device)

        gt = batch["gt"]

        pred = model(noisy)

        pred = pred.clamp(0, 1)

        predictions.append(
            pred.cpu().numpy()
        )

        targets.append(
            gt.numpy()
        )

        inputs.append(
            noisy.cpu().numpy()
        )


predictions = np.concatenate(
    predictions,
    axis=0
)

targets = np.concatenate(
    targets,
    axis=0
)

inputs = np.concatenate(
    inputs,
    axis=0
)


print(
    "Cached:",
    predictions.shape
)


# ============================================================
# METRICS
# ============================================================

def evaluate(pred):

    psnr_values = []
    ssim_values = []

    for p, t in zip(
        pred,
        targets
    ):

        p = np.clip(p[0], 0, 1)
        t = t[0]

        psnr_values.append(
            peak_signal_noise_ratio(
                t,
                p,
                data_range=1.0
            )
        )

        ssim_values.append(
            structural_similarity(
                t,
                p,
                data_range=1.0
            )
        )

    return (
        float(np.mean(psnr_values)),
        float(np.mean(ssim_values))
    )


# ============================================================
# BASELINE
# ============================================================

base_psnr, base_ssim = evaluate(
    predictions
)

print()
print("=" * 70)
print("BASELINE")
print("=" * 70)

print(
    f"PSNR : {base_psnr:.6f}"
)

print(
    f"SSIM : {base_ssim:.6f}"
)


best = {
    "name": "baseline",
    "psnr": base_psnr,
    "ssim": base_ssim
}


# ============================================================
# 1. CONTRAST SWEEP
# ============================================================

print()
print("=" * 70)
print("1. CONTRAST SWEEP")
print("=" * 70)

for alpha in np.arange(
    0.80,
    1.81,
    0.05
):

    out = (
        0.5
        + alpha * (
            predictions - 0.5
        )
    )

    out = np.clip(
        out,
        0,
        1
    )

    psnr, ssim = evaluate(out)

    print(
        f"alpha={alpha:.2f} | "
        f"PSNR={psnr:.5f} | "
        f"SSIM={ssim:.5f}"
    )

    if ssim > best["ssim"]:

        best = {
            "name": f"contrast_{alpha:.2f}",
            "psnr": psnr,
            "ssim": ssim
        }


# ============================================================
# 2. GAMMA SWEEP
# ============================================================

print()
print("=" * 70)
print("2. GAMMA SWEEP")
print("=" * 70)

for gamma in [
    0.70,
    0.80,
    0.90,
    1.00,
    1.10,
    1.20,
    1.30
]:

    safe = np.clip(
        predictions,
        0,
        1
    )

    out = np.power(
        safe,
        gamma
    )

    psnr, ssim = evaluate(out)

    print(
        f"gamma={gamma:.2f} | "
        f"PSNR={psnr:.5f} | "
        f"SSIM={ssim:.5f}"
    )

    if ssim > best["ssim"]:

        best = {
            "name": f"gamma_{gamma:.2f}",
            "psnr": psnr,
            "ssim": ssim
        }


# ============================================================
# 3. CONTRAST + GAMMA
# ============================================================

print()
print("=" * 70)
print("3. CONTRAST + GAMMA")
print("=" * 70)

for alpha in [
    1.05,
    1.10,
    1.15,
    1.20,
    1.25
]:

    contrast = (
        0.5
        + alpha * (
            predictions - 0.5
        )
    )

    contrast = np.clip(
        contrast,
        0,
        1
    )

    for gamma in [
        0.85,
        0.90,
        0.95,
        1.05,
        1.10,
        1.15
    ]:

        out = np.power(
            contrast,
            gamma
        )

        psnr, ssim = evaluate(out)

        print(
            f"alpha={alpha:.2f} "
            f"gamma={gamma:.2f} | "
            f"PSNR={psnr:.5f} | "
            f"SSIM={ssim:.5f}"
        )

        if ssim > best["ssim"]:

            best = {
                "name":
                    f"contrast_{alpha:.2f}_gamma_{gamma:.2f}",
                "psnr": psnr,
                "ssim": ssim
            }


# ============================================================
# 4. BICUBIC BLEND
# ============================================================

print()
print("=" * 70)
print("4. SIV-AI + BICUBIC BLEND")
print("=" * 70)

input_tensor = torch.from_numpy(
    inputs
).to(device)


with torch.no_grad():

    bicubic = F.interpolate(
        input_tensor,
        size=(256, 256),
        mode="bicubic",
        align_corners=False
    )

bicubic = (
    bicubic
    .clamp(0, 1)
    .cpu()
    .numpy()
)


for alpha in np.arange(
    0.00,
    1.01,
    0.05
):

    out = (
        alpha * predictions
        + (1.0 - alpha) * bicubic
    )

    out = np.clip(
        out,
        0,
        1
    )

    psnr, ssim = evaluate(out)

    print(
        f"SIV={alpha:.2f} "
        f"BICUBIC={1-alpha:.2f} | "
        f"PSNR={psnr:.5f} | "
        f"SSIM={ssim:.5f}"
    )

    if ssim > best["ssim"]:

        best = {
            "name":
                f"blend_siv_{alpha:.2f}",
            "psnr": psnr,
            "ssim": ssim
        }


# ============================================================
# RESULT
# ============================================================

print()
print("=" * 70)
print("BEST RESULT")
print("=" * 70)

print(
    "Method:",
    best["name"]
)

print(
    f"PSNR : {best['psnr']:.6f}"
)

print(
    f"SSIM : {best['ssim']:.6f}"
)

print(
    f"SSIM gain: "
    f"{best['ssim'] - base_ssim:+.6f}"
)

print("=" * 70)