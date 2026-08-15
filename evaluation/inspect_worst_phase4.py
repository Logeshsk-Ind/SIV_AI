from pathlib import Path
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.kla_dataset import KLADataset
from src.models.siv_ai_phase4 import SIVAI


# ============================================================
# CONFIG
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

OUTPUT_DIR = ROOT / "evaluation" / "phase4_worst_cases"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

SEED = 42
VAL_RATIO = 0.20

WORST_NAMES = [
    "000958",
    "000627",
    "002982",
    "002975",
    "002974",
]


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)


# ============================================================
# DATASET
# ============================================================

dataset = KLADataset(DATA_DIR)

# Reproduce EXACT validation split
generator = torch.Generator().manual_seed(SEED)

indices = torch.randperm(
    len(dataset),
    generator=generator
).tolist()

val_size = int(
    len(dataset) * VAL_RATIO
)

val_indices = indices[-val_size:]

# Dataset index lookup
name_to_index = {}

for idx in val_indices:

    sample = dataset[idx]

    name_to_index[
        sample["name"]
    ] = idx


# ============================================================
# MODEL
# ============================================================

model = SIVAI().to(device)

state_dict = torch.load(
    CHECKPOINT,
    map_location=device
)

model.load_state_dict(
    state_dict,
    strict=True
)

model.eval()


# ============================================================
# PROCESS
# ============================================================

for name in WORST_NAMES:

    if name not in name_to_index:

        print(
            f"WARNING: {name} "
            f"not found in validation set"
        )

        continue

    idx = name_to_index[name]

    sample = dataset[idx]

    noisy = sample["noisy"].unsqueeze(0).to(device)
    gt = sample["gt"].unsqueeze(0).to(device)

    with torch.no_grad():

        prediction = model(noisy)

    prediction = prediction.clamp(
        0,
        1
    )

    noisy_np = (
        noisy[0, 0]
        .cpu()
        .numpy()
    )

    pred_np = (
        prediction[0, 0]
        .cpu()
        .numpy()
    )

    gt_np = (
        gt[0, 0]
        .cpu()
        .numpy()
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("SAMPLE:", name)
    print("=" * 70)

    print(
        "Noisy:",
        noisy_np.shape,
        "min=",
        noisy_np.min(),
        "max=",
        noisy_np.max(),
        "mean=",
        noisy_np.mean(),
        "std=",
        noisy_np.std()
    )

    print(
        "Prediction:",
        pred_np.shape,
        "min=",
        pred_np.min(),
        "max=",
        pred_np.max(),
        "mean=",
        pred_np.mean(),
        "std=",
        pred_np.std()
    )

    print(
        "GT:",
        gt_np.shape,
        "min=",
        gt_np.min(),
        "max=",
        gt_np.max(),
        "mean=",
        gt_np.mean(),
        "std=",
        gt_np.std()
    )

    # --------------------------------------------------------
    # Save individual images
    # --------------------------------------------------------

    np.save(
        OUTPUT_DIR / f"{name}_noisy.npy",
        noisy_np
    )

    np.save(
        OUTPUT_DIR / f"{name}_prediction.npy",
        pred_np
    )

    np.save(
        OUTPUT_DIR / f"{name}_gt.npy",
        gt_np
    )

    # --------------------------------------------------------
    # Visual comparison
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5)
    )

    axes[0].imshow(
        noisy_np,
        cmap="gray"
    )

    axes[0].set_title(
        f"NoisyLR\n{name}"
    )

    axes[1].imshow(
        pred_np,
        cmap="gray"
    )

    axes[1].set_title(
        "Phase-4 Prediction"
    )

    axes[2].imshow(
        gt_np,
        cmap="gray"
    )

    axes[2].set_title(
        "Ground Truth"
    )

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()

    output_file = (
        OUTPUT_DIR
        / f"{name}_comparison.png"
    )

    plt.savefig(
        output_file,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "Saved:",
        output_file
    )


print()
print("=" * 70)
print("WORST CASE INSPECTION COMPLETED")
print("=" * 70)