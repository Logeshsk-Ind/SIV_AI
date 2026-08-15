from pathlib import Path
import sys
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset


# ============================================================
# PROJECT
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.kla_dataset import KLADataset
from src.models.siv_ai_phase4 import SIVAI


# ============================================================
# CONFIG
# ============================================================

DATA_DIR = ROOT / "data" / "raw" / "train" / "train"

SOURCE_CHECKPOINT = (
    ROOT
    / "phase4_source"
    / "siv_ai_phase4_weights.pth"
)

OUTPUT_CHECKPOINT = (
    ROOT
    / "phase4_source"
    / "siv_ai_phase4_ssim_best.pth"
)

SEED = 42
VAL_RATIO = 0.20

BATCH_SIZE = 4

EPOCHS = 8

LEARNING_RATE = 2e-5

# Loss weights
L1_WEIGHT = 0.30
SSIM_WEIGHT = 0.55
EDGE_WEIGHT = 0.15


# ============================================================
# SEED
# ============================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# DIFFERENTIABLE SSIM
# ============================================================

def ssim_loss(pred, target):

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = F.avg_pool2d(
        pred,
        kernel_size=11,
        stride=1,
        padding=5
    )

    mu_y = F.avg_pool2d(
        target,
        kernel_size=11,
        stride=1,
        padding=5
    )

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = (
        F.avg_pool2d(
            pred * pred,
            11,
            1,
            5
        )
        - mu_x2
    )

    sigma_y2 = (
        F.avg_pool2d(
            target * target,
            11,
            1,
            5
        )
        - mu_y2
    )

    sigma_xy = (
        F.avg_pool2d(
            pred * target,
            11,
            1,
            5
        )
        - mu_xy
    )

    numerator = (
        (2 * mu_xy + C1)
        * (2 * sigma_xy + C2)
    )

    denominator = (
        (mu_x2 + mu_y2 + C1)
        * (sigma_x2 + sigma_y2 + C2)
    )

    ssim_map = numerator / (denominator + 1e-8)

    return 1.0 - ssim_map.mean()


# ============================================================
# EDGE LOSS
# ============================================================

def gradient_loss(pred, target):

    pred_x = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    pred_y = pred[:, :, 1:, :] - pred[:, :, :-1, :]

    target_x = target[:, :, :, 1:] - target[:, :, :, :-1]
    target_y = target[:, :, 1:, :] - target[:, :, :-1, :]

    loss_x = F.l1_loss(pred_x, target_x)
    loss_y = F.l1_loss(pred_y, target_y)

    return loss_x + loss_y


# ============================================================
# TOTAL LOSS
# ============================================================

def restoration_loss(pred, target):

    l1 = F.l1_loss(pred, target)

    ssim = ssim_loss(pred, target)

    edge = gradient_loss(pred, target)

    total = (
        L1_WEIGHT * l1
        + SSIM_WEIGHT * ssim
        + EDGE_WEIGHT * edge
    )

    return total, l1, ssim, edge


# ============================================================
# MAIN
# ============================================================

def main():

    set_seed(SEED)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 70)
    print("SIV-AI PHASE 4 SSIM FINE-TUNING")
    print("=" * 70)

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

    print()
    print("Dataset:", DATA_DIR)
    print("Total:", len(dataset))

    generator = torch.Generator().manual_seed(SEED)

    indices = torch.randperm(
        len(dataset),
        generator=generator
    ).tolist()

    val_size = int(
        len(dataset) * VAL_RATIO
    )

    train_indices = indices[:-val_size]
    val_indices = indices[-val_size:]

    train_dataset = Subset(
        dataset,
        train_indices
    )

    val_dataset = Subset(
        dataset,
        val_indices
    )

    print("Training:", len(train_dataset))
    print("Validation:", len(val_dataset))

    # --------------------------------------------------------
    # Loaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = SIVAI().to(device)

    params = sum(
        p.numel()
        for p in model.parameters()
    )

    print()
    print("Model parameters:", params)

    assert params == 110049

    # --------------------------------------------------------
    # Load Phase-4 checkpoint
    # --------------------------------------------------------

    print()
    print("Loading:", SOURCE_CHECKPOINT)

    state_dict = torch.load(
        SOURCE_CHECKPOINT,
        map_location=device
    )

    model.load_state_dict(
        state_dict,
        strict=True
    )

    print("Checkpoint loaded successfully.")

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-5
    )

    # --------------------------------------------------------
    # Scheduler
    # --------------------------------------------------------

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=3e-6
    )

    # --------------------------------------------------------
    # Best tracking
    # --------------------------------------------------------

    best_val_loss = float("inf")

    # ========================================================
    # TRAIN
    # ========================================================

    for epoch in range(EPOCHS):

        model.train()

        running_loss = 0.0
        running_l1 = 0.0
        running_ssim = 0.0
        running_edge = 0.0

        for batch_idx, batch in enumerate(train_loader):

            noisy = batch["noisy"].to(
                device,
                non_blocking=True
            )

            gt = batch["gt"].to(
                device,
                non_blocking=True
            )

            prediction = model(noisy)

            # Keep valid output range
            prediction = prediction.clamp(0, 1)

            loss, l1, ssim, edge = restoration_loss(
                prediction,
                gt
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0
            )

            optimizer.step()

            running_loss += loss.item()
            running_l1 += l1.item()
            running_ssim += ssim.item()
            running_edge += edge.item()

            if (batch_idx + 1) % 100 == 0:

                print(
                    f"Epoch [{epoch+1}/{EPOCHS}] "
                    f"Batch [{batch_idx+1}/{len(train_loader)}] "
                    f"Loss={loss.item():.5f} "
                    f"L1={l1.item():.5f} "
                    f"SSIMLoss={ssim.item():.5f}"
                )

        scheduler.step()

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        model.eval()

        val_loss = 0.0

        with torch.no_grad():

            for batch in val_loader:

                noisy = batch["noisy"].to(device)
                gt = batch["gt"].to(device)

                prediction = model(noisy)

                prediction = prediction.clamp(0, 1)

                loss, _, _, _ = restoration_loss(
                    prediction,
                    gt
                )

                val_loss += loss.item()

        val_loss /= len(val_loader)

        n = len(train_loader)

        print()
        print("=" * 70)
        print(
            f"EPOCH {epoch+1}/{EPOCHS}"
        )
        print(
            f"Train Loss : {running_loss/n:.6f}"
        )
        print(
            f"Train L1   : {running_l1/n:.6f}"
        )
        print(
            f"Train SSIM Loss : {running_ssim/n:.6f}"
        )
        print(
            f"Train Edge : {running_edge/n:.6f}"
        )
        print(
            f"Val Loss   : {val_loss:.6f}"
        )
        print(
            f"LR         : {scheduler.get_last_lr()[0]:.8f}"
        )
        print("=" * 70)

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            torch.save(
                model.state_dict(),
                OUTPUT_CHECKPOINT
            )

            print(
                "BEST CHECKPOINT SAVED:",
                OUTPUT_CHECKPOINT
            )

    print()
    print("=" * 70)
    print("SSIM FINE-TUNING COMPLETED")
    print("=" * 70)
    print(
        "Best validation loss:",
        best_val_loss
    )


if __name__ == "__main__":
    main()