from pathlib import Path
import sys
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset


# ============================================================
# PROJECT ROOT
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ============================================================
# IMPORTS
# ============================================================

from src.data.kla_dataset import KLADataset
from src.models.wavelet_sr_v2 import WaveletSRV2


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = (
    ROOT
    / "data"
    / "raw"
    / "train"
    / "train"
)

CHECKPOINT_DIR = ROOT / "checkpoints"

CHECKPOINT = (
    CHECKPOINT_DIR
    / "wavelet_sr_v2_best.pth"
)

BATCH_SIZE = 4

EPOCHS = 30

LEARNING_RATE = 1e-4

VAL_RATIO = 0.20

SEED = 42

NUM_WORKERS = 0


# ============================================================
# LOSS WEIGHTS
# ============================================================

L1_WEIGHT = 1.0

SSIM_WEIGHT = 0.2

EDGE_WEIGHT = 0.1


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# SSIM
# ============================================================

def ssim_loss(pred, target):

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = F.avg_pool2d(
        pred,
        kernel_size=7,
        stride=1,
        padding=3
    )

    mu_y = F.avg_pool2d(
        target,
        kernel_size=7,
        stride=1,
        padding=3
    )

    sigma_x = (
        F.avg_pool2d(
            pred * pred,
            kernel_size=7,
            stride=1,
            padding=3
        )
        - mu_x * mu_x
    )

    sigma_y = (
        F.avg_pool2d(
            target * target,
            kernel_size=7,
            stride=1,
            padding=3
        )
        - mu_y * mu_y
    )

    sigma_xy = (
        F.avg_pool2d(
            pred * target,
            kernel_size=7,
            stride=1,
            padding=3
        )
        - mu_x * mu_y
    )

    numerator = (
        (2 * mu_x * mu_y + C1)
        * (2 * sigma_xy + C2)
    )

    denominator = (
        (mu_x * mu_x + mu_y * mu_y + C1)
        * (sigma_x + sigma_y + C2)
    )

    ssim = numerator / (denominator + 1e-8)

    return 1.0 - ssim.mean()


# ============================================================
# EDGE LOSS
# ============================================================

def edge_loss(pred, target):

    sobel_x = torch.tensor(
        [
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1],
        ],
        dtype=pred.dtype,
        device=pred.device
    ).view(1, 1, 3, 3)

    sobel_y = torch.tensor(
        [
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1],
        ],
        dtype=pred.dtype,
        device=pred.device
    ).view(1, 1, 3, 3)

    pred_x = F.conv2d(
        pred,
        sobel_x,
        padding=1
    )

    pred_y = F.conv2d(
        pred,
        sobel_y,
        padding=1
    )

    target_x = F.conv2d(
        target,
        sobel_x,
        padding=1
    )

    target_y = F.conv2d(
        target,
        sobel_y,
        padding=1
    )

    pred_edge = torch.sqrt(
        pred_x ** 2
        + pred_y ** 2
        + 1e-6
    )

    target_edge = torch.sqrt(
        target_x ** 2
        + target_y ** 2
        + 1e-6
    )

    return F.l1_loss(
        pred_edge,
        target_edge
    )


# ============================================================
# COMBINED LOSS
# ============================================================

def combined_loss(pred, target):

    l1 = F.l1_loss(
        pred,
        target
    )

    ssim = ssim_loss(
        pred,
        target
    )

    edge = edge_loss(
        pred,
        target
    )

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

    print()
    print("=" * 70)
    print("SIV-AI WAVELETSR V2 TRAINING")
    print("=" * 70)

    # --------------------------------------------------------
    # Seed
    # --------------------------------------------------------

    set_seed(SEED)

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
    # Checkpoint directory
    # --------------------------------------------------------

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset = KLADataset(
        DATA_DIR
    )

    print(
        "Total dataset:",
        len(dataset)
    )

    # --------------------------------------------------------
    # Train / validation split
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

    print(
        "Training samples:",
        len(train_dataset)
    )

    print(
        "Validation samples:",
        len(val_dataset)
    )

    # --------------------------------------------------------
    # Data loaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda")
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda")
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = WaveletSRV2().to(device)

    parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        "Model parameters:",
        parameters
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4
    )

    # --------------------------------------------------------
    # Scheduler
    # --------------------------------------------------------

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS
    )

    # --------------------------------------------------------
    # Best validation loss
    # --------------------------------------------------------

    best_val_loss = float("inf")

    # ========================================================
    # TRAINING LOOP
    # ========================================================

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        # ----------------------------------------------------
        # Training
        # ----------------------------------------------------

        model.train()

        train_total = 0.0

        train_l1 = 0.0

        train_ssim = 0.0

        train_edge = 0.0

        for batch_idx, batch in enumerate(
            train_loader,
            start=1
        ):

            noisy = batch["noisy"].to(
                device,
                non_blocking=True
            )

            gt = batch["gt"].to(
                device,
                non_blocking=True
            )

            # Forward
            prediction = model(
                noisy
            )

            # Keep output stable
            prediction = prediction.clamp(
                0,
                1
            )

            # Loss
            loss, l1, ssim, edge = combined_loss(
                prediction,
                gt
            )

            # Backpropagation
            optimizer.zero_grad(
                set_to_none=True
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            train_total += loss.item()

            train_l1 += l1.item()

            train_ssim += ssim.item()

            train_edge += edge.item()

            if batch_idx % 100 == 0:

                print(
                    f"Epoch [{epoch}/{EPOCHS}] "
                    f"Batch [{batch_idx}/{len(train_loader)}] "
                    f"Loss: {loss.item():.6f}"
                )

        # ----------------------------------------------------
        # Average training losses
        # ----------------------------------------------------

        train_total /= len(train_loader)

        train_l1 /= len(train_loader)

        train_ssim /= len(train_loader)

        train_edge /= len(train_loader)

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        model.eval()

        val_total = 0.0

        val_l1 = 0.0

        val_ssim = 0.0

        val_edge = 0.0

        with torch.no_grad():

            for batch in val_loader:

                noisy = batch["noisy"].to(
                    device,
                    non_blocking=True
                )

                gt = batch["gt"].to(
                    device,
                    non_blocking=True
                )

                prediction = model(
                    noisy
                )

                prediction = prediction.clamp(
                    0,
                    1
                )

                loss, l1, ssim, edge = combined_loss(
                    prediction,
                    gt
                )

                val_total += loss.item()

                val_l1 += l1.item()

                val_ssim += ssim.item()

                val_edge += edge.item()

        val_total /= len(val_loader)

        val_l1 /= len(val_loader)

        val_ssim /= len(val_loader)

        val_edge /= len(val_loader)

        # ----------------------------------------------------
        # Scheduler
        # ----------------------------------------------------

        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        # ----------------------------------------------------
        # Results
        # ----------------------------------------------------

        print()
        print("-" * 70)

        print(
            f"Epoch [{epoch}/{EPOCHS}]"
        )

        print(
            f"Train Total : {train_total:.6f}"
        )

        print(
            f"Train L1    : {train_l1:.6f}"
        )

        print(
            f"Train SSIM  : {train_ssim:.6f}"
        )

        print(
            f"Train Edge  : {train_edge:.6f}"
        )

        print(
            f"Val Total   : {val_total:.6f}"
        )

        print(
            f"Val L1      : {val_l1:.6f}"
        )

        print(
            f"Val SSIM    : {val_ssim:.6f}"
        )

        print(
            f"Val Edge    : {val_edge:.6f}"
        )

        print(
            f"Learning Rate: {current_lr:.8f}"
        )

        # ----------------------------------------------------
        # Save best checkpoint
        # ----------------------------------------------------

        if val_total < best_val_loss:

            best_val_loss = val_total

            checkpoint = {

                "epoch":
                    epoch,

                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "scheduler_state_dict":
                    scheduler.state_dict(),

                "train_loss":
                    train_total,

                "val_loss":
                    val_total,

                "train_l1":
                    train_l1,

                "train_ssim":
                    train_ssim,

                "train_edge":
                    train_edge,

                "val_l1":
                    val_l1,

                "val_ssim":
                    val_ssim,

                "val_edge":
                    val_edge,

                "seed":
                    SEED,

                "val_ratio":
                    VAL_RATIO,

                "batch_size":
                    BATCH_SIZE,

                "learning_rate":
                    LEARNING_RATE,

                "model":
                    "WaveletSRV2",

                "l1_weight":
                    L1_WEIGHT,

                "ssim_weight":
                    SSIM_WEIGHT,

                "edge_weight":
                    EDGE_WEIGHT,
            }

            torch.save(
                checkpoint,
                CHECKPOINT
            )

            print()
            print(
                "BEST CHECKPOINT SAVED:"
            )

            print(
                CHECKPOINT
            )

        print("-" * 70)

    # ========================================================
    # COMPLETE
    # ========================================================

    print()
    print("=" * 70)
    print("WAVELETSR V2 TRAINING COMPLETED")
    print("=" * 70)

    print(
        "Best validation loss:",
        f"{best_val_loss:.6f}"
    )

    print(
        "Checkpoint:",
        CHECKPOINT
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()