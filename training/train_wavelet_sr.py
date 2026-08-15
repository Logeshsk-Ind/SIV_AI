from pathlib import Path
import sys
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset


# ============================================================
# Project root
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ============================================================
# Project imports
# ============================================================

from src.data.kla_dataset import KLADataset
from src.models.wavelet_sr import WaveletSR


# ============================================================
# Configuration
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
    / "wavelet_sr_best.pth"
)

BATCH_SIZE = 4

EPOCHS = 20

LEARNING_RATE = 1e-4

VAL_RATIO = 0.20

SEED = 42

NUM_WORKERS = 0


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Main
# ============================================================

def main():

    print()
    print("=" * 60)
    print("SIV-AI WaveletSR Training")
    print("=" * 60)

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

    dataset = KLADataset(DATA_DIR)

    print(
        "Total dataset size:",
        len(dataset)
    )

    # --------------------------------------------------------
    # Deterministic train / validation split
    # --------------------------------------------------------

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
        pin_memory=(device.type == "cuda"),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = WaveletSR().to(device)

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
    # Loss
    # --------------------------------------------------------

    criterion = nn.L1Loss()

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4,
    )

    # --------------------------------------------------------
    # Best validation loss
    # --------------------------------------------------------

    best_val_loss = float("inf")

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    print()
    print(
        "WaveletSR training starts"
    )

    for epoch in range(1, EPOCHS + 1):

        # ====================================================
        # Training mode
        # ====================================================

        model.train()

        train_loss_sum = 0.0

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

            # ------------------------------------------------
            # Forward
            # ------------------------------------------------

            prediction = model(noisy)

            # ------------------------------------------------
            # Loss
            # ------------------------------------------------

            loss = criterion(
                prediction,
                gt
            )

            # ------------------------------------------------
            # Backward
            # ------------------------------------------------

            optimizer.zero_grad(
                set_to_none=True
            )

            loss.backward()

            optimizer.step()

            train_loss_sum += (
                loss.item()
            )

            # ------------------------------------------------
            # Progress
            # ------------------------------------------------

            if batch_idx % 100 == 0:

                print(
                    f"Epoch [{epoch}/{EPOCHS}] "
                    f"Train Batch "
                    f"[{batch_idx}/{len(train_loader)}] "
                    f"Loss: {loss.item():.6f}"
                )

        # ====================================================
        # Average training loss
        # ====================================================

        train_loss = (
            train_loss_sum
            / len(train_loader)
        )

        # ====================================================
        # Validation
        # ====================================================

        model.eval()

        val_loss_sum = 0.0

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

                loss = criterion(
                    prediction,
                    gt
                )

                val_loss_sum += (
                    loss.item()
                )

        val_loss = (
            val_loss_sum
            / len(val_loader)
        )

        # ====================================================
        # Epoch results
        # ====================================================

        print()

        print(
            f"Epoch [{epoch}/{EPOCHS}] "
            f"Train Loss: {train_loss:.6f} "
            f"Val Loss: {val_loss:.6f}"
        )

        # ====================================================
        # Save best checkpoint
        # ====================================================

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            checkpoint = {

                "epoch": epoch,

                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "train_loss":
                    train_loss,

                "val_loss":
                    val_loss,

                "seed":
                    SEED,

                "val_ratio":
                    VAL_RATIO,

                "batch_size":
                    BATCH_SIZE,

                "learning_rate":
                    LEARNING_RATE,

                "model":
                    "WaveletSR",

                "restormer_width":
                    32,

                "restormer_blocks":
                    4,

                "restormer_heads":
                    4,

                "nafnet_width":
                    32,

                "nafnet_blocks":
                    5,
            }

            torch.save(
                checkpoint,
                CHECKPOINT
            )

            print(
                "Saved checkpoint:",
                CHECKPOINT
            )

    # ========================================================
    # Complete
    # ========================================================

    print()
    print("=" * 60)
    print("WAVELETSR TRAINING COMPLETED")
    print("=" * 60)

    print(
        "Best validation loss:",
        f"{best_val_loss:.6f}"
    )

    print(
        "Checkpoint:",
        CHECKPOINT
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()