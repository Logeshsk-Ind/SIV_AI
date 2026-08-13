from pathlib import Path
import sys
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset


# --------------------------------------------------
# Add project root to Python path
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.kla_dataset import KLADataset
from models.baseline_sr import BaselineSR


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DATA_DIR = ROOT / "data" / "raw" / "train" / "train"
CHECKPOINT_DIR = ROOT / "checkpoints"

DAY2_CHECKPOINT = CHECKPOINT_DIR / "baseline_day2_best.pth"
DAY3_CHECKPOINT = CHECKPOINT_DIR / "baseline_day3_best.pth"

BATCH_SIZE = 4
EPOCHS = 3
LEARNING_RATE = 1e-4

SEED = 42
VAL_RATIO = 0.2


# --------------------------------------------------
# Reproducibility
# --------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------
# Main training function
# --------------------------------------------------

def main():

    # --------------------------------------------------
    # Set seed
    # --------------------------------------------------

    set_seed(SEED)

    # --------------------------------------------------
    # Select device
    # --------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    # --------------------------------------------------
    # Create checkpoint directory
    # --------------------------------------------------

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------
    # Load dataset
    # --------------------------------------------------

    dataset = KLADataset(DATA_DIR)

    print("Total dataset size:", len(dataset))

    # --------------------------------------------------
    # Train / validation split
    # --------------------------------------------------

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

    print("Training samples:", len(train_dataset))
    print("Validation samples:", len(val_dataset))

    # --------------------------------------------------
    # DataLoaders
    # --------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    # --------------------------------------------------
    # Create model
    # --------------------------------------------------

    model = BaselineSR().to(device)

    # --------------------------------------------------
    # Load Day 2 best checkpoint
    # --------------------------------------------------

    if DAY2_CHECKPOINT.exists():

        checkpoint = torch.load(
            DAY2_CHECKPOINT,
            map_location=device
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        print(
            "Loaded Day 2 checkpoint for Day 3 fine-tuning"
        )

    else:

        print(
            "Day 2 checkpoint not found; "
            "starting from scratch"
        )

    # --------------------------------------------------
    # Loss function
    # --------------------------------------------------

    criterion = nn.L1Loss()

    # --------------------------------------------------
    # Optimizer
    # --------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    # --------------------------------------------------
    # Best validation loss
    # --------------------------------------------------

    best_val_loss = float("inf")

    # --------------------------------------------------
    # Epoch loop
    # --------------------------------------------------

    for epoch in range(EPOCHS):

        # ==================================================
        # Training
        # ==================================================

        model.train()

        running_train_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):

            noisy = batch["noisy"].to(device)
            gt = batch["gt"].to(device)

            # Forward pass
            prediction = model(noisy)

            # Calculate loss
            loss = criterion(
                prediction,
                gt
            )

            # Backpropagation
            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            running_train_loss += loss.item()

            # Print progress
            if (batch_idx + 1) % 100 == 0:

                print(
                    f"Epoch [{epoch + 1}/{EPOCHS}] "
                    f"Train Batch "
                    f"[{batch_idx + 1}/{len(train_loader)}] "
                    f"Loss: {loss.item():.6f}"
                )

        # Average training loss
        train_loss = (
            running_train_loss /
            len(train_loader)
        )

        # ==================================================
        # Validation
        # ==================================================

        model.eval()

        running_val_loss = 0.0

        with torch.no_grad():

            for batch in val_loader:

                noisy = batch["noisy"].to(device)
                gt = batch["gt"].to(device)

                prediction = model(noisy)

                loss = criterion(
                    prediction,
                    gt
                )

                running_val_loss += loss.item()

        # Average validation loss
        val_loss = (
            running_val_loss /
            len(val_loader)
        )

        # --------------------------------------------------
        # Print epoch results
        # --------------------------------------------------

        print()

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"Train Loss: {train_loss:.6f} "
            f"Val Loss: {val_loss:.6f}"
        )

        # --------------------------------------------------
        # Save best checkpoint
        # --------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "seed": SEED,
                    "val_ratio": VAL_RATIO,
                },
                DAY3_CHECKPOINT
            )

            print(
                f"Saved checkpoint: "
                f"{DAY3_CHECKPOINT}"
            )

    # --------------------------------------------------
    # Training completed
    # --------------------------------------------------

    print()
    print("TRAINING COMPLETED")
    print(
        f"Best validation loss: "
        f"{best_val_loss:.6f}"
    )


# --------------------------------------------------
# Entry point
# --------------------------------------------------

if __name__ == "__main__":
    main()