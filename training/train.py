from pathlib import Path
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# Add project root to Python path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.kla_dataset import KLADataset
from models.baseline_sr import BaselineSR


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DATA_DIR = ROOT / "data" / "raw" / "train" / "train"
CHECKPOINT_DIR = ROOT / "checkpoints"

BATCH_SIZE = 4
EPOCHS = 3
LEARNING_RATE = 1e-4


# --------------------------------------------------
# Training
# --------------------------------------------------

def main():

    # Select device
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    # Create checkpoint directory
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # Load dataset
    dataset = KLADataset(DATA_DIR)

    print("Dataset size:", len(dataset))

    # Create DataLoader
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    # Create model
    model = BaselineSR().to(device)

    # Loss function
    criterion = nn.L1Loss()

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    best_loss = float("inf")

    # --------------------------------------------------
    # Epoch loop
    # --------------------------------------------------

    for epoch in range(EPOCHS):

        model.train()

        running_loss = 0.0

        for batch_idx, batch in enumerate(loader):

            noisy = batch["noisy"].to(device)
            gt = batch["gt"].to(device)

            # Forward pass
            prediction = model(noisy)

            # Calculate loss
            loss = criterion(prediction, gt)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Print progress every 100 batches
            if (batch_idx + 1) % 100 == 0:
                print(
                    f"Epoch [{epoch + 1}/{EPOCHS}] "
                    f"Batch [{batch_idx + 1}/{len(loader)}] "
                    f"Loss: {loss.item():.6f}"
                )

        # Average epoch loss
        epoch_loss = running_loss / len(loader)

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"Average Loss: {epoch_loss:.6f}"
        )

        # Save best model
        if epoch_loss < best_loss:

            best_loss = epoch_loss

            checkpoint_path = (
                CHECKPOINT_DIR / "baseline_day2_best.pth"
            )

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": epoch_loss,
                },
                checkpoint_path,
            )

            print(
                f"Saved checkpoint: {checkpoint_path}"
            )

    print()
    print("TRAINING COMPLETED")
    print(f"Best loss: {best_loss:.6f}")


if __name__ == "__main__":
    main()
    
