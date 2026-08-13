from pathlib import Path
import sys

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.kla_dataset import KLADataset
from models.baseline_sr import BaselineSR


DATA_DIR = ROOT / "data" / "raw" / "train" / "train"
CHECKPOINT = ROOT / "checkpoints" / "baseline_day3_best.pth"

BATCH_SIZE = 4
VAL_RATIO = 0.20
SEED = 42


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    dataset = KLADataset(DATA_DIR)

    print("Total samples:", len(dataset))

    # Deterministic 80/20 split
    generator = torch.Generator().manual_seed(SEED)

    indices = torch.randperm(
        len(dataset),
        generator=generator
    ).tolist()

    val_size = int(len(dataset) * VAL_RATIO)

    val_indices = indices[-val_size:]

    val_dataset = Subset(
        dataset,
        val_indices
    )

    print("Validation samples:", len(val_dataset))

    loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    # Load model
    model = BaselineSR().to(device)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print("Checkpoint loaded successfully")

    psnr_values = []
    ssim_values = []

    with torch.no_grad():

        for batch_idx, batch in enumerate(loader):

            noisy = batch["noisy"].to(device)
            gt = batch["gt"].numpy()

            prediction = model(noisy)

            prediction = (
                prediction
                .clamp(0, 1)
                .cpu()
                .numpy()
            )

            for pred, target in zip(
                prediction,
                gt
            ):

                pred_image = pred[0]
                target_image = target[0]

                psnr = peak_signal_noise_ratio(
                    target_image,
                    pred_image,
                    data_range=1.0
                )

                ssim = structural_similarity(
                    target_image,
                    pred_image,
                    data_range=1.0
                )

                psnr_values.append(psnr)
                ssim_values.append(ssim)

            if (batch_idx + 1) % 50 == 0:
                print(
                    f"Processed "
                    f"{(batch_idx + 1) * BATCH_SIZE}/"
                    f"{len(val_dataset)}"
                )

    mean_psnr = float(np.mean(psnr_values))
    mean_ssim = float(np.mean(ssim_values))

    print()
    print("Day 3 Baseline Dataset Evaluation")
    print("----------------------------------")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Mean PSNR: {mean_psnr:.4f} dB")
    print(f"Mean SSIM: {mean_ssim:.4f}")


if __name__ == "__main__":
    main()