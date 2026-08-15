from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.kla_dataset import KLADataset
from src.models.siv_ai_phase4 import SIVAI


DATA_DIR = ROOT / "data" / "raw" / "train" / "train"

CHECKPOINT = (
    ROOT
    / "phase4_source"
    / "siv_ai_phase4_ssim_best.pth"
)

SEED = 42
VAL_RATIO = 0.2
BATCH_SIZE = 1


def main():

    print("=" * 70)
    print("SIV-AI PHASE 4 SSIM FINE-TUNED EVALUATION")
    print("=" * 70)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    # --------------------------------------------------
    # Dataset
    # --------------------------------------------------

    dataset = KLADataset(DATA_DIR)

    print()
    print("Dataset:", DATA_DIR)
    print("Total samples:", len(dataset))

    generator = torch.Generator().manual_seed(SEED)

    indices = torch.randperm(
        len(dataset),
        generator=generator
    ).tolist()

    val_size = int(len(dataset) * VAL_RATIO)

    val_indices = indices[-val_size:]

    print("Validation samples:", len(val_indices))

    val_dataset = Subset(
        dataset,
        val_indices
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    print()
    print("Creating Phase-4 model...")

    model = SIVAI().to(device)

    print(
        "Model parameters:",
        sum(p.numel() for p in model.parameters())
    )

    # --------------------------------------------------
    # Checkpoint
    # --------------------------------------------------

    print()
    print("Loading checkpoint:")
    print(CHECKPOINT)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device
    )

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)

    print("Checkpoint loaded successfully.")

    model.eval()

    # --------------------------------------------------
    # Metrics
    # --------------------------------------------------

    psnr_values = []
    ssim_values = []
    mse_values = []

    # --------------------------------------------------
    # Evaluation
    # --------------------------------------------------

    with torch.no_grad():

        for batch_idx, batch in enumerate(val_loader):

            noisy = batch["noisy"].to(device)
            gt = batch["gt"].to(device)

            prediction = model(noisy)

            prediction = prediction.clamp(0, 1)

            prediction_np = (
                prediction[0, 0]
                .cpu()
                .numpy()
            )

            gt_np = (
                gt[0, 0]
                .cpu()
                .numpy()
            )

            mse = np.mean(
                (prediction_np - gt_np) ** 2
            )

            psnr = peak_signal_noise_ratio(
                gt_np,
                prediction_np,
                data_range=1.0
            )

            ssim = structural_similarity(
                gt_np,
                prediction_np,
                data_range=1.0
            )

            mse_values.append(mse)
            psnr_values.append(psnr)
            ssim_values.append(ssim)

            if (batch_idx + 1) % 100 == 0:

                print(
                    f"Processed "
                    f"{batch_idx + 1}/{len(val_loader)}"
                )

    # --------------------------------------------------
    # Results
    # --------------------------------------------------

    mean_psnr = float(np.mean(psnr_values))
    mean_ssim = float(np.mean(ssim_values))
    mean_mse = float(np.mean(mse_values))

    print()
    print("=" * 70)
    print("SIV-AI PHASE 4 SSIM FINE-TUNED RESULTS")
    print("=" * 70)

    print(
        f"Validation samples : {len(val_indices)}"
    )

    print(
        f"Mean PSNR           : {mean_psnr:.6f} dB"
    )

    print(
        f"Mean SSIM           : {mean_ssim:.6f}"
    )

    print(
        f"Mean MSE            : {mean_mse:.8f}"
    )

    print()
    print("=" * 70)
    print("BASELINE COMPARISON")
    print("=" * 70)

    print("Original Phase-4:")
    print("PSNR : 27.910144 dB")
    print("SSIM : 0.753018")

    print()
    print("SSIM Fine-Tuned:")

    print(
        f"PSNR : {mean_psnr:.6f} dB"
    )

    print(
        f"SSIM : {mean_ssim:.6f}"
    )

    print()
    print("Changes:")

    print(
        f"PSNR difference : "
        f"{mean_psnr - 27.910144:+.6f} dB"
    )

    print(
        f"SSIM difference : "
        f"{mean_ssim - 0.753018:+.6f}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()