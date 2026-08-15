from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity


ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT))

from src.data.kla_dataset import KLADataset
from src.models.wavelet_sr import WaveletSR


DATA_DIR = ROOT / "data" / "raw" / "train" / "train"

CHECKPOINT = ROOT / "checkpoints" / "wavelet_sr_best.pth"

SEED = 42
VAL_RATIO = 0.20


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    dataset = KLADataset(DATA_DIR)

    generator = torch.Generator().manual_seed(SEED)

    indices = torch.randperm(
        len(dataset),
        generator=generator
    ).tolist()

    val_size = int(len(dataset) * VAL_RATIO)

    val_indices = indices[-val_size:]

    model = WaveletSR().to(device)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    bicubic_psnr = []
    bicubic_ssim = []

    wavelet_psnr = []
    wavelet_ssim = []

    with torch.no_grad():

        for count, index in enumerate(val_indices, 1):

            sample = dataset[index]

            noisy = sample["noisy"].unsqueeze(0).to(device)
            gt = sample["gt"].unsqueeze(0).to(device)

            # --------------------------------------------------
            # Bicubic baseline
            # --------------------------------------------------

            bicubic = F.interpolate(
                noisy,
                size=gt.shape[-2:],
                mode="bicubic",
                align_corners=False
            )

            bicubic = bicubic.clamp(0, 1)

            # --------------------------------------------------
            # WaveletSR
            # --------------------------------------------------

            prediction = model(noisy)

            prediction = prediction.clamp(0, 1)

            # --------------------------------------------------
            # NumPy
            # --------------------------------------------------

            gt_np = gt[0, 0].cpu().numpy()

            bicubic_np = bicubic[0, 0].cpu().numpy()

            prediction_np = prediction[0, 0].cpu().numpy()

            # --------------------------------------------------
            # Metrics
            # --------------------------------------------------

            bicubic_psnr.append(
                peak_signal_noise_ratio(
                    gt_np,
                    bicubic_np,
                    data_range=1.0
                )
            )

            bicubic_ssim.append(
                structural_similarity(
                    gt_np,
                    bicubic_np,
                    data_range=1.0
                )
            )

            wavelet_psnr.append(
                peak_signal_noise_ratio(
                    gt_np,
                    prediction_np,
                    data_range=1.0
                )
            )

            wavelet_ssim.append(
                structural_similarity(
                    gt_np,
                    prediction_np,
                    data_range=1.0
                )
            )

            if count % 200 == 0:
                print(
                    f"Processed {count}/{len(val_indices)}"
                )

    print()
    print("=" * 60)
    print("BASELINE VS WAVELETSR")
    print("=" * 60)

    print()
    print("Bicubic:")
    print(
        f"PSNR: {np.mean(bicubic_psnr):.4f} dB"
    )
    print(
        f"SSIM: {np.mean(bicubic_ssim):.4f}"
    )

    print()
    print("WaveletSR:")
    print(
        f"PSNR: {np.mean(wavelet_psnr):.4f} dB"
    )
    print(
        f"SSIM: {np.mean(wavelet_ssim):.4f}"
    )

    print()
    print("Improvement:")
    print(
        f"PSNR: "
        f"{np.mean(wavelet_psnr) - np.mean(bicubic_psnr):+.4f} dB"
    )
    print(
        f"SSIM: "
        f"{np.mean(wavelet_ssim) - np.mean(bicubic_ssim):+.4f}"
    )


if __name__ == "__main__":
    main()