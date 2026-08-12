from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity


ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = ROOT / "outputs" / "day2_baseline"

RESTORED_PATH = OUTPUT_DIR / "restored.png"
GT_PATH = OUTPUT_DIR / "ground_truth.png"


def load_image(path):
    """
    Load a grayscale PNG and convert it to
    a floating-point array in the range [0, 1].
    """

    image = Image.open(path).convert("L")

    image = np.asarray(image).astype(np.float32) / 255.0

    return image


def main():

    # Load restored image and ground truth
    restored = load_image(RESTORED_PATH)
    gt = load_image(GT_PATH)

    print("Restored shape:", restored.shape)
    print("GT shape      :", gt.shape)

    # Verify dimensions
    assert restored.shape == gt.shape

    # Calculate PSNR
    psnr = peak_signal_noise_ratio(
        gt,
        restored,
        data_range=1.0
    )

    # Calculate SSIM
    ssim = structural_similarity(
        gt,
        restored,
        data_range=1.0
    )

    print()
    print("Day 2 Baseline Results")
    print("----------------------")
    print(f"PSNR: {psnr:.4f} dB")
    print(f"SSIM: {ssim:.4f}")


if __name__ == "__main__":
    main()