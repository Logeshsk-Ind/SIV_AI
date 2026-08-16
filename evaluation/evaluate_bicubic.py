import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import lpips
import torch


def load_npy(path):
    x = np.load(path).astype(np.float32)

    if x.ndim > 2:
        x = np.squeeze(x)

    x = np.clip(x, 0.0, 1.0)

    return x


def to_tensor(x):
    x = torch.from_numpy(x).float()
    x = x.unsqueeze(0).unsqueeze(0)
    x = x.repeat(1, 3, 1, 1)
    return x * 2.0 - 1.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--noisy", required=True)
    parser.add_argument("--gt", required=True)
    parser.add_argument("--limit", type=int, default=400)
    args = parser.parse_args()

    noisy_dir = Path(args.noisy)
    gt_dir = Path(args.gt)

    ids = sorted(
        p.stem for p in noisy_dir.glob("*.npy")
        if (gt_dir / f"{p.stem}.npy").exists()
    )[:args.limit]

    loss_fn = lpips.LPIPS(net="alex")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loss_fn = loss_fn.to(device)

    psnr_values = []
    ssim_values = []
    lpips_values = []

    print("=" * 70)
    print("BICUBIC BASELINE EVALUATION")
    print("=" * 70)

    for i, image_id in enumerate(ids, 1):

        noisy = load_npy(noisy_dir / f"{image_id}.npy")
        gt = load_npy(gt_dir / f"{image_id}.npy")

        h, w = gt.shape

        image = Image.fromarray(
            (noisy * 255).clip(0, 255).astype(np.uint8)
        )

        image = image.resize(
            (w, h),
            Image.Resampling.BICUBIC
        )

        pred = np.asarray(
            image,
            dtype=np.float32
        ) / 255.0

        psnr = peak_signal_noise_ratio(
            gt,
            pred,
            data_range=1.0
        )

        ssim = structural_similarity(
            gt,
            pred,
            data_range=1.0
        )

        gt_t = to_tensor(gt).to(device)
        pred_t = to_tensor(pred).to(device)

        with torch.no_grad():
            lp = loss_fn(gt_t, pred_t).item()

        psnr_values.append(psnr)
        ssim_values.append(ssim)
        lpips_values.append(lp)

        if i % 50 == 0:
            print(f"Processed {i}/{len(ids)}")

    print()
    print("=" * 70)
    print("BICUBIC RESULTS")
    print("=" * 70)
    print(f"Images evaluated : {len(ids)}")
    print(f"Mean PSNR        : {np.mean(psnr_values):.4f} dB")
    print(f"Mean SSIM        : {np.mean(ssim_values):.6f}")
    print(f"Mean LPIPS       : {np.mean(lpips_values):.6f}")
    print("=" * 70)


if __name__ == "__main__":
    main()