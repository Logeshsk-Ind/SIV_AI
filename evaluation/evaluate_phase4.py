from pathlib import Path
import argparse
import csv
import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import lpips


def load_gt(path):
    x = np.load(path).astype(np.float32)

    if x.ndim == 3:
        x = np.squeeze(x)

    if x.ndim != 2:
        raise ValueError(f"GT must be 2D, got {x.shape}")

    return x


def load_output(path):
    x = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
    x /= 255.0
    return x


def evaluate(gt, pred, lpips_model, device):
    if gt.shape != pred.shape:
        pred_img = Image.fromarray(
            np.clip(pred * 255, 0, 255).astype(np.uint8)
        ).resize(
            (gt.shape[1], gt.shape[0]),
            Image.Resampling.BICUBIC
        )
        pred = np.asarray(pred_img, dtype=np.float32) / 255.0

    gt = np.clip(gt, 0, 1)
    pred = np.clip(pred, 0, 1)

    psnr = peak_signal_noise_ratio(gt, pred, data_range=1.0)

    ssim = structural_similarity(
        gt,
        pred,
        data_range=1.0
    )

    gt_t = torch.from_numpy(gt)[None, None].repeat(1, 3, 1, 1)
    pred_t = torch.from_numpy(pred)[None, None].repeat(1, 3, 1, 1)

    gt_t = gt_t.to(device) * 2 - 1
    pred_t = pred_t.to(device) * 2 - 1

    with torch.no_grad():
        lp = lpips_model(gt_t, pred_t).item()

    return psnr, ssim, lp


def main():
    parser = argparse.ArgumentParser(
        description="SIV-AI Phase-4 PSNR / SSIM / LPIPS evaluation"
    )

    parser.add_argument(
        "--gt",
        required=True,
        help="Directory containing GT .npy files"
    )

    parser.add_argument(
        "--pred",
        required=True,
        help="Directory containing restored images"
    )

    parser.add_argument(
        "--output",
        default="results/phase4_metrics.csv",
        help="CSV output path"
    )

    args = parser.parse_args()

    gt_dir = Path(args.gt)
    pred_dir = Path(args.pred)
    csv_path = Path(args.output)

    if not gt_dir.exists():
        raise FileNotFoundError(f"GT directory not found: {gt_dir}")

    if not pred_dir.exists():
        raise FileNotFoundError(f"Prediction directory not found: {pred_dir}")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 70)
    print("SIV-AI PHASE-4 EVALUATION")
    print("=" * 70)
    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    print()
    print("GT directory   :", gt_dir)
    print("Prediction dir :", pred_dir)

    lpips_model = lpips.LPIPS(net="alex").to(device)
    lpips_model.eval()

    gt_files = {
        p.stem: p
        for p in gt_dir.glob("*.npy")
        if not p.name.startswith("._")
    }

    pred_files = {}

    for p in pred_dir.iterdir():
        if p.suffix.lower() in {
            ".png", ".jpg", ".jpeg", ".bmp",
            ".tif", ".tiff", ".webp"
        }:
            pred_files[p.stem.split("_restored")[0]] = p

    common = sorted(set(gt_files) & set(pred_files))

    print()
    print("GT images       :", len(gt_files))
    print("Predictions     :", len(pred_files))
    print("Matched images  :", len(common))
    print()

    if not common:
        raise RuntimeError(
            "No matching GT/prediction files found."
        )

    results = []

    for i, name in enumerate(common, 1):

        try:
            gt = load_gt(gt_files[name])
            pred = load_output(pred_files[name])

            psnr, ssim, lp = evaluate(
                gt,
                pred,
                lpips_model,
                device
            )

            results.append(
                [name, psnr, ssim, lp]
            )

            print(
                f"[{i:4d}/{len(common)}] "
                f"{name}  "
                f"PSNR={psnr:7.3f}  "
                f"SSIM={ssim:.5f}  "
                f"LPIPS={lp:.5f}"
            )

        except Exception as e:
            print(f"[SKIP] {name}: {e}")

    if not results:
        raise RuntimeError("No images were successfully evaluated.")

    arr = np.asarray(
        [r[1:] for r in results],
        dtype=np.float64
    )

    mean_psnr = arr[:, 0].mean()
    mean_ssim = arr[:, 1].mean()
    mean_lpips = arr[:, 2].mean()

    csv_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "image",
            "PSNR_dB",
            "SSIM",
            "LPIPS"
        ])

        writer.writerows(results)

        writer.writerow([])
        writer.writerow([
            "MEAN",
            mean_psnr,
            mean_ssim,
            mean_lpips
        ])

    print()
    print("=" * 70)
    print("PHASE-4 RESULTS")
    print("=" * 70)
    print(f"Images evaluated : {len(results)}")
    print(f"Mean PSNR        : {mean_psnr:.4f} dB")
    print(f"Mean SSIM        : {mean_ssim:.6f}")
    print(f"Mean LPIPS       : {mean_lpips:.6f}")
    print()
    print("CSV saved to:")
    print(csv_path)
    print("=" * 70)


if __name__ == "__main__":
    main()