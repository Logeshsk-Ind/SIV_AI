import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


ID = "000276"

NOISY = Path(r"data\raw\train\train\NoisyLR") / f"{ID}.npy"
GT = Path(r"data\raw\train\train\GT") / f"{ID}.npy"
SIVA = Path(r"outputs\phase4") / f"{ID}_restored.png"

OUT = Path(r"results") / f"{ID}_comparison.png"


def load_npy(path):
    x = np.load(path).astype(np.float32)
    x = np.squeeze(x)
    x = np.clip(x, 0, 1)
    return Image.fromarray((x * 255).astype(np.uint8))


def main():

    noisy = load_npy(NOISY)
    gt = load_npy(GT)
    siva = Image.open(SIVA).convert("L")

    # Bicubic upscale
    bicubic = noisy.resize(gt.size, Image.Resampling.BICUBIC)

    # Make all images same display size
    size = (256, 256)

    images = [
        ("GT\n256×256", gt.resize(size)),
        ("NoisyLR\n128×128", noisy),
        ("Bicubic\n256×256", bicubic),
        ("SIV-AI\n256×256", siva.resize(size))
    ]

    W = 256
    H = 300

    canvas = Image.new("RGB", (W * 4, H), "white")
    draw = ImageDraw.Draw(canvas)

    for i, (title, img) in enumerate(images):

        img = img.convert("RGB")

        x = i * W

        canvas.paste(img, (x, 44))

        draw.text(
            (x + 10, 8),
            title,
            fill="black"
        )

    # Metrics
    draw.text(
        (10, 275),
        "SIV-AI: PSNR 36.93 dB | SSIM 0.965 | LPIPS 0.059",
        fill="black"
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)

    canvas.save(OUT)

    print("=" * 70)
    print("COMPARISON IMAGE CREATED")
    print("=" * 70)
    print(f"Saved: {OUT}")
    print("=" * 70)


if __name__ == "__main__":
    main()