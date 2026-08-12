from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# Project root
ROOT = Path(__file__).resolve().parents[1]

GT_DIR = ROOT / "data" / "raw" / "train" / "train" / "GT"
NOISY_DIR = ROOT / "data" / "raw" / "train" / "train" / "NoisyLR"

OUTPUT_DIR = ROOT / "outputs" / "inspection"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Sample to inspect
sample_id = "000000"

gt_path = GT_DIR / f"{sample_id}.npy"
noisy_path = NOISY_DIR / f"{sample_id}.npy"

# Load arrays
gt = np.load(gt_path)
noisy = np.load(noisy_path)

# Upsample NoisyLR to GT resolution for visual comparison
noisy_up = np.repeat(np.repeat(noisy, 2, axis=0), 2, axis=1)

print("GT shape       :", gt.shape)
print("NoisyLR shape  :", noisy.shape)
print("Upsampled shape:", noisy_up.shape)

# Create visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(gt, cmap="gray")
axes[0].set_title("Ground Truth (GT)")
axes[0].axis("off")

axes[1].imshow(noisy, cmap="gray")
axes[1].set_title("NoisyLR")
axes[1].axis("off")

axes[2].imshow(noisy_up, cmap="gray")
axes[2].set_title("NoisyLR (Upsampled)")
axes[2].axis("off")

plt.tight_layout()

output_path = OUTPUT_DIR / "sample_000000.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight")
plt.close()

print("Saved visualization:", output_path)