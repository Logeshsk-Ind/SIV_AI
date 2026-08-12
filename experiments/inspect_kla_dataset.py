from pathlib import Path
import numpy as np

TRAIN_DIR = Path("data/raw/train")
TEST_DIR = Path("data/raw/test_noisy")


def inspect_folder(folder):
    files = list(folder.rglob("*.npy"))

    print(f"\nFolder: {folder}")
    print(f"Number of NPY files: {len(files)}")

    if not files:
        print("No .npy files found.")
        return

    for path in files[:5]:
        data = np.load(path)

        print("\nFile:", path)
        print("Shape:", data.shape)
        print("Dtype:", data.dtype)
        print("Min:", data.min())
        print("Max:", data.max())
        print("Mean:", data.mean())


if __name__ == "__main__":
    inspect_folder(TRAIN_DIR)
    inspect_folder(TEST_DIR)