from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.kla_dataset import KLADataset


DATA_DIR = ROOT / "data" / "raw" / "train" / "train"

SEED = 42
VAL_RATIO = 0.2


def main():

    print("=" * 70)
    print("SIV-AI PHASE 4 VALIDATION SPLIT CHECK")
    print("=" * 70)

    dataset = KLADataset(DATA_DIR)

    print("Dataset size:", len(dataset))

    generator = torch.Generator().manual_seed(SEED)

    indices = torch.randperm(
        len(dataset),
        generator=generator
    ).tolist()

    val_size = int(
        len(dataset) * VAL_RATIO
    )

    train_indices = indices[:-val_size]
    val_indices = indices[-val_size:]

    print("Seed:", SEED)
    print("Validation ratio:", VAL_RATIO)
    print("Training samples:", len(train_indices))
    print("Validation samples:", len(val_indices))

    print()
    print("First 20 validation indices:")
    print(val_indices[:20])

    print()
    print("First 20 validation names:")

    for idx in val_indices[:20]:
        print(
            idx,
            dataset.gt_files[idx].stem
        )

    print()
    print("Last 20 validation names:")

    for idx in val_indices[-20:]:
        print(
            idx,
            dataset.gt_files[idx].stem
        )

    print()
    print("=" * 70)
    print("SPLIT CHECK COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()