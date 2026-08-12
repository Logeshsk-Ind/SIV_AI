from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.kla_dataset import KLADataset


DATA_DIR = PROJECT_ROOT / "data" / "raw" / "train" / "train"


def test_kla_dataset():
    dataset = KLADataset(DATA_DIR)

    assert len(dataset) == 3200

    sample = dataset[0]

    assert sample["noisy"].shape == torch.Size([1, 128, 128])
    assert sample["gt"].shape == torch.Size([1, 256, 256])

    assert sample["noisy"].dtype == torch.float32
    assert sample["gt"].dtype == torch.float32


def test_kla_dataloader():
    dataset = KLADataset(DATA_DIR)

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
    )

    batch = next(iter(loader))

    assert batch["noisy"].shape == torch.Size([4, 1, 128, 128])
    assert batch["gt"].shape == torch.Size([4, 1, 256, 256])