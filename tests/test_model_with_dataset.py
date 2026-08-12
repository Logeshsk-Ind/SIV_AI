from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.kla_dataset import KLADataset
from models.baseline_sr import BaselineSR


DATA_DIR = PROJECT_ROOT / "data" / "raw" / "train" / "train"


def test_model_with_dataset():
    dataset = KLADataset(DATA_DIR)

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
    )

    batch = next(iter(loader))

    model = BaselineSR()

    prediction = model(batch["noisy"])

    assert prediction.shape == batch["gt"].shape

    loss = torch.nn.functional.l1_loss(
        prediction,
        batch["gt"],
    )

    assert torch.isfinite(loss)
    assert loss.item() >= 0