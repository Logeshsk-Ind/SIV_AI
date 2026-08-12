from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.kla_dataset import KLADataset
from models.baseline_sr import BaselineSR


DATA_DIR = PROJECT_ROOT / "data" / "raw" / "train" / "train"


def test_training_step():
    dataset = KLADataset(DATA_DIR)

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
    )

    batch = next(iter(loader))

    model = BaselineSR()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-4,
    )

    criterion = torch.nn.L1Loss()

    prediction = model(batch["noisy"])
    loss_before = criterion(prediction, batch["gt"])

    optimizer.zero_grad()
    loss_before.backward()
    optimizer.step()

    prediction_after = model(batch["noisy"])
    loss_after = criterion(prediction_after, batch["gt"])

    assert torch.isfinite(loss_before)
    assert torch.isfinite(loss_after)