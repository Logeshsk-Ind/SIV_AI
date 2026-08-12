from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.baseline_sr import BaselineSR


def test_baseline_model_forward():
    model = BaselineSR()

    x = torch.randn(4, 1, 128, 128)

    y = model(x)

    assert y.shape == (4, 1, 256, 256)