import torch

from models.residual_sr import ResidualSR


def test_residual_model_forward():

    model = ResidualSR()

    x = torch.randn(
        2,
        1,
        128,
        128
    )

    y = model(x)

    assert y.shape == (
        2,
        1,
        256,
        256
    )

    assert torch.isfinite(y).all()