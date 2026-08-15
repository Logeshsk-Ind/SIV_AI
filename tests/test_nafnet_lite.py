import torch

from src.models.nafnet_lite import NAFNetLite


def test_nafnet_lite_forward():

    model = NAFNetLite(
        in_channels=3,
        out_channels=3,
        width=32,
        num_blocks=5,
    )

    x = torch.randn(
        2,
        3,
        64,
        64,
    )

    y = model(x)

    assert y.shape == x.shape

    assert torch.isfinite(y).all()