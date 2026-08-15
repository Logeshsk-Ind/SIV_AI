import torch

from src.models.wavelet_sr import WaveletSR


def test_wavelet_sr_forward():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = WaveletSR().to(device)

    x = torch.randn(
        2,
        1,
        128,
        128,
        device=device,
    )

    model.eval()

    with torch.no_grad():
        y = model(x)

    # SIV-AI is a 2x super-resolution model.
    assert y.shape == (
        2,
        1,
        256,
        256,
    )


def test_wavelet_sr_backward():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = WaveletSR().to(device)

    x = torch.randn(
        1,
        1,
        128,
        128,
        device=device,
        requires_grad=True,
    )

    y = model(x)

    # The target must have the same resolution as the SR output.
    target = torch.randn(
        1,
        1,
        256,
        256,
        device=device,
    )

    loss = torch.mean(
        (y - target) ** 2
    )

    loss.backward()

    # Verify that gradients reached the input.
    assert x.grad is not None

    # Verify that at least one trainable parameter
    # received a gradient.
    has_parameter_gradient = any(
        parameter.grad is not None
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    assert has_parameter_gradient