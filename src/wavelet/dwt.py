import torch


class HaarDWT:
    """
    Differentiable 2D Haar Wavelet Transform.

    Input:
        x -> [B, C, H, W]

    Output:
        LL -> [B, C, H/2, W/2]
        LH -> [B, C, H/2, W/2]
        HL -> [B, C, H/2, W/2]
        HH -> [B, C, H/2, W/2]

    The implementation uses only PyTorch operations,
    so it works on CPU and CUDA and remains differentiable.
    """

    @staticmethod
    def forward(x):

        if x.ndim != 4:
            raise ValueError(
                f"Expected [B,C,H,W], got {tuple(x.shape)}"
            )

        b, c, h, w = x.shape

        if h % 2 != 0 or w % 2 != 0:
            raise ValueError(
                "Haar DWT requires even height and width."
            )

        x00 = x[:, :, 0::2, 0::2]
        x01 = x[:, :, 0::2, 1::2]
        x10 = x[:, :, 1::2, 0::2]
        x11 = x[:, :, 1::2, 1::2]

        scale = 0.5

        ll = (
            x00 + x01 + x10 + x11
        ) * scale

        lh = (
            x00 - x01 + x10 - x11
        ) * scale

        hl = (
            x00 + x01 - x10 - x11
        ) * scale

        hh = (
            x00 - x01 - x10 + x11
        ) * scale

        return ll, lh, hl, hh


class HaarIDWT:
    """
    Inverse 2D Haar Wavelet Transform.

    Input:
        LL, LH, HL, HH

    Output:
        reconstructed image [B,C,H,W]
    """

    @staticmethod
    def forward(ll, lh, hl, hh):

        if not (
            ll.shape
            == lh.shape
            == hl.shape
            == hh.shape
        ):
            raise ValueError(
                "All wavelet coefficient tensors "
                "must have identical shapes."
            )

        x00 = (
            ll + lh + hl + hh
        ) * 0.5

        x01 = (
            ll - lh + hl - hh
        ) * 0.5

        x10 = (
            ll + lh - hl - hh
        ) * 0.5

        x11 = (
            ll - lh - hl + hh
        ) * 0.5

        b, c, h, w = ll.shape

        output = torch.zeros(
            b,
            c,
            h * 2,
            w * 2,
            device=ll.device,
            dtype=ll.dtype,
        )

        output[:, :, 0::2, 0::2] = x00
        output[:, :, 0::2, 1::2] = x01
        output[:, :, 1::2, 0::2] = x10
        output[:, :, 1::2, 1::2] = x11

        return output


def dwt2d(x):
    return HaarDWT.forward(x)


def idwt2d(ll, lh, hl, hh):
    return HaarIDWT.forward(
        ll,
        lh,
        hl,
        hh,
    )