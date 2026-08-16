import torch
import torch.nn as nn
import torch.nn.functional as F

from src.wavelet.dwt import dwt2d, idwt2d
from src.models.restormer_lite import RestormerLite
from src.models.nafnet_lite import NAFNetLite


class WaveletSR(nn.Module):
    """
    SIV-AI Wavelet Super-Resolution Network.

    Pipeline:

        Low-resolution SEM image
                |
                v
            2D DWT
                |
        +-------+-------+
        |               |
        v               v
       LL        LH / HL / HH
        |               |
        v               v
    Restormer        NAFNet
        |               |
        +-------+-------+
                |
                v
        Restored wavelet
         coefficients
                |
                v
              IDWT
                |
                v
          2x upsampling
                |
                v
        Restored SEM image

    Input:
        [B, 1, 128, 128]

    Output:
        [B, 1, 256, 256]
    """

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        restormer_width=32,
        restormer_blocks=4,
        restormer_heads=4,
        nafnet_width=32,
        nafnet_blocks=5,
    ):
        super().__init__()

        # --------------------------------------------------
        # Restormer branch
        # Processes the low-frequency LL component.
        # --------------------------------------------------

        self.restormer = RestormerLite(
            in_channels=in_channels,
            out_channels=in_channels,
            width=restormer_width,
            num_blocks=restormer_blocks,
            num_heads=restormer_heads,
        )

        # --------------------------------------------------
        # NAFNet branch
        # Processes the three high-frequency components:
        #
        # LH
        # HL
        # HH
        #
        # Therefore input/output channels = 3.
        # --------------------------------------------------

        self.nafnet = NAFNetLite(
            in_channels=3,
            out_channels=3,
            width=nafnet_width,
            num_blocks=nafnet_blocks,
        )

        # --------------------------------------------------
        # Final refinement
        #
        # IDWT produces a reconstructed image at the
        # original low-resolution input size.
        #
        # We then perform 2x interpolation followed by
        # convolutional refinement.
        # --------------------------------------------------

        self.refinement = nn.Sequential(
            nn.Conv2d(
                in_channels,
                32,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                32,
                32,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                32,
                out_channels,
                kernel_size=3,
                padding=1,
            ),
        )

    def forward(self, x):
        """
        Forward pass.

        Input:
            x = [B, 1, H, W]

        Output:
            [B, 1, 2H, 2W]
        """

        # --------------------------------------------------
        # Save original low-resolution input
        # --------------------------------------------------

        input_image = x

        # --------------------------------------------------
        # DWT decomposition
        #
        # Each wavelet component becomes H/2 x W/2.
        # --------------------------------------------------

        ll, lh, hl, hh = dwt2d(x)

        # --------------------------------------------------
        # Low-frequency restoration
        # --------------------------------------------------

        ll_restored = self.restormer(ll)

        # --------------------------------------------------
        # High-frequency restoration
        #
        # Combine LH, HL and HH along channel dimension.
        #
        # [B,1,H/2,W/2] x 3
        #        ↓
        # [B,3,H/2,W/2]
        # --------------------------------------------------

        high_frequency = torch.cat(
            [lh, hl, hh],
            dim=1,
        )

        high_restored = self.nafnet(
            high_frequency
        )

        # --------------------------------------------------
        # Split restored high-frequency coefficients
        # --------------------------------------------------

        lh_restored = high_restored[:, 0:1, :, :]
        hl_restored = high_restored[:, 1:2, :, :]
        hh_restored = high_restored[:, 2:3, :, :]

        # --------------------------------------------------
        # Inverse DWT
        #
        # This reconstructs the original input resolution.
        # --------------------------------------------------

        reconstructed = idwt2d(
            ll_restored,
            lh_restored,
            hl_restored,
            hh_restored,
        )

        # --------------------------------------------------
        # 2x super-resolution
        #
        # 128x128 -> 256x256
        # --------------------------------------------------

        upsampled = F.interpolate(
            reconstructed,
            scale_factor=2,
            mode="bilinear",
            align_corners=False,
        )

        # --------------------------------------------------
        # Refinement
        # --------------------------------------------------

        restored = self.refinement(
            upsampled
        )

        # --------------------------------------------------
        # Global residual connection
        #
        # Upsample the original degraded image and add it
        # to the refined restoration.
        # --------------------------------------------------

        base = F.interpolate(
            input_image,
            scale_factor=2,
            mode="bilinear",
            align_corners=False,
        )

        restored = restored + base

        return restored


# ==========================================================
# Standalone test
# ==========================================================

if __name__ == "__main__":

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    model = WaveletSR().to(device)

    x = torch.randn(
        2,
        1,
        128,
        128,
        device=device,
    )

    with torch.no_grad():
        y = model(x)

    parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    print("Device:", device)
    print("Input shape:", x.shape)
    print("Output shape:", y.shape)
    print("Parameters:", parameters)

    assert y.shape == (
        2,
        1,
        256,
        256,
    )

    print("WaveletSR 2x output test PASSED")