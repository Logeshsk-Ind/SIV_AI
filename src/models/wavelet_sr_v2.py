import torch
import torch.nn as nn

from src.wavelet.dwt import dwt2d, idwt2d
from src.models.restormer_lite import RestormerLite
from src.models.nafnet_lite import NAFNetLite


class WaveletSRV2(nn.Module):
    """
    SIV-AI WaveletSR V2

    Improvements over V1:
        1. Wavelet-domain restoration
        2. Learned 2x upsampling using PixelShuffle
        3. High-resolution refinement
        4. Global residual reconstruction

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
        hr_width=32,
    ):
        super().__init__()

        # ==================================================
        # LOW-FREQUENCY BRANCH
        # ==================================================

        self.restormer = RestormerLite(
            in_channels=in_channels,
            out_channels=in_channels,
            width=restormer_width,
            num_blocks=restormer_blocks,
            num_heads=restormer_heads,
        )

        # ==================================================
        # HIGH-FREQUENCY BRANCH
        # ==================================================

        self.nafnet = NAFNetLite(
            in_channels=3,
            out_channels=3,
            width=nafnet_width,
            num_blocks=nafnet_blocks,
        )

        # ==================================================
        # POST-IDWT FEATURE EXTRACTION
        # ==================================================

        self.feature_extract = nn.Sequential(

            nn.Conv2d(
                in_channels,
                hr_width,
                kernel_size=3,
                padding=1,
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                hr_width,
                hr_width,
                kernel_size=3,
                padding=1,
            ),

            nn.ReLU(inplace=True),
        )

        # ==================================================
        # LEARNED 2X UPSAMPLING
        #
        # PixelShuffle:
        #
        # [B, 32, H, W]
        #
        #      ↓
        #
        # [B, 32*4, H, W]
        #
        #      ↓
        #
        # [B, 32, 2H, 2W]
        # ==================================================

        self.upsample = nn.Sequential(

            nn.Conv2d(
                hr_width,
                hr_width * 4,
                kernel_size=3,
                padding=1,
            ),

            nn.PixelShuffle(2),

            nn.ReLU(inplace=True),
        )

        # ==================================================
        # HIGH-RESOLUTION REFINEMENT
        # ==================================================

        self.refinement = nn.Sequential(

            nn.Conv2d(
                hr_width,
                hr_width,
                kernel_size=3,
                padding=1,
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                hr_width,
                hr_width,
                kernel_size=3,
                padding=1,
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                hr_width,
                out_channels,
                kernel_size=3,
                padding=1,
            ),
        )

    # ======================================================
    # FORWARD
    # ======================================================

    def forward(self, x):

        # --------------------------------------------------
        # Preserve original input
        # --------------------------------------------------

        input_image = x

        # --------------------------------------------------
        # Haar DWT
        # --------------------------------------------------

        ll, lh, hl, hh = dwt2d(x)

        # --------------------------------------------------
        # Low-frequency restoration
        # --------------------------------------------------

        ll_restored = self.restormer(ll)

        # --------------------------------------------------
        # High-frequency restoration
        # --------------------------------------------------

        high_frequency = torch.cat(
            [lh, hl, hh],
            dim=1,
        )

        high_restored = self.nafnet(
            high_frequency
        )

        # --------------------------------------------------
        # Split wavelet bands
        # --------------------------------------------------

        lh_restored = high_restored[:, 0:1]

        hl_restored = high_restored[:, 1:2]

        hh_restored = high_restored[:, 2:3]

        # --------------------------------------------------
        # Inverse DWT
        # --------------------------------------------------

        reconstructed = idwt2d(
            ll_restored,
            lh_restored,
            hl_restored,
            hh_restored,
        )

        # --------------------------------------------------
        # Feature extraction
        # --------------------------------------------------

        features = self.feature_extract(
            reconstructed
        )

        # --------------------------------------------------
        # Learned 2x upsampling
        # --------------------------------------------------

        features = self.upsample(
            features
        )

        # --------------------------------------------------
        # High-resolution refinement
        # --------------------------------------------------

        restored = self.refinement(
            features
        )

        # --------------------------------------------------
        # Global residual
        #
        # Original 128×128 image
        #       ↓
        # bilinear 2x
        #       ↓
        # 256×256
        # --------------------------------------------------

        base = torch.nn.functional.interpolate(
            input_image,
            scale_factor=2,
            mode="bilinear",
            align_corners=False,
        )

        restored = restored + base

        return restored


# ==========================================================
# TEST
# ==========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("SIV-AI WaveletSR V2 TEST")
    print("=" * 60)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    if device.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    # ------------------------------------------------------
    # Model
    # ------------------------------------------------------

    model = WaveletSRV2().to(device)

    # ------------------------------------------------------
    # Test input
    # ------------------------------------------------------

    x = torch.randn(
        2,
        1,
        128,
        128,
        device=device,
    )

    # ------------------------------------------------------
    # Forward
    # ------------------------------------------------------

    with torch.no_grad():

        y = model(x)

    # ------------------------------------------------------
    # Parameter count
    # ------------------------------------------------------

    parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        "Input shape:",
        x.shape
    )

    print(
        "Output shape:",
        y.shape
    )

    print(
        "Parameters:",
        parameters
    )

    # ------------------------------------------------------
    # Assertions
    # ------------------------------------------------------

    assert y.shape == (
        2,
        1,
        256,
        256,
    )

    assert torch.isfinite(y).all()

    print()
    print("WaveletSR V2 test PASSED")
    print("=" * 60)