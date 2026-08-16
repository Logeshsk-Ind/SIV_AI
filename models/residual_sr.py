import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """
    Residual block for SEM image super-resolution.
    """

    def __init__(self, channels=64):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1
            )
        )

    def forward(self, x):
        return x + self.block(x)


class ResidualSR(nn.Module):
    """
    Improved residual CNN for 2x SEM image super-resolution.

    Input:
        [B, 1, 128, 128]

    Output:
        [B, 1, 256, 256]
    """

    def __init__(
        self,
        channels=64,
        num_blocks=5
    ):
        super().__init__()

        # Initial feature extraction
        self.head = nn.Conv2d(
            1,
            channels,
            kernel_size=3,
            padding=1
        )

        # Residual feature extraction
        self.body = nn.Sequential(
            *[
                ResidualBlock(channels)
                for _ in range(num_blocks)
            ]
        )

        # Feature fusion after residual blocks
        self.body_conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1
        )

        # 2x upsampling
        self.upsample = nn.Sequential(
            nn.Conv2d(
                channels,
                channels * 4,
                kernel_size=3,
                padding=1
            ),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True)
        )

        # Final reconstruction
        self.output = nn.Conv2d(
            channels,
            1,
            kernel_size=3,
            padding=1
        )

    def forward(self, x):

        # Initial features
        features = self.head(x)

        # Residual features
        residual = self.body(features)

        # Global residual connection
        residual = self.body_conv(residual)

        features = features + residual

        # Upsampling
        features = self.upsample(features)

        # Reconstruction
        output = self.output(features)

        return output