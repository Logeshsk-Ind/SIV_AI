import torch
import torch.nn as nn


class BaselineSR(nn.Module):
    """
    Simple baseline CNN for 2x SEM image super-resolution.

    Input:
        [B, 1, 128, 128]

    Output:
        [B, 1, 256, 256]
    """

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.upsample = nn.Sequential(
            nn.Conv2d(64, 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True),
        )

        self.output = nn.Conv2d(
            1,
            1,
            kernel_size=3,
            padding=1
        )

    def forward(self, x):
        x = self.features(x)
        x = self.upsample(x)
        x = self.output(x)

        return x