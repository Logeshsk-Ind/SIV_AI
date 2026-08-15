import torch
import torch.nn as nn


class SimpleGate(nn.Module):
    """
    Simple gating mechanism used in NAFNet-style blocks.

    Splits the feature channels into two parts
    and performs element-wise multiplication.
    """

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlockLite(nn.Module):
    """
    Lightweight NAFNet-inspired restoration block.

    Designed for efficient image restoration while
    keeping the parameter count manageable.
    """

    def __init__(
        self,
        channels=32,
        expansion=2,
        dropout=0.0,
    ):
        super().__init__()

        hidden = channels * expansion

        self.norm1 = nn.GroupNorm(
            1,
            channels
        )

        self.conv1 = nn.Conv2d(
            channels,
            hidden * 2,
            kernel_size=1,
            padding=0,
        )

        self.dwconv = nn.Conv2d(
            hidden * 2,
            hidden * 2,
            kernel_size=3,
            padding=1,
            groups=hidden * 2,
        )

        self.sg = SimpleGate()

        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(
                hidden,
                hidden,
                kernel_size=1,
            ),
        )

        self.conv2 = nn.Conv2d(
            hidden,
            channels,
            kernel_size=1,
            padding=0,
        )

        self.dropout1 = nn.Dropout2d(
            dropout
        )

        self.norm2 = nn.GroupNorm(
            1,
            channels
        )

        self.ffn1 = nn.Conv2d(
            channels,
            hidden * 2,
            kernel_size=1,
        )

        self.ffn_gate = SimpleGate()

        self.ffn2 = nn.Conv2d(
            hidden,
            channels,
            kernel_size=1,
        )

        self.dropout2 = nn.Dropout2d(
            dropout
        )

        self.beta = nn.Parameter(
            torch.zeros(
                1,
                channels,
                1,
                1
            )
        )

        self.gamma = nn.Parameter(
            torch.zeros(
                1,
                channels,
                1,
                1
            )
        )

    def forward(self, x):

        residual = x

        y = self.norm1(x)

        y = self.conv1(y)

        y = self.dwconv(y)

        y = self.sg(y)

        y = y * self.sca(y)

        y = self.conv2(y)

        y = self.dropout1(y)

        x = residual + y * self.beta

        residual = x

        y = self.norm2(x)

        y = self.ffn1(y)

        y = self.ffn_gate(y)

        y = self.ffn2(y)

        y = self.dropout2(y)

        x = residual + y * self.gamma

        return x


class NAFNetLite(nn.Module):
    """
    Lightweight NAFNet-inspired restoration network.

    Input:
        [B, in_channels, H, W]

    Output:
        [B, out_channels, H, W]

    Default configuration is designed for processing
    concatenated high-frequency Haar wavelet bands.
    """

    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        width=32,
        num_blocks=5,
    ):
        super().__init__()

        self.intro = nn.Conv2d(
            in_channels,
            width,
            kernel_size=3,
            padding=1,
        )

        self.blocks = nn.Sequential(
            *[
                NAFBlockLite(
                    channels=width
                )
                for _ in range(num_blocks)
            ]
        )

        self.ending = nn.Conv2d(
            width,
            out_channels,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x):

        residual = x

        x = self.intro(x)

        x = self.blocks(x)

        x = self.ending(x)

        # Global residual connection.
        # This stabilizes restoration of high-frequency
        # wavelet coefficients.
        if x.shape == residual.shape:
            x = x + residual

        return x


if __name__ == "__main__":

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = NAFNetLite().to(device)

    x = torch.randn(
        2,
        3,
        64,
        64,
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