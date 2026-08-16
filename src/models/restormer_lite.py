"""
SIV-AI
Lightweight Restormer-inspired restoration network.

Purpose:
    Restore the LL (low-frequency) wavelet component
    of degraded SEM images.

Architecture:
    Input Projection
        ↓
    Restormer Blocks
        ├── LayerNorm
        ├── MDTA Channel Attention
        ├── Residual Connection
        ├── LayerNorm
        ├── GDFN
        └── Residual Connection
        ↓
    Output Projection
        ↓
    Restored LL component

This is a lightweight Restormer-inspired implementation,
not the full original Restormer architecture.
"""

from pathlib import Path
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------
# Project root
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ==================================================
# Layer Normalization
# ==================================================

class BiasFreeLayerNorm(nn.Module):
    """
    Bias-free LayerNorm used by Restormer-style blocks.

    Input:
        [B, C, H, W]

    Normalization:
        performed across channel dimension.
    """

    def __init__(self, channels):
        super().__init__()

        self.weight = nn.Parameter(
            torch.ones(channels)
        )

    def forward(self, x):

        # [B, C, H, W]
        b, c, h, w = x.shape

        # Move channels to last dimension
        x = x.permute(
            0, 2, 3, 1
        )

        # Normalize channels
        variance = x.var(
            dim=-1,
            keepdim=True,
            unbiased=False,
        )

        x = x / torch.sqrt(
            variance + 1e-5
        )

        x = x * self.weight

        # Restore [B, C, H, W]
        x = x.permute(
            0, 3, 1, 2
        )

        return x


class WithBiasLayerNorm(nn.Module):
    """
    LayerNorm with learnable bias.
    """

    def __init__(self, channels):
        super().__init__()

        self.weight = nn.Parameter(
            torch.ones(channels)
        )

        self.bias = nn.Parameter(
            torch.zeros(channels)
        )

    def forward(self, x):

        b, c, h, w = x.shape

        x = x.permute(
            0, 2, 3, 1
        )

        mean = x.mean(
            dim=-1,
            keepdim=True
        )

        variance = x.var(
            dim=-1,
            keepdim=True,
            unbiased=False,
        )

        x = (
            (x - mean)
            / torch.sqrt(
                variance + 1e-5
            )
        )

        x = (
            x * self.weight
            + self.bias
        )

        x = x.permute(
            0, 3, 1, 2
        )

        return x


class LayerNorm2d(nn.Module):
    """
    Wrapper used by the Restormer blocks.

    Default:
        Bias-free LayerNorm.
    """

    def __init__(
        self,
        channels,
        bias=False,
    ):
        super().__init__()

        if bias:
            self.body = WithBiasLayerNorm(
                channels
            )
        else:
            self.body = BiasFreeLayerNorm(
                channels
            )

    def forward(self, x):
        return self.body(x)


# ==================================================
# MDTA
# ==================================================

class MDTA(nn.Module):
    """
    Multi-DConv Head Transposed Attention.

    Lightweight Restormer-inspired attention.

    Important:
        Attention is calculated across feature channels,
        not directly across all spatial pixels.

    This avoids constructing a huge
    [H*W, H*W] attention matrix.

    Input:
        [B, C, H, W]

    Output:
        [B, C, H, W]
    """

    def __init__(
        self,
        channels,
        num_heads=4,
    ):
        super().__init__()

        if channels % num_heads != 0:
            raise ValueError(
                "channels must be divisible by num_heads"
            )

        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        # Generate Q, K, V
        self.qkv = nn.Conv2d(
            channels,
            channels * 3,
            kernel_size=1,
            bias=False,
        )

        # Depth-wise convolution
        self.qkv_dw = nn.Conv2d(
            channels * 3,
            channels * 3,
            kernel_size=3,
            padding=1,
            groups=channels * 3,
            bias=False,
        )

        # Learnable temperature
        self.temperature = nn.Parameter(
            torch.ones(
                num_heads,
                1,
                1,
            )
        )

        # Output projection
        self.project_out = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
            bias=False,
        )

    def forward(self, x):

        # --------------------------------------------------
        # Input dimensions
        # --------------------------------------------------

        b, c, h, w = x.shape

        # --------------------------------------------------
        # QKV projection
        # --------------------------------------------------

        qkv = self.qkv(x)

        qkv = self.qkv_dw(
            qkv
        )

        q, k, v = qkv.chunk(
            3,
            dim=1
        )

        # --------------------------------------------------
        # Split into attention heads
        # --------------------------------------------------

        q = q.reshape(
            b,
            self.num_heads,
            self.head_dim,
            h * w,
        )

        k = k.reshape(
            b,
            self.num_heads,
            self.head_dim,
            h * w,
        )

        v = v.reshape(
            b,
            self.num_heads,
            self.head_dim,
            h * w,
        )

        # --------------------------------------------------
        # Normalize Q and K
        # --------------------------------------------------

        q = F.normalize(
            q,
            dim=-1,
        )

        k = F.normalize(
            k,
            dim=-1,
        )

        # --------------------------------------------------
        # Channel attention
        # --------------------------------------------------
        #
        # q:
        # [B, heads, head_dim, HW]
        #
        # k^T:
        # [B, heads, HW, head_dim]
        #
        # result:
        # [B, heads, head_dim, head_dim]
        #
        # Therefore attention complexity depends on
        # feature channels rather than HW × HW.
        # --------------------------------------------------

        attention = torch.matmul(
            q,
            k.transpose(
                -2,
                -1,
            ),
        )

        attention = (
            attention
            * self.temperature
        )

        attention = torch.softmax(
            attention,
            dim=-1,
        )

        # --------------------------------------------------
        # Apply attention to V
        # --------------------------------------------------
        #
        # attention:
        # [B, heads, head_dim, head_dim]
        #
        # v:
        # [B, heads, head_dim, HW]
        #
        # output:
        # [B, heads, head_dim, HW]
        # --------------------------------------------------

        output = torch.matmul(
            attention,
            v,
        )

        # --------------------------------------------------
        # Merge heads
        # --------------------------------------------------

        output = output.reshape(
            b,
            c,
            h,
            w,
        )

        # --------------------------------------------------
        # Output projection
        # --------------------------------------------------

        output = self.project_out(
            output
        )

        return output


# ==================================================
# GDFN
# ==================================================

class GDFN(nn.Module):
    """
    Gated-Dconv Feed-Forward Network.

    Lightweight Restormer-inspired feed-forward block.

    Input:
        [B, C, H, W]

    Output:
        [B, C, H, W]
    """

    def __init__(
        self,
        channels,
        expansion=2.0,
    ):
        super().__init__()

        hidden = int(
            channels * expansion
        )

        # Expand features
        self.project_in = nn.Conv2d(
            channels,
            hidden * 2,
            kernel_size=1,
            bias=False,
        )

        # Depth-wise convolution
        self.dwconv = nn.Conv2d(
            hidden * 2,
            hidden * 2,
            kernel_size=3,
            padding=1,
            groups=hidden * 2,
            bias=False,
        )

        # Project back
        self.project_out = nn.Conv2d(
            hidden,
            channels,
            kernel_size=1,
            bias=False,
        )

    def forward(self, x):

        x = self.project_in(
            x
        )

        x = self.dwconv(
            x
        )

        x1, x2 = x.chunk(
            2,
            dim=1
        )

        # Gated activation
        x = F.gelu(
            x1
        ) * x2

        x = self.project_out(
            x
        )

        return x


# ==================================================
# Restormer Block
# ==================================================

class RestormerBlockLite(nn.Module):
    """
    Lightweight Restormer-inspired block.

    Structure:

        LayerNorm
             ↓
        MDTA Attention
             ↓
        Residual
             ↓
        LayerNorm
             ↓
        GDFN
             ↓
        Residual
    """

    def __init__(
        self,
        channels=32,
        num_heads=4,
    ):
        super().__init__()

        self.norm1 = LayerNorm2d(
            channels
        )

        self.attention = MDTA(
            channels,
            num_heads,
        )

        self.norm2 = LayerNorm2d(
            channels
        )

        self.ffn = GDFN(
            channels
        )

    def forward(self, x):

        # Attention branch
        x = (
            x
            + self.attention(
                self.norm1(x)
            )
        )

        # Feed-forward branch
        x = (
            x
            + self.ffn(
                self.norm2(x)
            )
        )

        return x


# ==================================================
# Restormer Lite
# ==================================================

class RestormerLite(nn.Module):
    """
    Lightweight Restormer-inspired restoration network.

    Designed for the LL wavelet component of SEM images.

    Input:
        [B, 1, H, W]

    Output:
        [B, 1, H, W]

    Example:
        Input:
            [2, 1, 64, 64]

        Output:
            [2, 1, 64, 64]
    """

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        width=32,
        num_blocks=4,
        num_heads=4,
    ):
        super().__init__()

        # --------------------------------------------------
        # Input projection
        # --------------------------------------------------

        self.input_projection = nn.Conv2d(
            in_channels,
            width,
            kernel_size=3,
            padding=1,
        )

        # --------------------------------------------------
        # Transformer blocks
        # --------------------------------------------------

        self.blocks = nn.Sequential(
            *[
                RestormerBlockLite(
                    channels=width,
                    num_heads=num_heads,
                )
                for _ in range(num_blocks)
            ]
        )

        # --------------------------------------------------
        # Output projection
        # --------------------------------------------------

        self.output_projection = nn.Conv2d(
            width,
            out_channels,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x):

        # Keep original input for residual connection
        residual = x

        # Input projection
        x = self.input_projection(
            x
        )

        # Restormer blocks
        x = self.blocks(
            x
        )

        # Output projection
        x = self.output_projection(
            x
        )

        # Global residual connection
        if (
            x.shape
            == residual.shape
        ):
            x = x + residual

        return x


# ==================================================
# Self Test
# ==================================================

if __name__ == "__main__":

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device:",
        device
    )

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    # --------------------------------------------------
    # Test input
    # --------------------------------------------------

    x = torch.randn(
        2,
        1,
        64,
        64,
        device=device,
    )

    # --------------------------------------------------
    # Create model
    # --------------------------------------------------

    model = RestormerLite(
        in_channels=1,
        out_channels=1,
        width=32,
        num_blocks=4,
        num_heads=4,
    ).to(device)

    # --------------------------------------------------
    # Parameter count
    # --------------------------------------------------

    parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        "Input shape:",
        x.shape
    )

    # --------------------------------------------------
    # Forward pass
    # --------------------------------------------------

    model.eval()

    with torch.no_grad():

        y = model(x)

    print(
        "Output shape:",
        y.shape
    )

    print(
        "Parameters:",
        parameters
    )

    # --------------------------------------------------
    # Shape verification
    # --------------------------------------------------

    assert y.shape == x.shape

    print(
        "RestormerLite test PASSED"
    )
