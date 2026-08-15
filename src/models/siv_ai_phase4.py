import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# SIMPLE GATE
# ============================================================

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


# ============================================================
# NAF BLOCK
# ============================================================

class NAFBlock(nn.Module):

    def __init__(self, channels=32):
        super().__init__()

        self.beta = nn.Parameter(torch.zeros((1, channels, 1, 1)))
        self.gamma = nn.Parameter(torch.zeros((1, channels, 1, 1)))

        self.norm1 = nn.GroupNorm(1, channels)

        self.conv1 = nn.Conv2d(
            channels,
            channels * 2,
            kernel_size=1,
            bias=True
        )

        self.dwconv = nn.Conv2d(
            channels * 2,
            channels * 2,
            kernel_size=3,
            padding=1,
            groups=channels * 2,
            bias=True
        )

        self.sg = SimpleGate()

        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
            bias=True
        )

        self.norm2 = nn.GroupNorm(1, channels)

        self.ffn = nn.Sequential(
            nn.Conv2d(
                channels,
                channels * 2,
                kernel_size=1,
                bias=True
            ),
            SimpleGate(),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=1,
                bias=True
            )
        )

    def forward(self, x):

        y = self.norm1(x)

        y = self.conv1(y)
        y = self.dwconv(y)
        y = self.sg(y)
        y = self.conv2(y)

        x = x + self.beta * y

        y = self.norm2(x)
        y = self.ffn(y)

        x = x + self.gamma * y

        return x


# ============================================================
# NAFNET BRANCH
# ============================================================

class NAFNetBranch(nn.Module):

    def __init__(self, channels=32, blocks=4):
        super().__init__()

        self.blocks = nn.Sequential(
            *[
                NAFBlock(channels)
                for _ in range(blocks)
            ]
        )

    def forward(self, x):
        return self.blocks(x)


# ============================================================
# MDTA
# ============================================================

class MDTA(nn.Module):

    def __init__(self, channels=32):
        super().__init__()

        self.norm = nn.GroupNorm(1, channels)

        self.qkv = nn.Conv2d(
            channels,
            channels * 3,
            kernel_size=1,
            bias=True
        )

        self.q_dwconv = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=True
        )

        self.k_dwconv = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=True
        )

        self.v_dwconv = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=True
        )

        self.project = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
            bias=True
        )

    def forward(self, x):

        identity = x

        x = self.norm(x)

        qkv = self.qkv(x)

        q, k, v = torch.chunk(qkv, 3, dim=1)

        q = self.q_dwconv(q)
        k = self.k_dwconv(k)
        v = self.v_dwconv(v)

        b, c, h, w = q.shape

        # Channel-wise attention
        q = q.view(b, c, h * w)
        k = k.view(b, c, h * w)
        v = v.view(b, c, h * w)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attention = torch.bmm(q, k.transpose(1, 2))

        attention = torch.softmax(attention, dim=-1)

        out = torch.bmm(attention, v)

        out = out.view(b, c, h, w)

        out = self.project(out)

        return identity + out


# ============================================================
# RESTORMER BLOCK
# ============================================================

class RestormerBlock(nn.Module):

    def __init__(self, channels=32):
        super().__init__()

        self.attention = MDTA(channels)

        self.norm = nn.GroupNorm(1, channels)

        self.ffn = nn.Sequential(
            nn.Conv2d(
                channels,
                channels * 2,
                kernel_size=1,
                bias=True
            ),
            nn.GELU(),
            nn.Conv2d(
                channels * 2,
                channels,
                kernel_size=1,
                bias=True
            )
        )

    def forward(self, x):

        x = self.attention(x)

        y = self.norm(x)
        y = self.ffn(y)

        x = x + y

        return x


# ============================================================
# RESTORMER BRANCH
# ============================================================

class RestormerBranch(nn.Module):

    def __init__(self, channels=32, blocks=2):
        super().__init__()

        self.blocks = nn.Sequential(
            *[
                RestormerBlock(channels)
                for _ in range(blocks)
            ]
        )

    def forward(self, x):
        return self.blocks(x)


# ============================================================
# INPUT ENCODER
# ============================================================

class InputEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(

            nn.Conv2d(
                1,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.GELU(),

            nn.Conv2d(
                32,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.GELU()
        )

        self.upsample = nn.Sequential(

            nn.Conv2d(
                32,
                128,
                kernel_size=3,
                padding=1
            ),

            nn.PixelShuffle(2),

            nn.GELU()
        )

    def forward(self, x):

        x = self.conv(x)

        x = self.upsample(x)

        return x


# ============================================================
# HAAR DWT
# ============================================================

class HaarDWT(nn.Module):

    def forward(self, x):

        x00 = x[:, :, 0::2, 0::2]
        x01 = x[:, :, 0::2, 1::2]
        x10 = x[:, :, 1::2, 0::2]
        x11 = x[:, :, 1::2, 1::2]

        ll = (x00 + x01 + x10 + x11) * 0.5

        lh = (x00 - x01 + x10 - x11) * 0.5

        hl = (x00 + x01 - x10 - x11) * 0.5

        hh = (x00 - x01 - x10 + x11) * 0.5

        return ll, lh, hl, hh


# ============================================================
# HAAR IDWT
# ============================================================

class HaarIDWT(nn.Module):

    def forward(self, ll, lh, hl, hh):

        b, c, h, w = ll.shape

        x = torch.zeros(
            b,
            c,
            h * 2,
            w * 2,
            dtype=ll.dtype,
            device=ll.device
        )

        x[:, :, 0::2, 0::2] = (
            ll + lh + hl + hh
        ) * 0.5

        x[:, :, 0::2, 1::2] = (
            ll - lh + hl - hh
        ) * 0.5

        x[:, :, 1::2, 0::2] = (
            ll + lh - hl - hh
        ) * 0.5

        x[:, :, 1::2, 1::2] = (
            ll - lh - hl + hh
        ) * 0.5

        return x


# ============================================================
# HIGH FREQUENCY RECONSTRUCTION
# ============================================================

class HFFrequencyReconstruction(nn.Module):

    def __init__(self):
        super().__init__()

        self.projection = nn.Conv2d(
            32,
            96,
            kernel_size=1,
            bias=True
        )

    def forward(self, x):

        x = self.projection(x)

        return x


# ============================================================
# RECONSTRUCTION HEAD
# ============================================================

class ReconstructionHead(nn.Module):

    def __init__(self):
        super().__init__()

        self.head = nn.Sequential(

            nn.Conv2d(
                32,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.GELU(),

            nn.Conv2d(
                32,
                1,
                kernel_size=3,
                padding=1
            ),

            nn.Sigmoid()
        )

    def forward(self, x):
        return self.head(x)


# ============================================================
# SIV-AI PHASE 4
# ============================================================

class SIVAI(nn.Module):

    def __init__(self):

        super().__init__()

        self.encoder = InputEncoder()

        self.dwt = HaarDWT()

        self.nafnet = NAFNetBranch(
            channels=32,
            blocks=4
        )

        self.hf_projection = nn.Conv2d(
            96,
            32,
            kernel_size=1,
            bias=True
        )

        self.restormer = RestormerBranch(
            channels=32,
            blocks=2
        )

        self.hf_reconstruction = HFFrequencyReconstruction()

        self.idwt = HaarIDWT()

        self.reconstruction = ReconstructionHead()

    def forward(self, x):

        # ----------------------------------------------------
        # Encoder
        # [B,1,128,128] -> [B,32,256,256]
        # ----------------------------------------------------

        x = self.encoder(x)

        # ----------------------------------------------------
        # Wavelet decomposition
        # [B,32,256,256]
        # -> four [B,32,128,128]
        # ----------------------------------------------------

        ll, lh, hl, hh = self.dwt(x)

        # ----------------------------------------------------
        # Low-frequency restoration
        # ----------------------------------------------------

        ll_restored = self.nafnet(ll)

        # ----------------------------------------------------
        # Combine high-frequency subbands
        # 3 × 32 = 96 channels
        # ----------------------------------------------------

        hf = torch.cat(
            [lh, hl, hh],
            dim=1
        )

        # ----------------------------------------------------
        # 96 -> 32
        # ----------------------------------------------------

        hf = self.hf_projection(hf)

        # ----------------------------------------------------
        # High-frequency restoration
        # ----------------------------------------------------

        hf = self.restormer(hf)

        # ----------------------------------------------------
        # 32 -> 96
        # ----------------------------------------------------

        hf = self.hf_reconstruction(hf)

        # ----------------------------------------------------
        # Split back into LH / HL / HH
        # ----------------------------------------------------

        lh_restored, hl_restored, hh_restored = torch.chunk(
            hf,
            3,
            dim=1
        )

        # ----------------------------------------------------
        # Inverse wavelet
        # [B,32,128,128] -> [B,32,256,256]
        # ----------------------------------------------------

        x = self.idwt(
            ll_restored,
            lh_restored,
            hl_restored,
            hh_restored
        )

        # ----------------------------------------------------
        # Final reconstruction
        # ----------------------------------------------------

        x = self.reconstruction(x)

        return x


# ============================================================
# MODEL TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("SIV-AI PHASE 4 MODEL TEST")
    print("=" * 60)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    model = SIVAI().to(device)

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print("Parameters:", total_params)
    print("Trainable:", trainable_params)

    x = torch.rand(
        2,
        1,
        128,
        128,
        device=device
    )

    with torch.no_grad():

        y = model(x)

    print("Input :", tuple(x.shape))
    print("Output:", tuple(y.shape))

    print("=" * 60)

    if total_params == 110049:
        print("PARAMETER COUNT: PASS")
    else:
        print(
            "PARAMETER COUNT: FAIL",
            total_params,
            "expected 110049"
        )

    if tuple(y.shape) == (2, 1, 256, 256):
        print("OUTPUT SHAPE: PASS")
    else:
        print("OUTPUT SHAPE: FAIL")

    print("=" * 60)