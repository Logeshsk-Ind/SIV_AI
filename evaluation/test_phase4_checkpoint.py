import torch

from src.models.siv_ai_phase4 import SIVAI


# ============================================================
# CONFIG
# ============================================================

CHECKPOINT = r".\phase4_source\siv_ai_phase4_weights.pth"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("SIV-AI PHASE 4 CHECKPOINT TEST")
print("=" * 70)

print("Device:", DEVICE)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


model = SIVAI().to(DEVICE)


# ============================================================
# PARAMETER COUNT
# ============================================================

params = sum(
    p.numel()
    for p in model.parameters()
)

print("Model parameters:", params)

assert params == 110049, (
    f"Wrong architecture! "
    f"Expected 110049, got {params}"
)


# ============================================================
# LOAD WEIGHTS
# ============================================================

print()
print("Loading checkpoint:")
print(CHECKPOINT)

state_dict = torch.load(
    CHECKPOINT,
    map_location=DEVICE
)


print("Checkpoint tensors:", len(state_dict))


# ============================================================
# STRICT LOAD
# ============================================================

result = model.load_state_dict(
    state_dict,
    strict=True
)


print()
print("Checkpoint loaded successfully.")
print(result)


# ============================================================
# INFERENCE TEST
# ============================================================

model.eval()

x = torch.rand(
    2,
    1,
    128,
    128,
    device=DEVICE
)

with torch.no_grad():

    y = model(x)


print()
print("Input shape :", tuple(x.shape))
print("Output shape:", tuple(y.shape))

print(
    "Output min:",
    y.min().item()
)

print(
    "Output max:",
    y.max().item()
)

print(
    "Output mean:",
    y.mean().item()
)


# ============================================================
# FINAL
# ============================================================

assert tuple(y.shape) == (
    2,
    1,
    256,
    256
)

print()
print("=" * 70)
print("PHASE 4 CHECKPOINT TEST PASSED")
print("=" * 70)