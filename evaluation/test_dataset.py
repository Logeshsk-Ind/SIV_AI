from pathlib import Path
from src.data.kla_dataset import KLADataset

print("=" * 70)
print("SIV-AI KLA DATASET TEST")
print("=" * 70)

# CHANGE THIS ONLY IF YOUR DATASET IS SOMEWHERE ELSE
DATASET_ROOT = Path(r"E:\project\SIV_AI\data\raw\train\train")

print("Dataset:", DATASET_ROOT)

dataset = KLADataset(DATASET_ROOT)

print("Total samples:", len(dataset))

sample = dataset[0]

print()
print("First sample:")
print("Name :", sample["name"])
print("Noisy:", tuple(sample["noisy"].shape))
print("GT   :", tuple(sample["gt"].shape))

print()
print("Noisy min:", sample["noisy"].min().item())
print("Noisy max:", sample["noisy"].max().item())
print("GT min   :", sample["gt"].min().item())
print("GT max   :", sample["gt"].max().item())

print()
print("=" * 70)
print("DATASET TEST PASSED")
print("=" * 70)