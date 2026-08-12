from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class KLADataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)

        self.gt_dir = self.root_dir / "GT"
        self.noisy_dir = self.root_dir / "NoisyLR"

        self.gt_files = sorted(self.gt_dir.glob("*.npy"))
        self.noisy_files = sorted(self.noisy_dir.glob("*.npy"))

        if len(self.gt_files) != len(self.noisy_files):
            raise ValueError(
                f"GT count ({len(self.gt_files)}) does not match "
                f"NoisyLR count ({len(self.noisy_files)})"
            )

        for gt_file, noisy_file in zip(self.gt_files, self.noisy_files):
            if gt_file.stem != noisy_file.stem:
                raise ValueError(
                    f"Pair mismatch: {gt_file.name} <-> {noisy_file.name}"
                )

    def __len__(self):
        return len(self.gt_files)

    def __getitem__(self, index):
        gt = np.load(self.gt_files[index]).astype(np.float32)
        noisy = np.load(self.noisy_files[index]).astype(np.float32)

        gt = torch.from_numpy(gt).unsqueeze(0)
        noisy = torch.from_numpy(noisy).unsqueeze(0)

        return {
            "noisy": noisy,
            "gt": gt,
            "name": self.gt_files[index].stem,
        }