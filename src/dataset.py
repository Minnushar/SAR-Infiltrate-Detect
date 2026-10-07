import numpy as np
import torch
from torch.utils.data import Dataset

from src.data_loader import SARDataset
from src.patching import extract_patches

class SARPatchDataset(Dataset):
    def __init__(self):
        self.sar_dataset = SARDataset()

        self.patches = []

        # Load every VV + VH image
        for i in range(len(self.sar_dataset)):
            image, filename = self.sar_dataset[i]

            # Extract 64x64 patches
            patches = extract_patches(image)

            # Store patches
            self.patches.extend(patches)

        self.patches = np.array(self.patches, dtype=np.float32)

        print("Total patches:", len(self.patches))
        print("Patch array shape:", self.patches.shape)

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, index):
        patch = torch.tensor(self.patches[index], dtype=torch.float32)

        return patch


if __name__ == "__main__":

    dataset = SARPatchDataset()

    print("Dataset length:", len(dataset))

    sample = dataset[0]

    print("Single patch shape:", sample.shape)
    print("Single patch dtype:", sample.dtype)