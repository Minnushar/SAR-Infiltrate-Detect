import os
import numpy as np
import rasterio


class SARDataset:
    def __init__(self, vv_dir="data/raw/SAR/VV", vh_dir="data/raw/SAR/VH"):
        self.vv_dir = vv_dir
        self.vh_dir = vh_dir

        # Find files that exist in both VV and VH folders
        vv_files = {
            f for f in os.listdir(vv_dir)
            if f.lower().endswith(".tif")
        }

        vh_files = {
            f for f in os.listdir(vh_dir)
            if f.lower().endswith(".tif")
        }

        self.files = sorted(vv_files.intersection(vh_files))

        if not self.files:
            raise ValueError("No matching VV/VH files found.")

        print(f"Found {len(self.files)} matching VV/VH pairs.")

    def load_sample(self, filename):
        """Load one matching VV + VH pair."""

        vv_path = os.path.join(self.vv_dir, filename)
        vh_path = os.path.join(self.vh_dir, filename)

        with rasterio.open(vv_path) as src:
            vv = src.read(1).astype(np.float32)

        with rasterio.open(vh_path) as src:
            vh = src.read(1).astype(np.float32)

        # Normalize each polarization to 0-1
        vv = (vv - vv.min()) / (vv.max() - vv.min() + 1e-8)
        vh = (vh - vh.min()) / (vh.max() - vh.min() + 1e-8)

        # Combine into a 2-channel array
        sample = np.stack([vv, vh], axis=0)

        return sample

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        filename = self.files[index]
        return self.load_sample(filename), filename


if __name__ == "__main__":
    dataset = SARDataset()

    print("Dataset size:", len(dataset))

    sample, filename = dataset[0]

    print("Sample filename:", filename)
    print("Sample shape:", sample.shape)
    print("VV min/max:", sample[0].min(), sample[0].max())
    print("VH min/max:", sample[1].min(), sample[1].max())