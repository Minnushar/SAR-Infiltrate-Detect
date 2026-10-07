import numpy as np
from src.data_loader import SARDataset


def extract_patches(image, patch_size=64):
    """
    Extract non-overlapping patches from a SAR image.

    Input:
        image: NumPy array with shape (channels, height, width)

    Output:
        patches: NumPy array with shape
                 (number_of_patches, channels, patch_size, patch_size)
    """

    channels, height, width = image.shape

    patches = []

    for y in range(0, height - patch_size + 1, patch_size):
        for x in range(0, width - patch_size + 1, patch_size):

            patch = image[
                :,
                y:y + patch_size,
                x:x + patch_size
            ]

            patches.append(patch)

    return np.array(patches)


if __name__ == "__main__":

    # Load the real SAR dataset
    dataset = SARDataset()

    print("Dataset size:", len(dataset))

    # Load the first real VV + VH image
    image, filename = dataset[0]

    print("Filename:", filename)
    print("Original image shape:", image.shape)

    # Extract patches
    patches = extract_patches(image)

    print("Patches shape:", patches.shape)
    print("Number of patches:", len(patches))