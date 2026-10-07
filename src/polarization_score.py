import numpy as np

from src.dataset import SARPatchDataset


def calculate_polarization_score(patch):
    """
    Calculate a polarization anomaly score for one SAR patch.

    The patch contains:
        patch[0] = VV
        patch[1] = VH

    The score measures how different the local VV and VH
    scattering patterns are.

    Higher score = stronger polarization difference.
    """

    vv = patch[0]
    vh = patch[1]

    # Calculate absolute VV-VH difference
    difference = np.abs(vv - vh)

    # Average difference across the patch
    score = np.mean(difference)

    return float(score)


if __name__ == "__main__":

    # Load the real SAR dataset
    dataset = SARPatchDataset()

    print("\nCalculating polarization scores...")

    scores = []

    for i in range(len(dataset)):

        patch = dataset[i].numpy()

        score = calculate_polarization_score(patch)

        scores.append(score)

    scores = np.array(scores)

    print("\nPolarization analysis complete.")

    print("Number of patches:", len(scores))
    print("Minimum polarization score:", scores.min())
    print("Maximum polarization score:", scores.max())
    print("Mean polarization score:", scores.mean())

    # Top 10 patches
    top_indices = np.argsort(scores)[-10:][::-1]

    print("\nTop 10 patches by polarization score:")

    for rank, index in enumerate(top_indices, start=1):

        print(
            f"{rank}. Patch {index} "
            f"→ Score: {scores[index]:.6f}"
        )