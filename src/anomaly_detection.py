import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

from src.dataset import SARPatchDataset
from src.autoencoder import SARAutoencoder
from src.context_score import calculate_context_score


def calculate_reconstruction_errors(model, dataset, device):
    """
    Calculate reconstruction error for every SAR patch.
    Higher error = stronger spatial anomaly score.
    """

    model.eval()

    errors = []

    with torch.no_grad():

        for i in range(len(dataset)):

            patch = dataset[i]

            patch = patch.unsqueeze(0).to(device)

            reconstruction = model(patch)

            error = torch.mean(
                (patch - reconstruction) ** 2
            ).item()

            errors.append(error)

    return np.array(errors)


def create_heatmap(errors, image_index=0):
    """
    Create an anomaly heatmap for one 512x512 SAR image.

    Each image contains:
        64 patches = 8 x 8 grid
    """

    start = image_index * 64
    end = start + 64

    image_errors = errors[start:end]

    score_grid = image_errors.reshape(8, 8)

    heatmap = Image.fromarray(
        score_grid.astype(np.float32),
        mode="F"
    )

    heatmap = heatmap.resize(
        (512, 512),
        Image.Resampling.BILINEAR
    )

    return np.array(heatmap)


if __name__ == "__main__":

    # Device
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Using device:", device)

    # Load dataset
    dataset = SARPatchDataset()

    # Load trained model
    model = SARAutoencoder().to(device)

    model.load_state_dict(
        torch.load(
            "models/sar_autoencoder.pth",
            map_location=device
        )
    )

    print("Trained model loaded successfully.")

    # Calculate spatial anomaly scores
    errors = calculate_reconstruction_errors(
        model,
        dataset,
        device
    )

    print("\nAnomaly analysis complete.")
    print("Number of patches:", len(errors))
    print("Minimum reconstruction error:", errors.min())
    print("Maximum reconstruction error:", errors.max())
    print("Mean reconstruction error:", errors.mean())

    # -------------------------------------------------
    # LOCAL CONTEXT ANALYSIS
    # -------------------------------------------------

    # First image = first 64 patches
    spatial_grid = errors[:64].reshape(8, 8)

    # Calculate local-context scores
    context_grid = calculate_context_score(
        spatial_grid
    )

    # Normalize spatial scores
    spatial_norm = (
        spatial_grid - spatial_grid.min()
    ) / (
        spatial_grid.max() - spatial_grid.min() + 1e-8
    )

    # Normalize context scores
    context_norm = (
        context_grid - context_grid.min()
    ) / (
        context_grid.max() - context_grid.min() + 1e-8
    )

    # Combine spatial + context evidence
    combined_grid = (
        0.7 * spatial_norm +
        0.3 * context_norm
    )

    print("\nContext analysis for 0A:")
    print(
        "Maximum spatial score:",
        spatial_norm.max()
    )

    print(
        "Maximum context score:",
        context_norm.max()
    )

    print(
        "Maximum combined score:",
        combined_grid.max()
    )

    # -------------------------------------------------
    # CREATE HEATMAP
    # -------------------------------------------------

    heatmap = create_heatmap(
        errors,
        image_index=0
    )

    # Normalize heatmap for visualization
    heatmap_normalized = (
        heatmap - heatmap.min()
    ) / (
        heatmap.max() - heatmap.min() + 1e-8
    )

    # Save heatmap
    output_path = (
        "outputs/anomaly_maps/"
        "0A_anomaly_heatmap.png"
    )

    plt.figure(figsize=(8, 8))

    plt.imshow(
        heatmap_normalized,
        cmap="hot"
    )

    plt.colorbar(
        label="Spatial Anomaly Score"
    )

    plt.title(
        "SAR Spatial Anomaly Heatmap - 0A"
    )

    plt.axis("off")

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print("\nHeatmap saved to:")
    print(output_path)