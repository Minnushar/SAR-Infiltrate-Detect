import numpy as np
import torch
import matplotlib.pyplot as plt

from src.dataset import SARPatchDataset
from src.autoencoder import SARAutoencoder
from src.context_score import calculate_context_score
from src.polarization_score import calculate_polarization_score
from src.region_detection import (
    detect_anomalous_regions,
    rank_regions
)


def normalize_scores(scores):
    """
    Normalize scores to the range 0-1.
    """

    scores = np.asarray(
        scores,
        dtype=np.float32
    )

    return (
        scores - scores.min()
    ) / (
        scores.max() - scores.min() + 1e-8
    )


def calculate_spatial_scores(
    model,
    dataset,
    device
):
    """
    Calculate autoencoder reconstruction error
    for every SAR patch.

    Higher reconstruction error indicates
    stronger spatial anomaly evidence.
    """

    model.eval()

    scores = []

    with torch.no_grad():

        for i in range(len(dataset)):

            patch = dataset[i]

            patch = patch.unsqueeze(0).to(device)

            reconstruction = model(patch)

            error = torch.mean(
                (patch - reconstruction) ** 2
            ).item()

            scores.append(error)

    return np.array(
        scores,
        dtype=np.float32
    )


def calculate_polarization_scores(dataset):
    """
    Calculate VV/VH polarization score
    for every SAR patch.
    """

    scores = []

    for i in range(len(dataset)):

        patch = dataset[i].numpy()

        score = calculate_polarization_score(
            patch
        )

        scores.append(score)

    return np.array(
        scores,
        dtype=np.float32
    )


def calculate_cmeas_for_image(
    spatial_scores,
    polarization_scores,
    image_index=0,
    spatial_weight=0.5,
    context_weight=0.2,
    polarization_weight=0.3
):
    """
    Calculate CMEAS for one SAR image.

    Current experimental weights:

        Spatial       = 50%
        Context       = 20%
        Polarization = 30%

    These are initial experimental weights.
    They are not claimed to be optimal.
    """

    # Each 512x512 image contains
    # 64 patches arranged as 8x8.
    start = image_index * 64
    end = start + 64

    # Extract scores for this image
    spatial = spatial_scores[start:end]

    polarization = polarization_scores[start:end]

    # Convert into 8x8 grids
    spatial_grid = spatial.reshape(8, 8)

    polarization_grid = polarization.reshape(8, 8)

    # Calculate local-context evidence
    context_grid = calculate_context_score(
        spatial_grid
    )

    # Normalize each evidence source
    spatial_norm = normalize_scores(
        spatial_grid
    )

    context_norm = normalize_scores(
        context_grid
    )

    polarization_norm = normalize_scores(
        polarization_grid
    )

    # -------------------------------------------------
    # CMEAS MULTI-EVIDENCE FUSION
    # -------------------------------------------------

    cmeas = (
        spatial_weight * spatial_norm
        + context_weight * context_norm
        + polarization_weight * polarization_norm
    )

    return (
        spatial_norm,
        context_norm,
        polarization_norm,
        cmeas
    )


def save_heatmap(
    score_grid,
    output_path,
    title
):
    """
    Save an anomaly score grid as a heatmap.
    """

    plt.figure(figsize=(8, 8))

    plt.imshow(
        score_grid,
        cmap="hot",
        interpolation="nearest"
    )

    plt.colorbar(
        label="Normalized Anomaly Score"
    )

    plt.title(title)

    plt.xlabel("Patch Column")

    plt.ylabel("Patch Row")

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()


if __name__ == "__main__":

    # =================================================
    # 1. DEVICE
    # =================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Using device:", device)

    # =================================================
    # 2. LOAD DATASET
    # =================================================

    dataset = SARPatchDataset()

    # =================================================
    # 3. LOAD TRAINED AUTOENCODER
    # =================================================

    model = SARAutoencoder().to(device)

    model.load_state_dict(
        torch.load(
            "models/sar_autoencoder.pth",
            map_location=device
        )
    )

    print(
        "Trained model loaded successfully."
    )

    # =================================================
    # 4. CALCULATE SPATIAL SCORES
    # =================================================

    print(
        "\nCalculating spatial scores..."
    )

    spatial_scores = calculate_spatial_scores(
        model,
        dataset,
        device
    )

    print(
        "Spatial scores calculated."
    )

    # =================================================
    # 5. CALCULATE POLARIZATION SCORES
    # =================================================

    print(
        "\nCalculating polarization scores..."
    )

    polarization_scores = (
        calculate_polarization_scores(
            dataset
        )
    )

    print(
        "Polarization scores calculated."
    )

    # =================================================
    # 6. CALCULATE CMEAS
    # =================================================

    (
        spatial_norm,
        context_norm,
        polarization_norm,
        cmeas
    ) = calculate_cmeas_for_image(
        spatial_scores,
        polarization_scores,
        image_index=0
    )

    print(
        "\nCMEAS analysis for 0A:"
    )

    print(
        "Maximum spatial evidence:",
        spatial_norm.max()
    )

    print(
        "Maximum context evidence:",
        context_norm.max()
    )

    print(
        "Maximum polarization evidence:",
        polarization_norm.max()
    )

    print(
        "Maximum CMEAS score:",
        cmeas.max()
    )

    # =================================================
    # 7. FIND HIGHEST CMEAS PATCH
    # =================================================

    highest_location = np.unravel_index(
        np.argmax(cmeas),
        cmeas.shape
    )

    patch_number = (
        highest_location[0] * 8
        + highest_location[1]
    )

    print(
        "Highest CMEAS patch:",
        patch_number
    )

    print(
        "Highest CMEAS location:",
        highest_location
    )

    # =================================================
    # 8. DETECT CANDIDATE REGIONS
    # =================================================

    (
        anomaly_mask,
        labeled_regions,
        region_count,
        threshold
    ) = detect_anomalous_regions(
        cmeas,
        threshold_percentile=90
    )

    print(
        "\nRegion detection for 0A:"
    )

    print(
        "CMEAS threshold:",
        threshold
    )

    print(
        "Number of candidate regions:",
        region_count
    )

    print(
        "\nAnomaly mask:"
    )

    print(
        anomaly_mask.astype(int)
    )

    # =================================================
    # 9. RANK CANDIDATE REGIONS
    # =================================================

    regions = rank_regions(
        cmeas,
        labeled_regions,
        region_count
    )

    print(
        "\nRanked candidate regions:"
    )

    for region in regions:

        print(
            f"Region {region['region_id']} | "
            f"Patches: {region['patch_count']} | "
            f"Mean CMEAS: {region['mean_score']:.4f} | "
            f"Max CMEAS: {region['max_score']:.4f} | "
            f"Locations: {region['positions']}"
        )

    # =================================================
    # 10. SAVE HEATMAPS
    # =================================================

    save_heatmap(
        spatial_norm,
        "outputs/anomaly_maps/0A_spatial.png",
        "Spatial Anomaly - 0A"
    )

    save_heatmap(
        context_norm,
        "outputs/anomaly_maps/0A_context.png",
        "Local Context Anomaly - 0A"
    )

    save_heatmap(
        polarization_norm,
        "outputs/anomaly_maps/0A_polarization.png",
        "Polarization Evidence - 0A"
    )

    save_heatmap(
        cmeas,
        "outputs/anomaly_maps/0A_cmeas.png",
        "CMEAS Multi-Evidence Anomaly Map - 0A"
    )

    print(
        "\nHeatmaps saved successfully."
    )