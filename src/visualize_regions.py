import os
import numpy as np
import matplotlib.pyplot as plt
import rasterio

from src.dataset import SARPatchDataset
from src.autoencoder import SARAutoencoder
from src.context_score import calculate_context_score
from src.polarization_score import calculate_polarization_score
from src.region_detection import (
    detect_anomalous_regions,
    rank_regions
)

import torch


# ============================================================
# NORMALIZATION
# ============================================================

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


# ============================================================
# LOAD ORIGINAL SAR IMAGE
# ============================================================

def load_sar_image(
    filename
):
    """
    Load the VV SAR image for visualization.

    The VV image is used as the grayscale background.
    """

    path = os.path.join(
        "data",
        "raw",
        "SAR",
        "VV",
        filename
    )

    with rasterio.open(path) as src:

        image = src.read(
            1
        ).astype(
            np.float32
        )

    image = (
        image - image.min()
    ) / (
        image.max()
        - image.min()
        + 1e-8
    )

    return image


# ============================================================
# CALCULATE CMEAS
# ============================================================

def calculate_cmeas(
    model,
    dataset,
    image_index,
    device
):
    """
    Calculate spatial, context, polarization
    and CMEAS scores for one image.
    """

    start = image_index * 64
    end = start + 64

    spatial_scores = []

    polarization_scores = []

    model.eval()

    with torch.no_grad():

        for i in range(
            start,
            end
        ):

            patch = (
                dataset[i]
                .unsqueeze(0)
                .to(device)
            )

            reconstruction = model(
                patch
            )

            error = torch.mean(
                (patch - reconstruction) ** 2
            ).item()

            spatial_scores.append(
                error
            )

            patch_numpy = (
                dataset[i]
                .numpy()
            )

            polarization_score = (
                calculate_polarization_score(
                    patch_numpy
                )
            )

            polarization_scores.append(
                polarization_score
            )

    spatial_grid = np.array(
        spatial_scores
    ).reshape(
        8,
        8
    )

    polarization_grid = np.array(
        polarization_scores
    ).reshape(
        8,
        8
    )

    # --------------------------------------------------------
    # CONTEXT
    # --------------------------------------------------------

    context_grid = (
        calculate_context_score(
            spatial_grid
        )
    )

    # --------------------------------------------------------
    # NORMALIZATION
    # --------------------------------------------------------

    spatial_norm = normalize_scores(
        spatial_grid
    )

    context_norm = normalize_scores(
        context_grid
    )

    polarization_norm = normalize_scores(
        polarization_grid
    )

    # --------------------------------------------------------
    # CMEAS
    # --------------------------------------------------------

    cmeas = (
        0.5 * spatial_norm
        + 0.2 * context_norm
        + 0.3 * polarization_norm
    )

    return cmeas


# ============================================================
# CONVERT PATCH REGION TO IMAGE COORDINATES
# ============================================================

def region_to_bbox(
    positions,
    patch_size=64
):
    """
    Convert patch-grid positions into
    image pixel bounding-box coordinates.

    Each patch is 64x64 pixels.

    Returns:
        xmin, ymin, xmax, ymax
    """

    rows = [
        position[0]
        for position in positions
    ]

    cols = [
        position[1]
        for position in positions
    ]

    min_row = min(rows)
    max_row = max(rows)

    min_col = min(cols)
    max_col = max(cols)

    xmin = (
        min_col
        * patch_size
    )

    ymin = (
        min_row
        * patch_size
    )

    xmax = (
        (max_col + 1)
        * patch_size
    )

    ymax = (
        (max_row + 1)
        * patch_size
    )

    return (
        xmin,
        ymin,
        xmax,
        ymax
    )


# ============================================================
# SAVE CMEAS HEATMAP
# ============================================================

def save_cmeas_heatmap(
    cmeas,
    filename
):
    """
    Save the 8x8 CMEAS heatmap.
    """

    os.makedirs(
        "outputs/anomaly_maps",
        exist_ok=True
    )

    plt.figure(
        figsize=(8, 8)
    )

    plt.imshow(
        cmeas,
        cmap="hot",
        interpolation="nearest"
    )

    plt.colorbar(
        label="CMEAS Score"
    )

    plt.title(
        f"CMEAS Anomaly Map - {filename}"
    )

    plt.xlabel(
        "Patch Column"
    )

    plt.ylabel(
        "Patch Row"
    )

    output_path = os.path.join(
        "outputs",
        "anomaly_maps",
        filename.replace(
            ".tif",
            "_cmeas.png"
        )
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    return output_path


# ============================================================
# SAVE REGION OVERLAY
# ============================================================

def save_region_overlay(
    image,
    regions,
    filename
):
    """
    Draw candidate-region bounding boxes
    on the original SAR image.

    The highest-priority region is marked
    as Region 1.
    """

    os.makedirs(
        "outputs/anomaly_maps",
        exist_ok=True
    )

    plt.figure(
        figsize=(10, 10)
    )

    plt.imshow(
        image,
        cmap="gray"
    )

    for rank, region in enumerate(
        regions,
        start=1
    ):

        (
            xmin,
            ymin,
            xmax,
            ymax
        ) = region_to_bbox(
            region["positions"]
        )

        width = (
            xmax - xmin
        )

        height = (
            ymax - ymin
        )

        rectangle = plt.Rectangle(
            (
                xmin,
                ymin
            ),
            width,
            height,
            fill=False,
            linewidth=2
        )

        plt.gca().add_patch(
            rectangle
        )

        # ----------------------------------------------------
        # LABEL
        # ----------------------------------------------------

        label = (
            f"R{rank} | "
            f"P:{region['priority_score']:.2f}"
        )

        plt.text(
            xmin + 3,
            ymin + 15,
            label,
            fontsize=9,
            backgroundcolor="white"
        )

    plt.title(
        f"CMEAS Candidate Regions - {filename}"
    )

    plt.axis(
        "off"
    )

    output_path = os.path.join(
        "outputs",
        "anomaly_maps",
        filename.replace(
            ".tif",
            "_regions.png"
        )
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    return output_path


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # DEVICE
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Using device:",
        device
    )

    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    dataset = SARPatchDataset()

    filenames = (
        dataset.sar_dataset.files
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = SARAutoencoder().to(
        device
    )

    model.load_state_dict(
        torch.load(
            "models/sar_autoencoder.pth",
            map_location=device
        )
    )

    print(
        "Trained model loaded successfully."
    )

    # --------------------------------------------------------
    # PROCESS ALL IMAGES
    # --------------------------------------------------------

    print(
        "\nGenerating CMEAS visualizations..."
    )

    for image_index, filename in enumerate(
        filenames
    ):

        print(
            f"\nProcessing {filename}..."
        )

        # ----------------------------------------------------
        # CMEAS
        # ----------------------------------------------------

        cmeas = calculate_cmeas(
            model,
            dataset,
            image_index,
            device
        )

        # ----------------------------------------------------
        # DETECT REGIONS
        # ----------------------------------------------------

        (
            anomaly_mask,
            labeled_regions,
            region_count,
            threshold
        ) = detect_anomalous_regions(
            cmeas,
            threshold_percentile=90
        )

        # ----------------------------------------------------
        # RANK REGIONS
        # ----------------------------------------------------

        regions = rank_regions(
            cmeas,
            labeled_regions,
            region_count
        )

        # ----------------------------------------------------
        # LOAD ORIGINAL IMAGE
        # ----------------------------------------------------

        image = load_sar_image(
            filename
        )

        # ----------------------------------------------------
        # SAVE CMEAS MAP
        # ----------------------------------------------------

        heatmap_path = (
            save_cmeas_heatmap(
                cmeas,
                filename
            )
        )

        # ----------------------------------------------------
        # SAVE REGION OVERLAY
        # ----------------------------------------------------

        overlay_path = (
            save_region_overlay(
                image,
                regions,
                filename
            )
        )

        # ----------------------------------------------------
        # PRINT RESULTS
        # ----------------------------------------------------

        print(
            f"  Candidate regions: "
            f"{region_count}"
        )

        print(
            f"  Threshold: "
            f"{threshold:.4f}"
        )

        if regions:

            print(
                f"  Top region priority: "
                f"{regions[0]['priority_score']:.4f}"
            )

            print(
                f"  Top region patches: "
                f"{regions[0]['patch_count']}"
            )

        print(
            f"  CMEAS map saved: "
            f"{heatmap_path}"
        )

        print(
            f"  Region overlay saved: "
            f"{overlay_path}"
        )

    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "VISUALIZATION COMPLETED"
    )

    print(
        "=" * 70
    )