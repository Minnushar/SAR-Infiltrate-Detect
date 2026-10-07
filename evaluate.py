import os
import numpy as np
import pandas as pd
import torch

from src.dataset import SARPatchDataset
from src.autoencoder import SARAutoencoder
from src.context_score import calculate_context_score
from src.polarization_score import calculate_polarization_score

from src.region_detection import (
    detect_anomalous_regions,
    rank_regions
)


# ============================================================
# SCORE NORMALIZATION
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
# SPATIAL ANOMALY SCORE
# ============================================================

def calculate_spatial_scores(
    model,
    dataset,
    device
):
    """
    Calculate CNN autoencoder reconstruction error
    for every SAR patch.
    """

    model.eval()

    scores = []

    with torch.no_grad():

        for i in range(len(dataset)):

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

            scores.append(error)

    return np.array(
        scores,
        dtype=np.float32
    )


# ============================================================
# POLARIZATION SCORE
# ============================================================

def calculate_polarization_scores(
    dataset
):
    """
    Calculate VV/VH polarization difference
    for every SAR patch.
    """

    scores = []

    for i in range(len(dataset)):

        patch = (
            dataset[i]
            .numpy()
        )

        score = calculate_polarization_score(
            patch
        )

        scores.append(score)

    return np.array(
        scores,
        dtype=np.float32
    )


# ============================================================
# CMEAS CALCULATION
# ============================================================

def calculate_cmeas(
    spatial_scores,
    polarization_scores,
    image_index
):
    """
    Calculate CMEAS for one image.

    Current experimental weights:

        Spatial       = 50%
        Context       = 20%
        Polarization = 30%

    Temporal evidence is currently omitted because
    genuine temporal correspondence has not yet been
    established in the available dataset.
    """

    start = image_index * 64
    end = start + 64

    spatial = spatial_scores[
        start:end
    ]

    polarization = polarization_scores[
        start:end
    ]

    spatial_grid = spatial.reshape(
        8,
        8
    )

    polarization_grid = (
        polarization.reshape(
            8,
            8
        )
    )

    # --------------------------------------------------------
    # LOCAL CONTEXT
    # --------------------------------------------------------

    context_grid = calculate_context_score(
        spatial_grid
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

    return (
        spatial_norm,
        context_norm,
        polarization_norm,
        cmeas
    )


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

    image_files = (
        dataset.sar_dataset.files
    )

    print(
        "\nImages available:",
        len(image_files)
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
    # SPATIAL SCORES
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # POLARIZATION SCORES
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    results = []

    print(
        "\n"
        + "=" * 70
    )

    print(
        "MULTI-IMAGE CMEAS EVALUATION"
    )

    print(
        "=" * 70
    )

    # ========================================================
    # PROCESS EVERY IMAGE
    # ========================================================

    for image_index, filename in enumerate(
        image_files
    ):

        (
            spatial_norm,
            context_norm,
            polarization_norm,
            cmeas
        ) = calculate_cmeas(
            spatial_scores,
            polarization_scores,
            image_index
        )

        # ----------------------------------------------------
        # REGION DETECTION
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
        # REGION RANKING
        # ----------------------------------------------------

        regions = rank_regions(
            cmeas,
            labeled_regions,
            region_count
        )

        # ----------------------------------------------------
        # TOP PATCH
        # ----------------------------------------------------

        spatial_top_patch = int(
            np.argmax(
                spatial_norm
            )
        )

        cmeas_top_patch = int(
            np.argmax(
                cmeas
            )
        )

        # ----------------------------------------------------
        # LARGEST REGION
        # ----------------------------------------------------

        if regions:

            largest_region = max(
                regions,
                key=lambda x:
                    x["patch_count"]
            )

            largest_region_patches = (
                largest_region[
                    "patch_count"
                ]
            )

            # Highest priority region
            top_region = regions[0]

            top_region_priority = (
                top_region[
                    "priority_score"
                ]
            )

            top_region_mean = (
                top_region[
                    "mean_score"
                ]
            )

            top_region_max = (
                top_region[
                    "max_score"
                ]
            )

            top_region_patches = (
                top_region[
                    "patch_count"
                ]
            )

        else:

            largest_region_patches = 0

            top_region_priority = 0.0

            top_region_mean = 0.0

            top_region_max = 0.0

            top_region_patches = 0

        # ----------------------------------------------------
        # TOP 5 CMEAS PATCHES
        # ----------------------------------------------------

        flat_cmeas = (
            cmeas.flatten()
        )

        top5 = np.argsort(
            flat_cmeas
        )[-5:][::-1]

        top5_string = ",".join(
            str(int(x))
            for x in top5
        )

        # ----------------------------------------------------
        # PRINT IMAGE RESULTS
        # ----------------------------------------------------

        print(
            f"\n{filename}"
        )

        print(
            f"  Spatial mean: "
            f"{spatial_norm.mean():.4f}"
        )

        print(
            f"  Context mean: "
            f"{context_norm.mean():.4f}"
        )

        print(
            f"  Polarization mean: "
            f"{polarization_norm.mean():.4f}"
        )

        print(
            f"  CMEAS mean: "
            f"{cmeas.mean():.4f}"
        )

        print(
            f"  Spatial max: "
            f"{spatial_norm.max():.4f}"
        )

        print(
            f"  CMEAS max: "
            f"{cmeas.max():.4f}"
        )

        print(
            f"  Spatial top patch: "
            f"{spatial_top_patch}"
        )

        print(
            f"  CMEAS top patch: "
            f"{cmeas_top_patch}"
        )

        print(
            f"  CMEAS threshold: "
            f"{threshold:.4f}"
        )

        print(
            f"  Candidate regions: "
            f"{region_count}"
        )

        print(
            f"  Largest region size: "
            f"{largest_region_patches} patch(es)"
        )

        if regions:

            print(
                "\n  Ranked candidate regions:"
            )

            for rank, region in enumerate(
                regions,
                start=1
            ):

                print(
                    f"    {rank}. "
                    f"Region {region['region_id']} | "
                    f"Patches: "
                    f"{region['patch_count']} | "
                    f"Mean: "
                    f"{region['mean_score']:.4f} | "
                    f"Max: "
                    f"{region['max_score']:.4f} | "
                    f"Priority: "
                    f"{region['priority_score']:.4f}"
                )

        # ----------------------------------------------------
        # STORE RESULTS
        # ----------------------------------------------------

        results.append({

            "image":
                filename,

            "spatial_mean":
                float(
                    spatial_norm.mean()
                ),

            "context_mean":
                float(
                    context_norm.mean()
                ),

            "polarization_mean":
                float(
                    polarization_norm.mean()
                ),

            "cmeas_mean":
                float(
                    cmeas.mean()
                ),

            "spatial_max":
                float(
                    spatial_norm.max()
                ),

            "cmeas_max":
                float(
                    cmeas.max()
                ),

            "spatial_std":
                float(
                    spatial_norm.std()
                ),

            "cmeas_std":
                float(
                    cmeas.std()
                ),

            "spatial_top_patch":
                spatial_top_patch,

            "cmeas_top_patch":
                cmeas_top_patch,

            "cmeas_top_5_patches":
                top5_string,

            "candidate_regions":
                region_count,

            "cmeas_threshold":
                float(
                    threshold
                ),

            "largest_region_patches":
                largest_region_patches,

            "top_region_priority":
                float(
                    top_region_priority
                ),

            "top_region_mean":
                float(
                    top_region_mean
                ),

            "top_region_max":
                float(
                    top_region_max
                ),

            "top_region_patches":
                int(
                    top_region_patches
                )
        })

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    os.makedirs(
        "outputs/metrics",
        exist_ok=True
    )

    results_df = pd.DataFrame(
        results
    )

    output_path = (
        "outputs/metrics/"
        "cmeas_evaluation.csv"
    )

    results_df.to_csv(
        output_path,
        index=False
    )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "Evaluation completed successfully."
    )

    print(
        "=" * 70
    )

    print(
        "\nResults saved to:"
    )

    print(
        output_path
    )

    print(
        "\nEvaluation table:"
    )

    print(
        results_df.to_string(
            index=False
        )
    )