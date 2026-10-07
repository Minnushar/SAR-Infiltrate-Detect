import numpy as np
from scipy import ndimage


def detect_anomalous_regions(
    cmeas_grid,
    threshold_percentile=90
):
    """
    Detect candidate anomalous regions from
    a CMEAS score grid.

    Patches above the selected percentile
    are treated as candidate anomalies.
    """

    threshold = np.percentile(
        cmeas_grid,
        threshold_percentile
    )

    binary_mask = (
        cmeas_grid >= threshold
    )

    labeled_regions, region_count = ndimage.label(
        binary_mask
    )

    return (
        binary_mask,
        labeled_regions,
        region_count,
        threshold
    )


def rank_regions(
    cmeas_grid,
    labeled_regions,
    region_count
):
    """
    Calculate statistics for each candidate region
    and rank regions using a combined priority score.

    Priority considers:

        1. Mean CMEAS
        2. Maximum CMEAS
        3. Spatial extent

    The resulting score is an experimental
    prioritization score, not a probability.
    """

    regions = []

    # Maximum possible number of patches
    total_patches = cmeas_grid.size

    for region_id in range(
        1,
        region_count + 1
    ):

        # Find all pixels/patches belonging
        # to this region
        region_mask = (
            labeled_regions == region_id
        )

        positions = np.argwhere(
            region_mask
        )

        scores = cmeas_grid[
            region_mask
        ]

        # ----------------------------------------------------
        # REGION STATISTICS
        # ----------------------------------------------------

        patch_count = len(
            positions
        )

        mean_score = np.mean(
            scores
        )

        max_score = np.max(
            scores
        )

        # ----------------------------------------------------
        # NORMALIZED REGION EXTENT
        # ----------------------------------------------------

        extent_score = (
            patch_count
            / total_patches
        )

        # ----------------------------------------------------
        # REGION PRIORITY SCORE
        # ----------------------------------------------------
        #
        # 50% mean anomaly strength
        # 30% maximum anomaly strength
        # 20% spatial extent
        #
        # These are initial experimental weights.
        # They are NOT learned or claimed optimal.
        # ----------------------------------------------------

        priority_score = (
            0.5 * mean_score
            + 0.3 * max_score
            + 0.2 * extent_score
        )

        regions.append({

            "region_id":
                region_id,

            "patch_count":
                patch_count,

            "mean_score":
                float(mean_score),

            "max_score":
                float(max_score),

            "extent_score":
                float(extent_score),

            "priority_score":
                float(priority_score),

            "positions":
                positions.tolist()
        })

    # --------------------------------------------------------
    # SORT BY PRIORITY
    # --------------------------------------------------------

    regions.sort(
        key=lambda x: x["priority_score"],
        reverse=True
    )

    return regions


if __name__ == "__main__":

    # ========================================================
    # TEST WITH RANDOM DATA
    # ========================================================

    np.random.seed(42)

    test_grid = np.random.rand(
        8,
        8
    )

    (
        mask,
        labeled_regions,
        region_count,
        threshold
    ) = detect_anomalous_regions(
        test_grid
    )

    regions = rank_regions(
        test_grid,
        labeled_regions,
        region_count
    )

    print(
        "CMEAS grid shape:",
        test_grid.shape
    )

    print(
        "Detection threshold:",
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
        mask.astype(int)
    )

    print(
        "\nRanked regions:"
    )

    for rank, region in enumerate(
        regions,
        start=1
    ):

        print(
            f"{rank}. "
            f"Region {region['region_id']} | "
            f"Patches: {region['patch_count']} | "
            f"Mean: {region['mean_score']:.4f} | "
            f"Max: {region['max_score']:.4f} | "
            f"Extent: {region['extent_score']:.4f} | "
            f"Priority: {region['priority_score']:.4f}"
        )