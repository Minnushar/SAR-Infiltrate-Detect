import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import rasterio
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
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="SAR-Infiltrate Detect",
    page_icon="🛰️",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title(
    "🛰️ SAR-Infiltrate Detect"
)

st.subheader(
    "AI-Powered Border Surveillance & "
    "Sub-Surface Anomaly Mapping"
)

st.info(
    "The system identifies anomalous SAR signatures "
    "using spatial reconstruction, local context and "
    "dual-polarization evidence."
)


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_scores(scores):

    scores = np.asarray(
        scores,
        dtype=np.float32
    )

    return (
        scores - scores.min()
    ) / (
        scores.max()
        - scores.min()
        + 1e-8
    )


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_resource
def load_dataset():

    return SARPatchDataset()


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = SARAutoencoder().to(
        device
    )

    model.load_state_dict(
        torch.load(
            "models/sar_autoencoder.pth",
            map_location=device
        )
    )

    model.eval()

    return model, device


# ============================================================
# LOAD ORIGINAL IMAGE
# ============================================================

def load_original_image(
    filename
):

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

    start = image_index * 64
    end = start + 64

    spatial_scores = []
    polarization_scores = []

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

            polarization_scores.append(
                calculate_polarization_score(
                    dataset[i].numpy()
                )
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

    context_grid = calculate_context_score(
        spatial_grid
    )

    spatial_norm = normalize_scores(
        spatial_grid
    )

    context_norm = normalize_scores(
        context_grid
    )

    polarization_norm = normalize_scores(
        polarization_grid
    )

    cmeas = (
        0.5 * spatial_norm
        + 0.2 * context_norm
        + 0.3 * polarization_norm
    )

    return (
        spatial_grid,
        context_grid,
        polarization_grid,
        cmeas
    )


# ============================================================
# LOAD EVERYTHING
# ============================================================

try:

    dataset = load_dataset()

    model, device = load_model()

except Exception as e:

    st.error(
        f"Could not load project files: {e}"
    )

    st.stop()


filenames = dataset.sar_dataset.files


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "Analysis Settings"
)

selected_file = st.sidebar.selectbox(
    "Select SAR Image",
    filenames
)

threshold_percentile = st.sidebar.slider(
    "Detection Threshold Percentile",
    min_value=80,
    max_value=99,
    value=90
)


# ============================================================
# IMAGE INDEX
# ============================================================

image_index = filenames.index(
    selected_file
)


# ============================================================
# CALCULATE
# ============================================================

with st.spinner(
    "Running CMEAS analysis..."
):

    (
        spatial_grid,
        context_grid,
        polarization_grid,
        cmeas
    ) = calculate_cmeas(
        model,
        dataset,
        image_index,
        device
    )


# ============================================================
# REGION DETECTION
# ============================================================

(
    anomaly_mask,
    labeled_regions,
    region_count,
    threshold
) = detect_anomalous_regions(
    cmeas,
    threshold_percentile
)

regions = rank_regions(
    cmeas,
    labeled_regions,
    region_count
)


# ============================================================
# METRICS
# ============================================================

st.header(
    "Analysis Summary"
)

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Candidate Regions",
        region_count
    )

with col2:

    st.metric(
        "CMEAS Maximum",
        f"{cmeas.max():.3f}"
    )

with col3:

    st.metric(
        "Detection Threshold",
        f"{threshold:.3f}"
    )

with col4:

    if regions:

        st.metric(
            "Top Region Priority",
            f"{regions[0]['priority_score']:.3f}"
        )

    else:

        st.metric(
            "Top Region Priority",
            "N/A"
        )


# ============================================================
# ORIGINAL SAR + CMEAS
# ============================================================

st.header(
    "SAR Analysis"
)

original_image = load_original_image(
    selected_file
)

col1, col2 = st.columns(2)


# ------------------------------------------------------------
# ORIGINAL SAR
# ------------------------------------------------------------

with col1:

    st.subheader(
        "Original SAR Image"
    )

    fig, ax = plt.subplots(
        figsize=(6, 6)
    )

    ax.imshow(
        original_image,
        cmap="gray"
    )

    ax.set_title(
        selected_file
    )

    ax.axis(
        "off"
    )

    st.pyplot(
        fig
    )

    plt.close(fig)


# ------------------------------------------------------------
# CMEAS
# ------------------------------------------------------------

with col2:

    st.subheader(
        "CMEAS Anomaly Map"
    )

    fig, ax = plt.subplots(
        figsize=(6, 6)
    )

    im = ax.imshow(
        cmeas,
        cmap="hot",
        interpolation="nearest"
    )

    ax.set_title(
        "CMEAS Score"
    )

    ax.set_xlabel(
        "Patch Column"
    )

    ax.set_ylabel(
        "Patch Row"
    )

    fig.colorbar(
        im,
        ax=ax,
        label="CMEAS"
    )

    st.pyplot(
        fig
    )

    plt.close(fig)


# ============================================================
# REGION OVERLAY
# ============================================================

st.header(
    "Candidate Region Detection"
)

fig, ax = plt.subplots(
    figsize=(8, 8)
)

ax.imshow(
    original_image,
    cmap="gray"
)

for rank, region in enumerate(
    regions,
    start=1
):

    positions = region[
        "positions"
    ]

    rows = [
        p[0]
        for p in positions
    ]

    cols = [
        p[1]
        for p in positions
    ]

    min_row = min(rows)
    max_row = max(rows)

    min_col = min(cols)
    max_col = max(cols)

    xmin = min_col * 64
    ymin = min_row * 64

    width = (
        max_col - min_col + 1
    ) * 64

    height = (
        max_row - min_row + 1
    ) * 64

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

    ax.add_patch(
        rectangle
    )

    ax.text(
        xmin + 3,
        ymin + 15,
        f"R{rank}",
        fontsize=10,
        backgroundcolor="white"
    )


ax.set_title(
    "Detected Candidate Regions"
)

ax.axis(
    "off"
)

st.pyplot(
    fig
)

plt.close(fig)


# ============================================================
# REGION TABLE
# ============================================================

st.header(
    "Ranked Candidate Regions"
)

if regions:

    table_data = []

    for rank, region in enumerate(
        regions,
        start=1
    ):

        table_data.append({

            "Rank":
                rank,

            "Region":
                f"R{rank}",

            "Patches":
                region["patch_count"],

            "Mean CMEAS":
                round(
                    region["mean_score"],
                    4
                ),

            "Maximum CMEAS":
                round(
                    region["max_score"],
                    4
                ),

            "Priority Score":
                round(
                    region["priority_score"],
                    4
                )
        })

    st.dataframe(
        pd.DataFrame(
            table_data
        ),
        use_container_width=True
    )

else:

    st.warning(
        "No candidate regions detected."
    )


# ============================================================
# EXPLANATION
# ============================================================

st.header(
    "How CMEAS Works"
)

st.markdown(
    """
**CMEAS (Context-Aware Multi-Evidence Anomaly Scoring)**
combines three evidence sources:

1. **Spatial anomaly** — reconstruction error from the CNN autoencoder.
2. **Local context** — how unusual a patch is compared with neighbouring patches.
3. **Polarization evidence** — difference between VV and VH SAR responses.

The current experimental weighting is:

- Spatial: **50%**
- Context: **20%**
- Polarization: **30%**

Higher CMEAS values indicate stronger anomalous SAR signatures.
They should be interpreted as **candidate anomalies**, not definitive
identification of infiltrators or underground structures.
"""
)


# ============================================================
# FOOTER
# ============================================================

st.caption(
    "SAR-Infiltrate Detect | Software-based research prototype"
)