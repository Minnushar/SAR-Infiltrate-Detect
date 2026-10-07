import os
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# LOAD EVALUATION RESULTS
# ============================================================

input_path = (
    "outputs/metrics/"
    "cmeas_evaluation.csv"
)

df = pd.read_csv(input_path)


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    "outputs/figures",
    exist_ok=True
)


# ============================================================
# PLOT TOP PATCH COMPARISON
# ============================================================

plt.figure(figsize=(12, 6))

x = range(len(df))

plt.plot(
    x,
    df["spatial_top_patch"],
    marker="o",
    label="Spatial Only"
)

plt.plot(
    x,
    df["cmeas_top_patch"],
    marker="s",
    label="CMEAS"
)

plt.xticks(
    x,
    df["image"],
    rotation=45
)

plt.xlabel(
    "SAR Image"
)

plt.ylabel(
    "Top-Ranked Patch"
)

plt.title(
    "Spatial Baseline vs CMEAS Top-Ranked Anomaly Patch"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()


# ============================================================
# SAVE FIGURE
# ============================================================

output_path = (
    "outputs/figures/"
    "spatial_vs_cmeas.png"
)

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print(
    "Figure saved successfully:"
)

print(
    output_path
)