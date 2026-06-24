"""
Compare mean-of-ratios (naive_CI_calculation.py) against ratio-of-means
(naive_ratio_of_means.py) for the Naive model.

Two-panel horizontal dot plot, one panel per tolerance level.
Countries are sorted by ratio-of-means at tol=0.05 (ascending from bottom).
Error bars show the 95% bootstrap CI for ratio-of-means.
Values exceeding the x-axis cap are annotated in-plot.

Output: results/comparison_mor_vs_rom.png
"""

import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

NAIVE_RESULTS_DIR = "../naive_results"
ROM_FILE = "../results/ratio_of_means_naive.csv"
OUT_FIG = "../results/comparison_mor_vs_rom.png"

COLOR_ROM = "#1E1E1E"   # ratio-of-means
COLOR_MOR = "#D4623A"   # mean-of-ratios
MS = 55                 # marker size (scatter s=)

# x-axis caps per panel (out-of-range values are annotated)
CAPS = {0.05: 7.0, 0.01: 20.0}

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
files = sorted(glob.glob(f"{NAIVE_RESULTS_DIR}/naive_results_*.csv"))
raw = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
raw.columns = raw.columns.str.strip()
for col in ["wind objective", "solar objective", "combined objective"]:
    raw[col] = raw[col].clip(lower=0)

mor = (
    raw.groupby(["country", "tol"])["ratio"]
    .mean()
    .reset_index()
    .rename(columns={"ratio": "mean_of_ratios"})
)

rom = pd.read_csv(ROM_FILE)
df = mor.merge(
    rom[["country", "tol", "ratio_of_means", "ci_lower_95", "ci_upper_95", "near_zero_denom"]],
    on=["country", "tol"],
)

# Country order: ascending ratio-of-means at tol=0.05 (bottom = lowest synergy)
order = (
    df[df["tol"] == 0.05]
    .sort_values("ratio_of_means", ascending=True)["country"]
    .tolist()
)
n = len(order)
y_pos = np.arange(n)

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(16, 11), sharey=True)
fig.subplots_adjust(wspace=0.04)

for ax, tol in zip(axes, [0.05, 0.01]):
    cap = CAPS[tol]
    sub = df[df["tol"] == tol].set_index("country").loc[order].reset_index()

    for yi, (_, row) in zip(y_pos, sub.iterrows()):
        mor_plot = min(row["mean_of_ratios"], cap)
        rom_plot = min(row["ratio_of_means"], cap)

        # thin connector between the two estimates
        ax.plot([mor_plot, rom_plot], [yi, yi],
                color="gray", lw=0.9, alpha=0.45, zorder=1)

        # 95% CI bar for ratio-of-means
        lo = row["ci_lower_95"] if not np.isnan(row["ci_lower_95"]) else row["ratio_of_means"]
        hi = row["ci_upper_95"] if not np.isnan(row["ci_upper_95"]) else row["ratio_of_means"]
        ax.plot([min(lo, cap), min(hi, cap)], [yi, yi],
                color=COLOR_ROM, lw=3, alpha=0.35, zorder=2)

    # mean-of-ratios markers
    for yi, (_, row) in zip(y_pos, sub.iterrows()):
        val = row["mean_of_ratios"]
        ax.scatter(min(val, cap), yi, color=COLOR_MOR, marker="D",
                   s=MS, zorder=4, label="Mean-of-ratios" if yi == 0 else "")
        if val > cap:
            ax.annotate(
                f"{val:.1f}",
                xy=(cap, yi), xytext=(cap - 0.25, yi + 0.38),
                fontsize=8.5, color=COLOR_MOR, ha="right",
            )

    # ratio-of-means markers
    for yi, (_, row) in zip(y_pos, sub.iterrows()):
        val = row["ratio_of_means"]
        marker = "^" if row["near_zero_denom"] else "o"
        ax.scatter(min(val, cap), yi, color=COLOR_ROM, marker=marker,
                   s=MS, zorder=5, label="Ratio-of-means" if yi == 0 else "")
        if val > cap:
            ax.annotate(
                f"{val:.1f}",
                xy=(cap, yi), xytext=(cap - 0.25, yi - 0.38),
                fontsize=8.5, color=COLOR_ROM, ha="right",
            )

    ax.axvline(1.0, color="black", lw=0.9, linestyle="--", alpha=0.45)
    ax.set_xlim(0.5, cap + 0.6)
    ax.set_title(f"Tolerance = {tol}", fontsize=15, pad=8)
    ax.set_xlabel("Synergy ratio", fontsize=13)
    ax.tick_params(axis="x", labelsize=12)

    if tol == 0.05:
        handles = [
            plt.scatter([], [], color=COLOR_MOR, marker="D", s=MS, label="Mean-of-ratios"),
            plt.scatter([], [], color=COLOR_ROM, marker="o", s=MS, label="Ratio-of-means"),
            plt.Line2D([0], [0], color=COLOR_ROM, lw=3, alpha=0.35, label="95% bootstrap CI"),
            plt.scatter([], [], color=COLOR_ROM, marker="^", s=MS, label="Near-zero denom flag"),
        ]
        ax.legend(handles=handles, fontsize=11, loc="lower right", framealpha=0.85)

# y-axis labels (left panel only, sharey=True)
axes[0].set_yticks(y_pos)
axes[0].set_yticklabels(order, fontsize=11)
axes[0].set_ylabel("Country", fontsize=13)

# note about x-axis caps
cap_note = (
    f"Note: x-axis capped at {CAPS[0.05]:.0f} (left) and {CAPS[0.01]:.0f} (right);"
    " out-of-range values annotated."
)
fig.text(0.5, -0.01, cap_note, ha="center", fontsize=10, color="gray")

fig.suptitle(
    "Mean-of-ratios vs Ratio-of-means  |  Naive model",
    fontsize=16, y=1.01,
)
plt.tight_layout()
plt.savefig(OUT_FIG, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved {OUT_FIG}")
