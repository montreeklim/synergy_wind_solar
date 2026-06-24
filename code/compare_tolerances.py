#!/usr/bin/env python3
"""
Cross-tolerance comparison: tol=0.01 vs tol=0.05 battery results (100 MW).

Usage:
    python compare_tolerances.py

Outputs:
  results/tol_comparison_scatter.png   — tol05 vs tol01 scatter per country×day
  results/tol_comparison_diff_heatmap.png — heatmap of (tol01 - tol05) differences
  results/tol_comparison_summary.csv   — per-country mean difference and range
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

RESULTS = "../results"
DAYS = [271, 301, 332, 362]

# ── 1. Load both tolerances ───────────────────────────────────────────────────

df01 = pd.read_csv(f"{RESULTS}/comparison_all_days_tol01_100.csv")
df05 = pd.read_csv(f"{RESULTS}/comparison_all_days_tol05_100.csv")

pivot01 = df01.pivot(index="country", columns="day", values="mean_ratio")
pivot05 = df05.pivot(index="country", columns="day", values="mean_ratio")
pivot01.columns = [f"Day {d}" for d in pivot01.columns]
pivot05.columns = [f"Day {d}" for d in pivot05.columns]

# Difference: tol01 minus tol05
diff = pivot01 - pivot05

country_name_map = {
    "AT": "Austria",    "BE": "Belgium",   "BG": "Bulgaria",
    "CH": "Switzerland","CZ": "Czech Rep.", "DE": "Germany",
    "DK": "Denmark",    "EE": "Estonia",   "EL": "Greece",
    "ES": "Spain",      "FI": "Finland",   "FR": "France",
    "HR": "Croatia",    "HU": "Hungary",   "IE": "Ireland",
    "IT": "Italy",      "LT": "Lithuania", "LU": "Luxembourg",
    "LV": "Latvia",     "NL": "Netherlands","NO": "Norway",
    "PL": "Poland",     "PT": "Portugal",  "RO": "Romania",
    "SE": "Sweden",     "SI": "Slovenia",  "SK": "Slovakia",
    "UK": "United Kingdom",
}

regions = {
    "Atlantic Maritime": ["IE", "UK", "FR", "BE", "NL"],
    "Continental":       ["DE", "PL", "CZ", "HU", "AT", "CH", "SK", "LU"],
    "Mediterranean":     ["ES", "PT", "IT", "EL", "HR", "SI", "BG", "RO"],
    "Nordic & Baltic":   ["NO", "SE", "FI", "DK", "EE", "LV", "LT"],
}
country_to_region = {c: r for r, cs in regions.items() for c in cs}

region_colors = {
    "Atlantic Maritime": "#2C7BB6",
    "Continental":       "#74ADD1",
    "Mediterranean":     "#F46D43",
    "Nordic & Baltic":   "#78C679",
    "Other":             "#999999",
}

# ── 2. Summary table ──────────────────────────────────────────────────────────

summary = pd.DataFrame({
    "mean_tol05":    pivot05.mean(axis=1),
    "mean_tol01":    pivot01.mean(axis=1),
    "mean_diff":     diff.mean(axis=1),
    "max_abs_diff":  diff.abs().max(axis=1),
    "days_tol01_higher": (diff > 0).sum(axis=1),
}).sort_values("mean_diff", ascending=False)

summary.index.name = "country"
print("Cross-tolerance summary (tol01 - tol05):")
print(summary.round(4).to_string())
summary.to_csv(f"{RESULTS}/tol_comparison_summary.csv")
print(f"\nSaved: {RESULTS}/tol_comparison_summary.csv")

# ── 3. Scatter: tol05 vs tol01 mean ratio ────────────────────────────────────

day_labels = [f"Day {d}" for d in DAYS]
markers = ["o", "s", "^", "D"]

fig, ax = plt.subplots(figsize=(8, 7))

all_vals = pd.concat([pivot01.stack(), pivot05.stack()]).dropna()
lim_lo = max(0.99, all_vals.min() - 0.01)
lim_hi = all_vals.max() + 0.02

ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "k--", linewidth=0.8, alpha=0.5, label="tol01 = tol05")

for i, day_col in enumerate(day_labels):
    x = pivot05[day_col].dropna()
    y = pivot01[day_col].dropna()
    common = x.index.intersection(y.index)
    x, y = x[common], y[common]

    for country in common:
        region = country_to_region.get(country, "Other")
        color  = region_colors[region]
        ax.scatter(x[country], y[country],
                   color=color, marker=markers[i], s=45,
                   edgecolors="none", alpha=0.85)
        if abs(y[country] - x[country]) > 0.04:
            ax.annotate(country,
                        xy=(x[country], y[country]),
                        xytext=(3, 3), textcoords="offset points",
                        fontsize=7, color=color)

# Legend: regions
region_handles = [
    mpatches.Patch(color=c, label=r)
    for r, c in region_colors.items() if r != "Other"
]
# Legend: day markers
day_handles = [
    plt.Line2D([0], [0], marker=markers[i], color="grey",
               linestyle="", markersize=7, label=day_labels[i])
    for i in range(len(day_labels))
]
ax.legend(handles=region_handles + day_handles, fontsize=8, loc="upper left", ncol=2)

ax.set_xlabel("Mean Synergy Ratio (tol = 0.05)", fontsize=12)
ax.set_ylabel("Mean Synergy Ratio (tol = 0.01)", fontsize=12)
ax.set_title("Synergy Ratio: tol=0.01 vs tol=0.05\n(100 MW, all 28 countries, 4 rolling windows)", fontsize=12)
ax.set_xlim(lim_lo, lim_hi)
ax.set_ylim(lim_lo, lim_hi)
ax.grid(True, alpha=0.3)
plt.tight_layout()

scatter_path = f"{RESULTS}/tol_comparison_scatter.png"
plt.savefig(scatter_path, dpi=300)
plt.close()
print(f"Saved: {scatter_path}")

# ── 4. Heatmap: tol01 − tol05 difference ─────────────────────────────────────

# Sort by absolute mean difference descending
diff_sorted = diff.loc[summary.index]
diff_sorted.index = [country_name_map.get(c, c) for c in diff_sorted.index]

annot = diff_sorted.copy().astype(object)
for col in annot.columns:
    annot[col] = annot[col].apply(
        lambda v: f"{v:+.3f}" if pd.notna(v) else "—"
    )

vabs = diff_sorted.abs().max().max()
fig, ax = plt.subplots(figsize=(9, 11))
sns.heatmap(
    diff_sorted.astype(float),
    annot=annot, fmt="",
    cmap="RdBu_r",
    center=0, vmin=-vabs, vmax=vabs,
    linewidths=0.4, linecolor="white",
    ax=ax,
    cbar_kws={"label": "Δ Synergy Ratio (tol=0.01 − tol=0.05)", "shrink": 0.6},
)
ax.set_title(
    "Difference in Mean Synergy Ratio: tol=0.01 − tol=0.05\n"
    "(positive = tol=0.01 gives higher synergy)",
    fontsize=13, pad=12
)
ax.set_xlabel("Rolling Window", fontsize=12)
ax.set_ylabel("")
ax.tick_params(axis="x", labelsize=11)
ax.tick_params(axis="y", labelsize=9)
plt.tight_layout()

diff_hm_path = f"{RESULTS}/tol_comparison_diff_heatmap.png"
plt.savefig(diff_hm_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {diff_hm_path}")

# ── 5. Boxplot of differences by region ──────────────────────────────────────

diff_long = diff.stack().reset_index()
diff_long.columns = ["country", "day", "diff"]
diff_long["region"] = diff_long["country"].map(lambda c: country_to_region.get(c, "Other"))
diff_long = diff_long[diff_long["region"] != "Other"]

region_order = ["Continental", "Atlantic Maritime", "Mediterranean", "Nordic & Baltic"]
palette = ["#AEC6CF", "#2C7BB6", "#FDB863", "#78C679"]

fig, ax = plt.subplots(figsize=(9, 6))
sns.boxplot(data=diff_long, x="region", y="diff",
            order=region_order, palette=palette, ax=ax)
ax.axhline(0, color="red", linestyle="--", linewidth=1, label="No difference")
ax.set_xlabel("Region", fontsize=13)
ax.set_ylabel("Δ Synergy Ratio (tol=0.01 − tol=0.05)", fontsize=12)
ax.set_title("Tolerance Sensitivity by Region\n(battery model, 100 MW installed capacity)", fontsize=13)
ax.legend(fontsize=11)
plt.xticks(rotation=20, ha="right", fontsize=12)
plt.tight_layout()

region_path = f"{RESULTS}/tol_comparison_region_boxplot.png"
plt.savefig(region_path, dpi=300)
plt.close()
print(f"Saved: {region_path}")

print("\n=== Cross-tolerance comparison complete ===")
