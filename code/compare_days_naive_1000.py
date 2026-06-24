#!/usr/bin/env python3
"""
Cross-day comparison of naive 1000-scenario model synergy ratios.

Usage:
    python compare_days_naive_1000.py --tol 0.05
    python compare_days_naive_1000.py --tol 0.01

Outputs:
  results/naive_1000_comparison_all_days_tol{TAG}.csv
  results/naive_1000_comparison_heatmap_tol{TAG}.png
  results/naive_1000_comparison_lineplot_tol{TAG}.png
"""

import argparse
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument("--tol", type=float, default=0.05, choices=[0.01, 0.05, 0.10])
args = parser.parse_args()
TOL = args.tol
tol_tag = f"{TOL:.2f}".replace("0.", "").replace(".", "")

RESULTS = "../results"
DAYS    = [271, 301, 332, 362]

# ── 1. Load and merge summary CSVs ────────────────────────────────────────────

frames = []
for day in DAYS:
    df = pd.read_csv(f"{RESULTS}/naive_1000_summary_day{day}_tol{tol_tag}.csv")
    df["day"] = day
    frames.append(df)

df_all = pd.concat(frames, ignore_index=True)

pivot = df_all.pivot(index="country", columns="day", values="ratio")
pivot.columns = [f"Day {d}" for d in pivot.columns]
pivot.index.name = "Country"
pivot["_mean"] = pivot.mean(axis=1)
pivot = pivot.sort_values("_mean", ascending=False).drop(columns="_mean")

out_csv = f"{RESULTS}/naive_1000_comparison_all_days_tol{tol_tag}.csv"
df_all.to_csv(out_csv, index=False)
print(f"Saved: {out_csv}")
print(f"\nFull pivot table:\n{pivot.round(3).to_string()}")

# ── 2. Annotated heatmap ──────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(9, 11))

annot = pivot.copy().astype(object)
for col in annot.columns:
    annot[col] = annot[col].apply(lambda v: f"{v:.2f}" if pd.notna(v) else "—")

vmax = float(np.nanpercentile(pivot.values.astype(float), 95))

sns.heatmap(
    pivot.astype(float),
    annot=annot, fmt="",
    cmap="YlGn",
    vmin=1.0, vmax=vmax,
    linewidths=0.4, linecolor="white",
    ax=ax,
    cbar_kws={"label": "Synergy Ratio (1000 scenarios)", "shrink": 0.6},
    mask=pivot.isna(),
)

for i, country in enumerate(pivot.index):
    for j, col in enumerate(pivot.columns):
        if pd.isna(pivot.loc[country, col]):
            ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=True, color="#CCCCCC", zorder=0))
            ax.text(j+0.5, i+0.5, "—", ha="center", va="center",
                    fontsize=10, color="#555555")

ax.set_title(
    f"Naive Model (1000 scenarios) — Synergy Ratio by Country and Rolling Window\n(tol={TOL})",
    fontsize=13, pad=12
)
ax.set_xlabel("Rolling Window", fontsize=12)
ax.set_ylabel("")
ax.tick_params(axis="x", labelsize=11)
ax.tick_params(axis="y", labelsize=10)

plt.tight_layout()
hm_path = f"{RESULTS}/naive_1000_comparison_heatmap_tol{tol_tag}.png"
plt.savefig(hm_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {hm_path}")

# ── 3. Line plot ──────────────────────────────────────────────────────────────

complete = pivot.dropna()
print(f"\nCountries in all 4 days ({len(complete)}): {sorted(complete.index.tolist())}")

country_name_map = {
    "AT": "Austria",     "BE": "Belgium",      "BG": "Bulgaria",
    "CH": "Switzerland", "CZ": "Czech Rep.",   "DE": "Germany",
    "DK": "Denmark",     "EE": "Estonia",      "EL": "Greece",
    "ES": "Spain",       "FI": "Finland",      "FR": "France",
    "HR": "Croatia",     "HU": "Hungary",      "IE": "Ireland",
    "IT": "Italy",       "LT": "Lithuania",    "LU": "Luxembourg",
    "LV": "Latvia",      "NL": "Netherlands",  "NO": "Norway",
    "PL": "Poland",      "PT": "Portugal",     "RO": "Romania",
    "SE": "Sweden",      "SI": "Slovenia",     "SK": "Slovakia",
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

day_labels = [f"Day {d}" for d in DAYS]
x = np.arange(len(DAYS))

fig, ax = plt.subplots(figsize=(11, 7))
for country in complete.index:
    y_vals = complete.loc[country, day_labels].values.astype(float)
    region = country_to_region.get(country, "Other")
    color  = region_colors[region]
    ax.plot(x, y_vals, marker="o", color=color, linewidth=1.4,
            markersize=5, alpha=0.85)
    ax.text(x[-1] + 0.05, y_vals[-1], country_name_map.get(country, country),
            fontsize=8, va="center", color=color)

ax.axhline(1.0, color="red", linestyle="--", linewidth=1, label="Ratio = 1")
ax.set_xticks(x)
ax.set_xticklabels(day_labels, fontsize=12)
ax.set_ylabel("Synergy Ratio (1000 scenarios)", fontsize=12)
ax.set_xlabel("Rolling Window", fontsize=12)
ax.set_title(
    f"Naive Model (1000 scenarios) — Synergy Ratio across Rolling Windows\n(tol={TOL})",
    fontsize=13
)
ax.set_xlim(-0.15, len(DAYS) - 1 + 2.0)

legend_handles = [mpatches.Patch(color=c, label=r)
                  for r, c in region_colors.items() if r != "Other"]
legend_handles.append(plt.Line2D([0], [0], color="red", linestyle="--", label="Ratio = 1"))
ax.legend(handles=legend_handles, fontsize=9, loc="upper left")

sns.set_theme(style="whitegrid")
ax.grid(True, alpha=0.3)
plt.tight_layout()
lp_path = f"{RESULTS}/naive_1000_comparison_lineplot_tol{tol_tag}.png"
plt.savefig(lp_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {lp_path}")

# ── 4. Summary ────────────────────────────────────────────────────────────────

print(f"\n-- Summary: range of ratios across days (naive 1000-scenario, tol={TOL}) --")
summary = pd.DataFrame({
    "min":   pivot.min(axis=1),
    "max":   pivot.max(axis=1),
    "range": pivot.max(axis=1) - pivot.min(axis=1),
    "mean":  pivot.mean(axis=1),
    "days_available": pivot.notna().sum(axis=1),
}).sort_values("range", ascending=False)
print(summary.round(3).to_string())
