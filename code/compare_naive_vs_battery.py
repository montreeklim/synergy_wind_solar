#!/usr/bin/env python3
"""
Compare naive model vs battery (1x) synergy ratios across all rolling windows.

Outputs:
  results/comparison_naive_vs_battery_boxplot.png   — grouped regional boxplot (tol=0.05 and 0.10)
  results/comparison_naive_vs_battery_scatter.png   — country scatter (cross-date means)
  results/comparison_naive_vs_battery_summary.csv   — cross-date mean SR table
"""

import os, glob, warnings
warnings.filterwarnings("ignore")

import sys
_conda_root = os.path.dirname(sys.executable)
if sys.platform == "win32":
    os.environ.setdefault("GDAL_DATA",  os.path.join(_conda_root, "Library", "share", "gdal"))
    os.environ.setdefault("PROJ_DATA",  os.path.join(_conda_root, "Library", "share", "proj"))
    os.environ.setdefault("PROJ_NETWORK", "OFF")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import scipy.stats as st

RESULTS = "../results"

# ── Country / region mappings ─────────────────────────────────────────────────

country_name_map = {
    "AT": "Austria",       "BE": "Belgium",        "BG": "Bulgaria",
    "CH": "Switzerland",   "CZ": "Czech Republic", "DE": "Germany",
    "DK": "Denmark",       "EE": "Estonia",        "ES": "Spain",
    "FI": "Finland",       "FR": "France",         "EL": "Greece",
    "HR": "Croatia",       "HU": "Hungary",        "IE": "Ireland",
    "IT": "Italy",         "LT": "Lithuania",      "LU": "Luxembourg",
    "LV": "Latvia",        "NL": "Netherlands",    "NO": "Norway",
    "PL": "Poland",        "PT": "Portugal",       "RO": "Romania",
    "SI": "Slovenia",      "SK": "Slovakia",       "SE": "Sweden",
    "UK": "United Kingdom",
}
regions = {
    "Atlantic Maritime": ["Ireland", "United Kingdom", "France", "Belgium", "Netherlands"],
    "Continental":       ["Germany", "Poland", "Czech Republic", "Austria", "Hungary",
                          "Switzerland", "Luxembourg", "Slovakia"],
    "Mediterranean":     ["Spain", "Portugal", "Italy", "Greece", "Croatia",
                          "Slovenia", "Bulgaria", "Romania"],
    "Nordic & Baltic":   ["Norway", "Sweden", "Finland", "Denmark", "Estonia",
                          "Latvia", "Lithuania"],
}
iso2_to_name   = country_name_map
name_to_region = {c: r for r, cs in regions.items() for c in cs}
iso2_to_region = {k: name_to_region.get(v) for k, v in iso2_to_name.items()}

region_order   = ["Continental", "Atlantic Maritime", "Mediterranean", "Nordic & Baltic"]
region_palette = {
    "Continental":       "#AEC6CF",
    "Atlantic Maritime": "#2C7BB6",
    "Mediterranean":     "#FDB863",
    "Nordic & Baltic":   "#78C679",
}

DAYS = [271, 301, 332, 362]

# ── 1. Load raw scenario-level ratios ─────────────────────────────────────────

def load_battery_raw(tol):
    tol_str = f"{tol:.2f}"
    files = glob.glob(f"../battery_results/100_installed_capacity/"
                      f"battery_*_day_*_tol_{tol_str}_100_installed_capacity.csv")
    records = []
    for f in files:
        parts = os.path.basename(f).split("_")
        country = parts[1]
        day     = int(parts[3])
        df      = pd.read_csv(f)[["set_number", "ratio"]]
        df["country"] = country
        df["day"]     = day
        records.append(df)
    df_all = pd.concat(records, ignore_index=True)
    df_all["model"] = f"Battery (tol={tol})"
    df_all["tol"]   = tol
    return df_all

def load_naive_raw(tol):
    files = glob.glob("../naive_results/naive_results_*.csv")
    df_all = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df_all[np.isclose(df_all["tol"], tol)][["country", "forecast_day", "set_number", "ratio"]].copy()
    df = df.rename(columns={"forecast_day": "day"})
    df["model"] = f"Naive (tol={tol})"
    df["tol"]   = tol
    return df

bat05 = load_battery_raw(0.05)
bat01 = load_battery_raw(0.01)
nav05 = load_naive_raw(0.05)
nav01 = load_naive_raw(0.01)
nav10 = load_naive_raw(0.10)

def add_region(df):
    df = df.copy()
    df["country_name"] = df["country"].map(iso2_to_name)
    df["Region"]       = df["country_name"].map(name_to_region)
    return df.dropna(subset=["Region"])

# ── 2. Summary table (cross-date means) ──────────────────────────────────────

def cross_date_mean(df):
    return df.groupby("country")["ratio"].mean().round(3)

summary = pd.DataFrame({
    "battery_tol01": cross_date_mean(bat01),
    "battery_tol05": cross_date_mean(bat05),
    "naive_tol01":   cross_date_mean(nav01),
    "naive_tol05":   cross_date_mean(nav05),
    "naive_tol10":   cross_date_mean(nav10),
}).sort_values("battery_tol05", ascending=False)

summary_path = f"{RESULTS}/comparison_naive_vs_battery_summary.csv"
summary.to_csv(summary_path)
print(f"Saved: {summary_path}")
print(summary.to_string())

# Rank correlations
for ncol in ["naive_tol01", "naive_tol05", "naive_tol10"]:
    rho = summary["battery_tol05"].corr(summary[ncol], method="spearman")
    print(f"  Spearman rho (battery_tol05 vs {ncol}): {rho:.3f}")

# ── 3. Grouped regional boxplot ───────────────────────────────────────────────
# Two panels: tol=0.05 (left) and tol=0.10 naive / tol=0.05 battery (right)
# Show battery tol=0.05 alongside naive tol=0.05 and naive tol=0.10

fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=False)

for ax, (bat_df, nav_df, tol_label) in zip(
    axes,
    [(bat05, nav05, "0.05"), (bat05, nav10, "0.05 / 0.10")]
):
    bat_plot = add_region(bat_df).copy()
    nav_plot = add_region(nav_df).copy()

    # Cap naive at 99th pct of THIS panel's data for display
    nav_cap = nav_plot["ratio"].quantile(0.99)
    nav_plot["ratio"] = nav_plot["ratio"].clip(upper=nav_cap)

    bat_plot["Model"] = "Battery"
    nav_plot["Model"] = "Naive"

    combined = pd.concat([bat_plot, nav_plot], ignore_index=True)
    combined["Region"] = pd.Categorical(combined["Region"],
                                         categories=region_order, ordered=True)

    # Palette: light shade = battery, dark = naive
    hue_palette = {"Battery": "#B0C4DE", "Naive": "#2C3E50"}

    sns.set_theme(style="whitegrid")
    sns.boxplot(
        data=combined, x="Region", y="ratio", hue="Model",
        order=region_order, palette=hue_palette, ax=ax,
        hue_order=["Battery", "Naive"],
    )
    ax.axhline(1.0, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Region", fontsize=13)
    ax.set_ylabel("Synergy Ratio", fontsize=13)
    nav_tol = tol_label.split("/")[-1].strip()
    ax.set_title(
        f"Battery tol=0.05  vs  Naive tol={nav_tol}",
        fontsize=12
    )
    ax.legend(title="Model", fontsize=11, title_fontsize=11)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=12)
    plt.setp(ax.get_yticklabels(), fontsize=11)

fig.suptitle(
    "Synergy Ratio by Region: Naive vs Battery (1x installed capacity, all 4 rolling windows)",
    fontsize=14, y=1.01
)
plt.tight_layout()
bp_path = f"{RESULTS}/comparison_naive_vs_battery_boxplot.png"
plt.savefig(bp_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {bp_path}")

# ── 4. Country scatter: naive tol=0.10 vs battery tol=0.05 ───────────────────

fig, ax = plt.subplots(figsize=(9, 8))

for country in summary.index:
    x = summary.loc[country, "naive_tol10"]
    y = summary.loc[country, "battery_tol05"]
    region = iso2_to_region.get(country, "Other")
    color  = region_palette.get(region, "#999999")
    ax.scatter(x, y, color=color, s=70, zorder=3)
    ax.annotate(country, (x, y), textcoords="offset points",
                xytext=(5, 3), fontsize=8, color=color)

# Reference line: if both models agreed perfectly (slope=1 is not meaningful
# since scales differ, so draw y=1 and x=1 instead)
ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
ax.axvline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

legend_handles = [
    mpatches.Patch(color=region_palette[r], label=r) for r in region_order
]
ax.legend(handles=legend_handles, fontsize=10, loc="upper left")

ax.set_xlabel("Naive mean SR (tol=0.10, cross-date avg)", fontsize=12)
ax.set_ylabel("Battery mean SR (tol=0.05, cross-date avg)", fontsize=12)
ax.set_title(
    "Country-level synergy: Naive (tol=0.10) vs Battery 1x (tol=0.05)\n"
    f"Spearman rank correlation: "
    f"{summary['battery_tol05'].corr(summary['naive_tol10'], method='spearman'):.3f}",
    fontsize=12
)
plt.tight_layout()
sc_path = f"{RESULTS}/comparison_naive_vs_battery_scatter.png"
plt.savefig(sc_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {sc_path}")

# ── 5. Side-by-side bar chart, sorted by battery SR ──────────────────────────

countries_sorted = summary.sort_values("battery_tol05", ascending=True).index.tolist()
x = np.arange(len(countries_sorted))
width = 0.35

fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Top panel: battery tol=0.05
ax = axes[0]
colors_bar = [region_palette.get(iso2_to_region.get(c, "Other"), "#999999")
              for c in countries_sorted]
ax.barh(x, [summary.loc[c, "battery_tol05"] for c in countries_sorted],
        color=colors_bar, edgecolor="white", height=0.7)
ax.axvline(1.0, color="red", linestyle="--", linewidth=1)
ax.set_yticks(x)
ax.set_yticklabels(countries_sorted, fontsize=9)
ax.set_xlabel("Mean Synergy Ratio (cross-date avg)", fontsize=11)
ax.set_title("Battery Model (1x installed capacity, tol=0.05)", fontsize=12)

# Bottom panel: naive tol=0.10 (same country order)
ax = axes[1]
nav10_vals = [min(summary.loc[c, "naive_tol10"], 10) for c in countries_sorted]
ax.barh(x, nav10_vals, color=colors_bar, edgecolor="white", height=0.7)
ax.axvline(1.0, color="red", linestyle="--", linewidth=1)
ax.set_yticks(x)
ax.set_yticklabels(countries_sorted, fontsize=9)
ax.set_xlabel("Mean Synergy Ratio (cross-date avg, capped at 10)", fontsize=11)
ax.set_title("Naive Model (tol=0.10)", fontsize=12)

# Shared legend
legend_handles = [mpatches.Patch(color=region_palette[r], label=r) for r in region_order]
fig.legend(handles=legend_handles, fontsize=10, loc="lower right",
           bbox_to_anchor=(0.98, 0.01))

fig.suptitle(
    "Synergy Ratios by Country — Battery (top) vs Naive (bottom)\n"
    "(both sorted by battery SR, naive capped at 10 for display)",
    fontsize=13, y=1.01
)
plt.tight_layout()
bar_path = f"{RESULTS}/comparison_naive_vs_battery_barchart.png"
plt.savefig(bar_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {bar_path}")
