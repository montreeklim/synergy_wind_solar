#!/usr/bin/env python3
"""
Analysis pipeline for naive (no-battery) 1000-scenario rolling-window results.

Each country file has one row per (forecast_day, tol) — a single pooled solve
over 1000 scenarios — so outputs are point estimates rather than CI-based.

Usage:
    python analyze_naive_results_1000.py --day 271 --tol 0.05

Reads  : ../naive_results_1000_*.csv  (28 per-country files in project root)
Outputs: ../results/naive_1000_summary_day{DAY}_tol{TAG}.csv
         ../results/naive_1000_boxplot_day{DAY}_tol{TAG}.png
         ../results/naive_1000_heatmap_day{DAY}_tol{TAG}.png
         ../results/naive_1000_decision_tree_day{DAY}_tol{TAG}.png
"""

import argparse
import os
import glob
import warnings
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
import matplotlib.patheffects as PathEffects
import seaborn as sns
import geopandas as gpd
import pycountry
from cartopy.crs import PlateCarree
from cartopy.io.shapereader import natural_earth
from matplotlib.colors import Normalize
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# ── CLI ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--day", required=True, type=int, choices=[271, 301, 332, 362])
parser.add_argument("--tol", type=float, default=0.05, choices=[0.01, 0.05, 0.10])
args = parser.parse_args()
DAY = args.day
TOL = args.tol
tol_tag = f"{TOL:.2f}".replace("0.", "").replace(".", "")

RESULTS_DIR = "../results"
TAG = f"day{DAY}_tol{tol_tag}"

print(f"\n=== Naive 1000-scenario model | Day {DAY} | tol={TOL} ===\n")

# ── 1. Load and merge all per-country CSVs ────────────────────────────────────

files = sorted(glob.glob("../naive_results_1000_*.csv"))
print(f"Found {len(files)} country files")

df_all = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

# Filter to the requested day and tol
df = df_all[(df_all["forecast_day"] == DAY) & (np.isclose(df_all["tol"], TOL))].copy()
print(f"Rows after filtering to day={DAY}, tol={TOL}: {len(df)}")
print(f"Countries: {sorted(df['country'].unique())}")

# ── 2. Point-estimate summary (n=1 per country, no CI) ────────────────────────

summary_rows = []
for country, grp in df.groupby("country"):
    ratio = float(grp["ratio"].iloc[0])
    summary_rows.append({
        "country":  country,
        "n_scenarios": 1000,
        "ratio":    round(ratio, 4),
    })

df_summary = pd.DataFrame(summary_rows).sort_values("country").reset_index(drop=True)
print("\nPoint estimates:")
print(df_summary.to_string(index=False))

summary_path = os.path.join(RESULTS_DIR, f"naive_1000_summary_{TAG}.csv")
df_summary.to_csv(summary_path, index=False)
print(f"\nSaved: {summary_path}")

# ── 3. Boxplot by region (spread across countries within each region) ─────────

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
country_to_region = {c: r for r, cs in regions.items() for c in cs}

df_plot = df_summary.copy()
df_plot["country_name"] = df_plot["country"].map(country_name_map)
df_plot["Region"] = df_plot["country_name"].map(country_to_region)
df_plot = df_plot.dropna(subset=["Region"])

region_order = ["Continental", "Atlantic Maritime", "Mediterranean", "Nordic & Baltic"]
palette = ["#AEC6CF", "#2C7BB6", "#FDB863", "#78C679"]

sns.set_theme(style="whitegrid")
fig, ax = plt.subplots(figsize=(10, 7))
sns.boxplot(data=df_plot, x="Region", y="ratio",
            order=region_order, palette=palette, ax=ax)
sns.stripplot(data=df_plot, x="Region", y="ratio",
              order=region_order, color="black", size=5, alpha=0.7, ax=ax)
ax.axhline(1.0, color="red", linestyle="--", linewidth=1, label="Ratio = 1 (no synergy)")
ax.legend(fontsize=12)
ax.set_xlabel("Region", fontsize=14)
ax.set_ylabel("Synergy Ratio (1000 scenarios)", fontsize=14)
ax.set_title(
    f"Naive Model (1000 scenarios) — Synergy Ratio by Region\n(day_{DAY}, tol={TOL})",
    fontsize=13,
)
plt.xticks(rotation=30, ha="right", fontsize=13)
plt.yticks(fontsize=13)
plt.tight_layout()
bp_path = os.path.join(RESULTS_DIR, f"naive_1000_boxplot_{TAG}.png")
plt.savefig(bp_path, dpi=300)
plt.close()
print(f"Saved: {bp_path}")

# ── 4. Geographic heatmap ─────────────────────────────────────────────────────

iso2to3 = {c.alpha_2: c.alpha_3 for c in pycountry.countries}
iso2to3.update({
    "UK": "GBR", "EL": "GRC", "GR": "GRC", "CZ": "CZE",
    "BA": "BIH", "ME": "MNE", "MK": "MKD", "RS": "SRB", "XK": "KOS",
})
all28_iso3 = {iso2to3[c] for c in country_name_map if c in iso2to3}

df_map = df_summary[["country", "ratio"]].rename(columns={"country": "ISO2"}).copy()
df_map["ISO3"] = df_map["ISO2"].map(iso2to3)

shp   = natural_earth("110m", "cultural", "admin_0_countries")
world = gpd.read_file(shp)

df_geo = (
    world[world.ADM0_A3.isin(all28_iso3)]
    .merge(df_map, left_on="ADM0_A3", right_on="ISO3", how="left")
    .explode(index_parts=False)
)

df_geo = df_geo.to_crs(epsg=4326)
rp = df_geo.geometry.representative_point()
df_geo["lon"], df_geo["lat"] = rp.x, rp.y
fr_mask = ((df_geo.ISO2 == "FR") &
           ((df_geo.lon < -5.5) | (df_geo.lon > 8.2) |
            (df_geo.lat < 41.3) | (df_geo.lat > 51.1)))
no_mask = (df_geo.ISO2 == "NO") & (df_geo.lat > 72)
df_geo = df_geo[~(fr_mask | no_mask)].drop(columns=["lon", "lat"])
df_geo = df_geo.to_crs(epsg=4326)

df_has  = df_geo[df_geo["ratio"].notna()].copy()
df_miss = df_geo[df_geo["ratio"].isna()].copy()

fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={"projection": PlateCarree()})

vmin = 1.0
vmax = df_has["ratio"].quantile(0.95)
norm = Normalize(vmin=vmin, vmax=vmax)
cmap = "YlGn"

df_has.plot(column="ratio", cmap=cmap, norm=norm,
            edgecolor="black", linewidth=0.5,
            ax=ax, transform=PlateCarree(), legend=False)
if not df_miss.empty:
    df_miss.plot(color="#CCCCCC", edgecolor="black", linewidth=0.5,
                 ax=ax, transform=PlateCarree())

ax.set_facecolor("#f0f0f0")
ax.set_xticks([]); ax.set_yticks([])

pos = ax.get_position()
cax = fig.add_axes([pos.x0, pos.y0 - 0.05, pos.width, 0.02])
sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm._A = []
cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
cbar.set_label("Synergy Ratio (1000 scenarios)", fontsize=12)
cbar.ax.tick_params(labelsize=10)

all_labels = df_geo.dissolve(by="ISO2").representative_point()
for iso2, pt in all_labels.geometry.items():
    txt = ax.text(pt.x, pt.y, iso2, transform=PlateCarree(),
                  ha="center", va="center", fontsize=7, fontweight="bold")
    txt.set_path_effects([
        PathEffects.Stroke(linewidth=1.5, foreground="white"),
        PathEffects.Normal(),
    ])

missing_countries = sorted(df_miss.dissolve(by="ISO2").index.tolist()) if not df_miss.empty else []
if missing_countries:
    patch = mpatches.Patch(color="#CCCCCC", label=f"No data: {', '.join(missing_countries)}")
    ax.legend(handles=[patch], loc="lower left", fontsize=9)

ax.set_title(
    f"Naive Model (1000 scenarios) — Synergy Ratio by Country\n(day_{DAY}, tol={TOL})",
    fontsize=15, pad=14,
)
hm_path = os.path.join(RESULTS_DIR, f"naive_1000_heatmap_{TAG}.png")
plt.savefig(hm_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {hm_path}")

# ── 5. ML features ────────────────────────────────────────────────────────────

features_by_country = {
    "AT": (0.020, 0.146, -0.249, 0.088),
    "BE": (0.020, 0.037, -0.358, 0.418),
    "BG": (0.012, 0.016, -0.318, 0.439),
    "CH": (0.007, 0.006, -0.347, 0.095),
    "CZ": (0.003, 0.003, -0.302, 0.211),
    "DE": (0.008, 0.020, -0.358, 0.289),
    "DK": (0.002, 0.020, -0.290, 0.077),
    "EE": (0.002, 0.045, -0.320, 0.006),
    "EL": (0.016, 0.022, -0.225, 0.471),
    "ES": (0.045, 0.164, -0.343, 0.170),
    "FI": (0.002, 0.057, -0.345, 0.003),
    "FR": (0.030, 0.084, -0.369, 0.251),
    "HR": (0.128, 0.587, -0.307, 0.081),
    "HU": (0.051, 0.405, -0.175, 0.069),
    "IE": (0.000, 0.000, -0.325, 0.000),
    "IT": (0.008, 0.009, -0.306, 0.311),
    "LT": (0.023, 0.118, -0.283, 0.091),
    "LU": (0.009, 0.012, -0.263, 0.374),
    "LV": (0.006, 0.068, -0.300, 0.016),
    "NL": (0.038, 0.176, -0.367, 0.144),
    "NO": (0.005, 0.206, -0.475, 0.003),
    "PL": (0.001, 0.020, -0.302, 0.007),
    "PT": (0.047, 0.258, -0.240, 0.078),
    "RO": (0.011, 0.036, -0.308, 0.235),
    "SI": (0.004, 0.003, -0.260, 0.012),
    "SK": (0.000, 0.000, -0.233, 0.007),
    "SE": (0.001, 0.024, -0.328, 0.008),
    "UK": (0.008, 0.031, -0.363, 0.173),
}

df_features = pd.DataFrame.from_dict(
    features_by_country, orient="index",
    columns=["drop", "reduce_var", "average_corr", "pv_share"]
)
df_features.index.name = "country"

df_ml = df_features.join(df_summary[["country", "ratio"]].set_index("country"), how="inner")
feat_cols = ["drop", "reduce_var", "average_corr", "pv_share"]
X = df_ml[feat_cols]
y = df_ml["ratio"]

# ── 6. Decision tree ──────────────────────────────────────────────────────────

dt = DecisionTreeRegressor(max_depth=3, random_state=42)
dt.fit(X, y)

plt.figure(figsize=(18, 10))
plot_tree(dt, feature_names=feat_cols, filled=True, rounded=True, precision=3, fontsize=9)
plt.title(f"Decision Tree — Naive 1000-scenario SR (day_{DAY}, tol={TOL})", fontsize=13)
plt.tight_layout()
dt_path = os.path.join(RESULTS_DIR, f"naive_1000_decision_tree_{TAG}.png")
plt.savefig(dt_path, dpi=300)
plt.close()
print(f"Saved: {dt_path}")

importances = pd.Series(dt.feature_importances_, index=feat_cols).sort_values(ascending=False)
print("Decision Tree Feature Importances:")
print(importances.to_string())

# ── 7. Linear regression ──────────────────────────────────────────────────────

reg = LinearRegression()
reg.fit(X, y)
r2 = r2_score(y, reg.predict(X))
print(f"\nLinear Regression R2 = {r2:.4f}")
print(f"Intercept: {reg.intercept_:.4f}")
for feat, coef in zip(feat_cols, reg.coef_):
    print(f"  {feat:15s}: {coef:+.4f}")

print(f"\n=== Day {DAY} | tol={TOL} complete ===\n")
