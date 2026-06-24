#!/usr/bin/env python3
"""
Analysis pipeline for day_362, tol=0.05, 384 MW installed capacity.

Steps:
  1. Compile per-country battery result CSVs from the project root.
  2. Compute 95% CI per country (t-distribution over scenario sets).
  3. Save CI table to results/.
  4. Generate boxplot by climatological region.
  5. Decision tree regression on mean synergy ratios.
  6. Linear regression with coefficient printout.

HR, IT, SI are excluded (no day_362 results available).
"""

import os
import glob
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

ROOT = ".."
RESULTS_DIR = "../results"
PATTERN = os.path.join(ROOT, "battery_*_day_362_tol_0.05_384_installed_capacity.csv")

# ── 1. Compile results ────────────────────────────────────────────────────────

files = sorted(glob.glob(PATTERN))
print(f"Found {len(files)} country files")

records = []
for f in files:
    country = os.path.basename(f).split("_")[1]
    df = pd.read_csv(f)
    df["country"] = country
    records.append(df)

df_all = pd.concat(records, ignore_index=True)
df_all = df_all.rename(columns={
    "combine objective": "combine_objective",
    "solar objective":   "solar_objective",
    "wind objective":    "wind_objective",
})
print(f"Countries with data: {sorted(df_all['country'].unique())}")

# ── 2. Compute 95% CIs (t-distribution over sets) ────────────────────────────

def ci_bounds(data):
    n = len(data)
    if n < 2:
        return np.nan, np.nan
    lo, hi = st.t.interval(0.95, df=n - 1, loc=np.mean(data), scale=st.sem(data))
    return lo, hi

ci_rows = []
for country, grp in df_all.groupby("country"):
    ratios = grp["ratio"].values
    lo, hi = ci_bounds(ratios)
    ci_rows.append({
        "country":    country,
        "n_sets":     len(ratios),
        "mean_ratio": round(float(np.mean(ratios)), 4),
        "ci_lower":   round(float(lo), 4),
        "ci_upper":   round(float(hi), 4),
        "ci_string":  f"({lo:.4f}, {hi:.4f})",
    })

df_ci = pd.DataFrame(ci_rows).sort_values("country").reset_index(drop=True)

print("\n95% CIs — day_362, tol=0.05, 384 MW installed capacity")
print(df_ci.to_string(index=False))

ci_path = os.path.join(RESULTS_DIR, "battery_CI_day362_tol005_384.csv")
df_ci.to_csv(ci_path, index=False)
print(f"\nSaved: {ci_path}")

# ── 3. Boxplot by climatological region ──────────────────────────────────────

country_name_map = {
    "AT": "Austria",       "BE": "Belgium",        "BG": "Bulgaria",
    "CH": "Switzerland",   "CZ": "Czech Republic", "DE": "Germany",
    "DK": "Denmark",       "EE": "Estonia",         "ES": "Spain",
    "FI": "Finland",       "FR": "France",          "EL": "Greece",
    "HR": "Croatia",       "HU": "Hungary",         "IE": "Ireland",
    "IT": "Italy",         "LT": "Lithuania",       "LU": "Luxembourg",
    "LV": "Latvia",        "NL": "Netherlands",     "NO": "Norway",
    "PL": "Poland",        "PT": "Portugal",        "RO": "Romania",
    "SI": "Slovenia",      "SK": "Slovakia",        "SE": "Sweden",
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

df_plot = df_all.copy()
df_plot["country_name"] = df_plot["country"].map(country_name_map)
df_plot["Region"] = df_plot["country_name"].map(country_to_region)
df_plot = df_plot.dropna(subset=["Region"])

region_order = ["Continental", "Atlantic Maritime", "Mediterranean", "Nordic & Baltic"]
palette = ["#AEC6CF", "#2C7BB6", "#FDB863", "#78C679"]

sns.set_theme(style="whitegrid")
fig, ax = plt.subplots(figsize=(10, 7))
sns.boxplot(
    data=df_plot, x="Region", y="ratio",
    order=region_order, palette=palette, ax=ax,
)
ax.axhline(1.0, color="red", linestyle="--", linewidth=1, label="Ratio = 1 (no synergy)")
ax.legend(fontsize=12)
ax.set_xlabel("Region", fontsize=14)
ax.set_ylabel("Synergy Ratio", fontsize=14)
ax.set_title(
    "Synergy Ratio by Region\n(day_362, tol=0.05, 384 MW installed capacity)",
    fontsize=13
)
plt.xticks(rotation=30, ha="right", fontsize=13)
plt.yticks(fontsize=13)
plt.tight_layout()
bp_path = os.path.join(RESULTS_DIR, "boxplot_day362_tol005_384.png")
plt.savefig(bp_path, dpi=300)
plt.close()
print(f"Saved: {bp_path}")

# ── 4. ML feature table (fixed — derived from EMHIRES data) ──────────────────

features_by_country = {
    #        drop   reduce_var  avg_corr   pv_share
    "AT": (0.020,  0.146,  -0.249,  0.088),
    "BE": (0.020,  0.037,  -0.358,  0.418),
    "BG": (0.012,  0.016,  -0.318,  0.439),
    "CH": (0.007,  0.006,  -0.347,  0.095),
    "CZ": (0.003,  0.003,  -0.302,  0.211),
    "DE": (0.008,  0.020,  -0.358,  0.289),
    "DK": (0.002,  0.020,  -0.290,  0.077),
    "EE": (0.002,  0.045,  -0.320,  0.006),
    "EL": (0.016,  0.022,  -0.225,  0.471),
    "ES": (0.045,  0.164,  -0.343,  0.170),
    "FI": (0.002,  0.057,  -0.345,  0.003),
    "FR": (0.030,  0.084,  -0.369,  0.251),
    "HR": (0.128,  0.587,  -0.307,  0.081),
    "HU": (0.051,  0.405,  -0.175,  0.069),
    "IE": (0.000,  0.000,  -0.325,  0.000),
    "IT": (0.008,  0.009,  -0.306,  0.311),
    "LT": (0.023,  0.118,  -0.283,  0.091),
    "LU": (0.009,  0.012,  -0.263,  0.374),
    "LV": (0.006,  0.068,  -0.300,  0.016),
    "NL": (0.038,  0.176,  -0.367,  0.144),
    "NO": (0.005,  0.206,  -0.475,  0.003),
    "PL": (0.001,  0.020,  -0.302,  0.007),
    "PT": (0.047,  0.258,  -0.240,  0.078),
    "RO": (0.011,  0.036,  -0.308,  0.235),
    "SI": (0.004,  0.003,  -0.260,  0.012),
    "SK": (0.000,  0.000,  -0.233,  0.007),
    "SE": (0.001,  0.024,  -0.328,  0.008),
    "UK": (0.008,  0.031,  -0.363,  0.173),
}

df_features = pd.DataFrame.from_dict(
    features_by_country, orient="index",
    columns=["drop", "reduce_var", "average_corr", "pv_share"]
)
df_features.index.name = "country"

# Inner join: only countries with day_362 results
df_ml = df_features.join(
    df_ci[["country", "mean_ratio"]].set_index("country"),
    how="inner"
)
missing = set(features_by_country) - set(df_ml.index)
print(f"\nML analysis: {len(df_ml)} countries  |  excluded (no day_362 data): {sorted(missing)}")

feat_cols = ["drop", "reduce_var", "average_corr", "pv_share"]
X = df_ml[feat_cols]
y = df_ml["mean_ratio"]

# ── 5. Decision tree ──────────────────────────────────────────────────────────

dt = DecisionTreeRegressor(max_depth=3, random_state=42)
dt.fit(X, y)

plt.figure(figsize=(18, 10))
plot_tree(
    dt, feature_names=feat_cols,
    filled=True, rounded=True, precision=3, fontsize=9
)
plt.title(
    "Decision Tree — Synergy Ratio (day_362, tol=0.05, 384 MW installed capacity)",
    fontsize=13
)
plt.tight_layout()
dt_path = os.path.join(RESULTS_DIR, "decision_tree_day362_tol005_384.png")
plt.savefig(dt_path, dpi=300)
plt.close()
print(f"Saved: {dt_path}")

importances = pd.Series(dt.feature_importances_, index=feat_cols).sort_values(ascending=False)
print("\nDecision Tree Feature Importances:")
print(importances.to_string())

# ── 6. Linear regression ──────────────────────────────────────────────────────

reg = LinearRegression()
reg.fit(X, y)
r2 = r2_score(y, reg.predict(X))

print(f"\nLinear Regression R² = {r2:.4f}")
print(f"Intercept: {reg.intercept_:.4f}")
for feat, coef in zip(feat_cols, reg.coef_):
    print(f"  {feat:15s}: {coef:+.4f}")

print("\nDone. All outputs in results/")
