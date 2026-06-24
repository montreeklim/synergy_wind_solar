#!/usr/bin/env python3
"""
Compare 1000-scenario point estimates against 100-scenario 95% CIs
for the naive model across all (day, tol) combinations.

Usage:
    python compare_1000_vs_100_naive.py

Reads:
    ../results/naive_CI_day{DAY}_tol{TAG}.csv         (100-scenario, 10 sets)
    ../results/naive_1000_summary_day{DAY}_tol{TAG}.csv (1000-scenario, 1 solve)

Outputs:
    ../results/naive_1000_vs_100_comparison.csv
    ../results/naive_1000_vs_100_coverage_heatmap.png   (within-CI flag per country/day/tol)
    ../results/naive_1000_vs_100_scatter_tol{TAG}.png   (1000 vs 100 mean, per tol)
"""

import os
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
import seaborn as sns

RESULTS = "../results"
DAYS    = [271, 301, 332, 362]
TOLS    = [0.01, 0.05, 0.10]

# Ratios above this threshold are treated as numerically unstable / extreme
EXTREME_THRESHOLD = 500

# ── 1. Load and join all (day, tol) pairs ─────────────────────────────────────

rows = []
for tol in TOLS:
    tag = f"{tol:.2f}".replace("0.", "").replace(".", "")
    for day in DAYS:
        ci_path  = f"{RESULTS}/naive_CI_day{day}_tol{tag}.csv"
        s1k_path = f"{RESULTS}/naive_1000_summary_day{day}_tol{tag}.csv"
        if not (os.path.exists(ci_path) and os.path.exists(s1k_path)):
            print(f"  Missing: day={day} tol={tol}")
            continue
        df_ci  = pd.read_csv(ci_path)
        df_1k  = pd.read_csv(s1k_path)
        merged = df_ci.merge(df_1k[["country", "ratio"]], on="country", suffixes=("", "_1000"))
        merged.rename(columns={"ratio": "ratio_1000"}, inplace=True)
        merged["day"] = day
        merged["tol"] = tol
        rows.append(merged)

df = pd.concat(rows, ignore_index=True)
df = df.sort_values(["tol", "day", "country"]).reset_index(drop=True)

# ── 2. Derived columns ────────────────────────────────────────────────────────

df["extreme_1000"] = df["ratio_1000"] > EXTREME_THRESHOLD

# Within CI: only meaningful for non-extreme rows
df["within_ci"] = (
    (df["ratio_1000"] >= df["ci_lower"]) &
    (df["ratio_1000"] <= df["ci_upper"])
)
df.loc[df["extreme_1000"], "within_ci"] = False

# Relative deviation from 100-scenario mean (signed, %)
df["rel_dev_pct"] = (df["ratio_1000"] - df["mean_ratio"]) / df["mean_ratio"] * 100
df.loc[df["extreme_1000"], "rel_dev_pct"] = np.nan  # suppress extreme in stats

# ── 3. Save full comparison CSV ───────────────────────────────────────────────

out_cols = ["tol", "day", "country",
            "n_sets", "mean_ratio", "ci_lower", "ci_upper",
            "ratio_1000", "extreme_1000", "within_ci", "rel_dev_pct"]
df[out_cols].to_csv(f"{RESULTS}/naive_1000_vs_100_comparison.csv", index=False, float_format="%.4f")
print(f"Saved: {RESULTS}/naive_1000_vs_100_comparison.csv")

# ── 4. Summary statistics per tol ─────────────────────────────────────────────

print("\n" + "="*65)
print("SUMMARY BY TOLERANCE (non-extreme rows only)")
print("="*65)
for tol in TOLS:
    sub = df[df["tol"] == tol]
    n_total   = len(sub)
    n_extreme = sub["extreme_1000"].sum()
    n_ok      = n_total - n_extreme
    n_within  = sub.loc[~sub["extreme_1000"], "within_ci"].sum()
    coverage  = n_within / n_ok * 100 if n_ok > 0 else 0
    med_dev   = sub.loc[~sub["extreme_1000"], "rel_dev_pct"].median()
    print(f"\n  tol={tol}")
    print(f"    Total obs      : {n_total}")
    print(f"    Extreme (>500) : {n_extreme}  ({n_extreme/n_total*100:.0f}%)")
    print(f"    Within 95% CI  : {n_within}/{n_ok}  ({coverage:.0f}%)")
    print(f"    Median rel dev : {med_dev:+.1f}%")

print("\n" + "="*65)
print("PER-DAY COVERAGE (tol=0.10, extreme excluded)")
print("="*65)
sub10 = df[(df["tol"] == 0.10) & (~df["extreme_1000"])]
for day in DAYS:
    s = sub10[sub10["day"] == day]
    n_in = s["within_ci"].sum()
    print(f"  Day {day}: {n_in}/{len(s)} within CI  ({n_in/len(s)*100:.0f}%)")

# ── 5. Coverage heatmap (within_ci flag, tol=0.10 only) ──────────────────────

sub10_all = df[df["tol"] == 0.10].copy()
sub10_all["label"] = sub10_all.apply(
    lambda r: "extreme" if r["extreme_1000"]
    else ("✓" if r["within_ci"] else "✗"),
    axis=1
)

pivot_flag = sub10_all.pivot(index="country", columns="day", values="within_ci").astype(float)
pivot_flag_ex = sub10_all.pivot(index="country", columns="day", values="extreme_1000")

# Sort countries by fraction within CI descending
pivot_flag["_frac"] = pivot_flag.mean(axis=1)
pivot_flag = pivot_flag.sort_values("_frac", ascending=False).drop(columns="_frac")

fig, ax = plt.subplots(figsize=(8, 11))

# Build annotation matrix
annot = pd.DataFrame("", index=pivot_flag.index, columns=pivot_flag.columns)
for c in pivot_flag.index:
    for d in pivot_flag.columns:
        row = sub10_all[(sub10_all["country"] == c) & (sub10_all["day"] == d)]
        if row.empty: continue
        if row["extreme_1000"].values[0]:
            annot.loc[c, d] = "EXT"
        elif row["within_ci"].values[0]:
            annot.loc[c, d] = "IN"
        else:
            annot.loc[c, d] = "OUT"

color_map = {"IN": 1.0, "OUT": 0.0, "EXT": 0.5}
color_mat = annot.applymap(lambda x: color_map.get(x, np.nan))

sns.heatmap(
    color_mat.astype(float),
    annot=annot, fmt="",
    cmap=["#e74c3c", "#f39c12", "#27ae60"],
    vmin=0, vmax=1,
    linewidths=0.5, linecolor="white",
    ax=ax,
    cbar=False,
)
ax.set_title(
    "1000-scenario ratio vs 100-scenario 95% CI (ε = 0.10)\n"
    "IN = within CI   OUT = outside CI   EXT = extreme (>500)",
    fontsize=12, pad=12
)
ax.set_xlabel("Rolling Window Day", fontsize=11)
ax.set_ylabel("")
ax.tick_params(axis="x", labelsize=10)
ax.tick_params(axis="y", labelsize=9)

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="#27ae60", label="IN — within 95% CI"),
    Patch(facecolor="#e74c3c", label="OUT — outside 95% CI"),
    Patch(facecolor="#f39c12", label="EXT — extreme (>500)"),
]
ax.legend(handles=legend_elements, loc="lower right", fontsize=9, framealpha=0.9)

plt.tight_layout()
hm_path = f"{RESULTS}/naive_1000_vs_100_coverage_heatmap.png"
plt.savefig(hm_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"\nSaved: {hm_path}")

# ── 6. Scatter: 1000-scenario ratio vs 100-scenario mean (per tol) ────────────

for tol in TOLS:
    tag = f"{tol:.2f}".replace("0.", "").replace(".", "")
    sub = df[(df["tol"] == tol) & (~df["extreme_1000"])].copy()

    fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharey=False)
    for ax, day in zip(axes, DAYS):
        s = sub[sub["day"] == day]
        ax.errorbar(
            s["mean_ratio"], s["ratio_1000"],
            xerr=[s["mean_ratio"] - s["ci_lower"],
                  s["ci_upper"]  - s["mean_ratio"]],
            fmt="none", color="steelblue", alpha=0.5, linewidth=1, capsize=3,
        )
        inside = s[s["within_ci"]]
        outside = s[~s["within_ci"]]
        ax.scatter(inside["mean_ratio"],  inside["ratio_1000"],
                   color="#27ae60", s=50, zorder=3, label="Within CI")
        ax.scatter(outside["mean_ratio"], outside["ratio_1000"],
                   color="#e74c3c", s=60, marker="^", zorder=3, label="Outside CI")

        lo = min(s["ci_lower"].min(), s["ratio_1000"].min(), 0)
        hi = max(s["ci_upper"].max(), s["ratio_1000"].max())
        pad = (hi - lo) * 0.05
        lim = (max(0, lo - pad), hi + pad)
        ax.plot(lim, lim, "k--", linewidth=0.8, alpha=0.5)  # y=x reference
        ax.set_xlim(lim); ax.set_ylim(lim)

        for _, r in s.iterrows():
            ax.annotate(r["country"], (r["mean_ratio"], r["ratio_1000"]),
                        textcoords="offset points", xytext=(4, 4), fontsize=6)

        ax.set_title(f"Day {day}", fontsize=11)
        ax.set_xlabel("100-scen mean ratio", fontsize=9)
        if ax is axes[0]:
            ax.set_ylabel("1000-scen ratio", fontsize=9)
        ax.legend(fontsize=7, loc="upper left")

    fig.suptitle(
        f"Naive model — 1000-scenario vs 100-scenario mean synergy ratio  (ε={tol})\n"
        f"Horizontal bars = 95% CI from 100 scenarios.  Extreme values (>500) excluded.",
        fontsize=12, y=1.01
    )
    plt.tight_layout()
    sc_path = f"{RESULTS}/naive_1000_vs_100_scatter_tol{tag}.png"
    plt.savefig(sc_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {sc_path}")

# ── 7. Detailed table for tol=0.10 ───────────────────────────────────────────

print("\n" + "="*65)
print("DETAILED TABLE: tol=0.10 (all days stacked)")
print("="*65)
sub10_detail = df[df["tol"] == 0.10][
    ["country", "day", "mean_ratio", "ci_lower", "ci_upper",
     "ratio_1000", "extreme_1000", "within_ci", "rel_dev_pct"]
].copy()
sub10_detail["status"] = sub10_detail.apply(
    lambda r: "EXT" if r["extreme_1000"] else ("IN" if r["within_ci"] else "OUT"), axis=1
)
print(sub10_detail.drop(columns=["extreme_1000", "within_ci"])
      .to_string(index=False, float_format=lambda x: f"{x:.3f}"))
