"""
Recompute synergy ratios and 95% CIs for all naive model per-country CSVs.

Degenerate cases (wind+solar objective = 0) are stored as '---' instead of 0.
CIs are computed only over valid (non-NaN) observations; n_sets reflects
the number of valid runs used.

Run from the code/ directory:
    python fix_naive_ratios.py
"""

import glob
import os

import numpy as np
import pandas as pd
import scipy.stats as st

NAIVE_DIR   = "../naive_results"
RESULTS_DIR = "../results"

# ── 1. Fix per-country CSVs ────────────────────────────────────────────────────

files = sorted(glob.glob(os.path.join(NAIVE_DIR, "naive_results_*.csv")))
print(f"Found {len(files)} country files\n")

fixed_dfs = []
for f in files:
    df = pd.read_csv(f, na_values=["---"])
    denom = df["wind objective"] + df["solar objective"]
    df["ratio"] = np.where(denom <= 0, np.nan, df["combined objective"] / denom)
    df.to_csv(f, index=False, float_format="%.5f", na_rep="---")
    n_degenerate = df["ratio"].isna().sum()
    print(f"  {os.path.basename(f):30s}  degenerate runs: {n_degenerate}")
    fixed_dfs.append(df)

df_all = pd.concat(fixed_dfs, ignore_index=True)
print(f"\nTotal rows: {len(df_all)}  |  Degenerate (---): {df_all['ratio'].isna().sum()}\n")

# ── 2. Recompute 95% CIs ──────────────────────────────────────────────────────

def ci_bounds(series):
    data = series.dropna().values
    n = len(data)
    if n == 0:
        return 0, np.nan, np.nan, np.nan
    mean = float(np.mean(data))
    if n <= 5 or np.std(data) == 0:
        return n, mean, np.nan, np.nan
    lo, hi = st.t.interval(0.95, df=n - 1, loc=mean, scale=st.sem(data))
    return n, mean, float(lo), float(hi)

for day in [271, 301, 332, 362]:
    for tol in [0.01, 0.05, 0.10]:
        tol_tag = f"{tol:.2f}".replace("0.", "").replace(".", "")
        df_sub = df_all[
            (df_all["forecast_day"] == day) & np.isclose(df_all["tol"], tol)
        ].copy()

        ci_rows = []
        for country, grp in df_sub.groupby("country"):
            n, mean, lo, hi = ci_bounds(grp["ratio"])
            have_ci = n > 5 and not np.isnan(lo)
            ci_rows.append({
                "country":    country,
                "n_sets":     n,
                "mean_ratio": round(mean, 4) if not np.isnan(mean) else np.nan,
                "ci_lower":   round(lo, 4) if have_ci else np.nan,
                "ci_upper":   round(hi, 4) if have_ci else np.nan,
                "ci_string":  f"({lo:.4f}, {hi:.4f})" if have_ci else "---",
            })

        df_ci = (
            pd.DataFrame(ci_rows)
            .sort_values("country")
            .reset_index(drop=True)
        )
        out = os.path.join(RESULTS_DIR, f"naive_CI_day{day}_tol{tol_tag}.csv")
        df_ci.to_csv(out, index=False, na_rep="---")
        print(f"day={day} tol={tol:.2f}  ->  {out}")
        ie = df_ci[df_ci["country"] == "IE"]
        ee = df_ci[df_ci["country"] == "EE"]
        for row in pd.concat([ie, ee]).itertuples(index=False):
            print(f"    {row.country}  n={row.n_sets}  mean={row.mean_ratio}  CI={row.ci_string}")

print("\nDone.")
