"""
Compute ratio-of-means synergy ratios for the Naive model.

Unlike naive_CI_calculation.py which computes mean(per-set ratio), this script
computes mean(combined) / (mean(wind) + mean(solar)) within each group, which is
robust to near-zero denominators that inflate individual-set ratios.

CIs use bootstrap (percentile method) rather than t-distribution, which is more
appropriate for a ratio estimator.

Outputs
-------
results/ratio_of_means_naive.csv          -- pooled across all forecast days
results/ratio_of_means_naive_by_day.csv   -- stratified by forecast day
"""

import glob
import numpy as np
import pandas as pd

NAIVE_RESULTS_DIR = "naive_results"
OUT_POOLED = "results/ratio_of_means_naive.csv"
OUT_BY_DAY = "results/ratio_of_means_naive_by_day.csv"
N_BOOT = 2000
ALPHA = 0.05
SEED = 42
# Flag rows where the denominator (wind+solar individual benefit) is below this
# fraction of the combined benefit — these ratios are artefacts, not synergy.
DENOM_REL_THRESHOLD = 0.05

rng = np.random.default_rng(SEED)

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
files = sorted(glob.glob(f"{NAIVE_RESULTS_DIR}/naive_results_*.csv"))
if not files:
    raise FileNotFoundError(f"No files found under {NAIVE_RESULTS_DIR}/")

df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
df.columns = df.columns.str.strip()

# Clip numerical solver noise (values like -0.00000) to zero
for col in ["wind objective", "solar objective", "combined objective"]:
    df[col] = df[col].clip(lower=0)

print(f"Loaded {len(df):,} rows from {len(files)} country files")
print(f"  tolerances   : {sorted(df['tol'].unique())}")
print(f"  forecast days: {sorted(df['forecast_day'].unique())}")
print(f"  countries    : {sorted(df['country'].unique())}\n")


# ---------------------------------------------------------------------------
# Bootstrap helper (vectorised)
# ---------------------------------------------------------------------------
def bootstrap_ratio_of_means(w, s, c, n_boot=N_BOOT, alpha=ALPHA):
    """
    Resample rows with replacement; compute ratio-of-means each time.
    Returns (point_estimate, ci_lower, ci_upper).
    """
    n = len(w)
    denom_point = w.mean() + s.mean()
    if denom_point == 0:
        return np.nan, np.nan, np.nan
    point = c.mean() / denom_point

    idx = rng.integers(0, n, size=(n_boot, n))
    w_b = w[idx].mean(axis=1)
    s_b = s[idx].mean(axis=1)
    c_b = c[idx].mean(axis=1)
    denom_b = w_b + s_b
    valid = denom_b > 0
    boot = np.where(valid, c_b / np.where(valid, denom_b, 1.0), np.nan)
    boot = boot[~np.isnan(boot)]

    if len(boot) < n_boot * 0.5:
        return point, np.nan, np.nan
    lo = np.percentile(boot, 100 * alpha / 2)
    hi = np.percentile(boot, 100 * (1 - alpha / 2))
    return point, lo, hi


def build_record(group_keys: dict, grp: pd.DataFrame) -> dict:
    w = grp["wind objective"].values
    s = grp["solar objective"].values
    c = grp["combined objective"].values

    point, lo, hi = bootstrap_ratio_of_means(w, s, c)

    denom = w.mean() + s.mean()
    comb = c.mean()
    near_zero = (denom / max(comb, 1e-12)) < DENOM_REL_THRESHOLD

    return {
        **group_keys,
        "n_obs": len(grp),
        "ratio_of_means": point,
        "ci_lower_95": lo,
        "ci_upper_95": hi,
        "ci_width": (hi - lo) if not (np.isnan(lo) or np.isnan(hi)) else np.nan,
        "mean_wind_obj": w.mean(),
        "mean_solar_obj": s.mean(),
        "mean_combined_obj": comb,
        "near_zero_denom": near_zero,
    }


# ---------------------------------------------------------------------------
# Pooled across forecast days (country × tol)
# ---------------------------------------------------------------------------
print("Computing pooled ratio-of-means (country × tol)...")
pooled_records = []
for (country, tol), grp in df.groupby(["country", "tol"]):
    pooled_records.append(build_record({"country": country, "tol": tol}, grp))

pooled_df = (
    pd.DataFrame(pooled_records)
    .sort_values(["country", "tol"])
    .reset_index(drop=True)
)
pooled_df.to_csv(OUT_POOLED, index=False, float_format="%.5f")
print(f"  Saved {len(pooled_df)} rows -> {OUT_POOLED}")


# ---------------------------------------------------------------------------
# Stratified by forecast day (country × tol × day)
# ---------------------------------------------------------------------------
print("Computing per-day ratio-of-means (country × tol × forecast_day)...")
day_records = []
for (country, tol, day), grp in df.groupby(["country", "tol", "forecast_day"]):
    day_records.append(
        build_record({"country": country, "tol": tol, "forecast_day": day}, grp)
    )

day_df = (
    pd.DataFrame(day_records)
    .sort_values(["country", "tol", "forecast_day"])
    .reset_index(drop=True)
)
day_df.to_csv(OUT_BY_DAY, index=False, float_format="%.5f")
print(f"  Saved {len(day_df)} rows -> {OUT_BY_DAY}")


# ---------------------------------------------------------------------------
# Quick comparison: mean-of-ratios vs ratio-of-means at tol=0.05
# ---------------------------------------------------------------------------
print("\n--- mean-of-ratios vs ratio-of-means (tol = 0.05) ---")
mor = (
    df.groupby(["country", "tol"])["ratio"]
    .mean()
    .reset_index()
    .rename(columns={"ratio": "mean_of_ratios"})
)
cmp = mor.merge(
    pooled_df[["country", "tol", "ratio_of_means", "ci_lower_95", "ci_upper_95", "near_zero_denom"]],
    on=["country", "tol"],
)
cmp05 = cmp[cmp["tol"] == 0.05].copy()
cmp05["difference"] = cmp05["ratio_of_means"] - cmp05["mean_of_ratios"]
print(
    cmp05[["country", "mean_of_ratios", "ratio_of_means", "ci_lower_95", "ci_upper_95", "near_zero_denom"]]
    .to_string(index=False, float_format=lambda x: f"{x:.4f}")
)

flagged = pooled_df[pooled_df["near_zero_denom"]].groupby("country")["tol"].apply(list)
if not flagged.empty:
    print(f"\nNear-zero denominator flags (denom < {DENOM_REL_THRESHOLD:.0%} of combined):")
    for country, tols in flagged.items():
        print(f"  {country}: tol = {tols}")
