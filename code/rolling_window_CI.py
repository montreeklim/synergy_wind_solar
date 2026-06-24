"""
Aggregate rolling-window battery results into confidence intervals.

Reads per-country CSV files produced by parallel_battery_model_cap.py
across multiple forecast days and tolerances, then computes:
  - per-date 95% CI (over 10 scenario sets)
  - cross-date mean SR and 95% CI (over 4 dates × 10 sets = 40 observations)

Usage:
    python rolling_window_CI.py
"""

import pandas as pd
import numpy as np
import scipy.stats as st
import os
import glob

# ---- configuration ----
FORECAST_DAYS = [271, 301, 332, 362]
TOLERANCES    = [0.01, 0.05, 0.10]
COUNTRIES = [
    "AT","BE","BG","CH","CZ","DE","DK","EE","ES","FI","FR","EL",
    "HR","HU","IE","IT","LT","LU","LV","NL","NO","PL","PT","RO",
    "SI","SK","SE","UK",
]


def ci95(data):
    """95% CI using t-distribution (n=10 or n=40)."""
    data = pd.Series(data).dropna()
    n = len(data)
    if n < 2:
        return (np.nan, np.nan)
    lo, hi = st.t.interval(0.95, df=n-1, loc=data.mean(), scale=st.sem(data))
    return (round(lo, 2), round(hi, 2))


# ---- load all per-country result files ----
records = []
for day in FORECAST_DAYS:
    for tol in TOLERANCES:
        tol_str = str(tol)
        for country in COUNTRIES:
            fname = (f'battery_{country}_day_{day}_tol_{tol}'
                     f'_384_installed_capacity.csv')
            if not os.path.exists(fname):
                continue
            df = pd.read_csv(fname)
            df['forecast_day'] = day
            df['tol'] = tol
            df['country'] = country
            records.append(df)

if not records:
    raise FileNotFoundError(
        "No battery result files found. Run parallel_battery_model_cap.py first."
    )

all_df = pd.concat(records, ignore_index=True)

# absolute gain: combined minus sum of individual objectives
all_df['abs_gain'] = (all_df['combine objective']
                      - all_df['wind objective']
                      - all_df['solar objective'])

# total installed capacity (MW) for normalisation
installed_capacity_2015 = {
    "AT": 2385, "BE": 5240, "BG": 1742, "CH": 816,  "CZ": 2344,
    "DE": 81840,"DK": 5863, "EE": 307,  "ES": 29970, "FI": 1093,
    "FR": 16504,"EL": 4219, "HR": 428,  "HU": 357,  "IE": 2401,
    "IT": 27850,"LT": 359,  "LU": 176,  "LV": 72,   "NL": 5070,
    "NO": 874,  "PL": 5273, "PT": 5255, "RO": 4172, "SI": 266,
    "SK": 535,  "SE": 3133, "UK": 22563,
}
all_df['capacity_mw'] = all_df['country'].map(installed_capacity_2015)
all_df['gain_per_mw'] = all_df['abs_gain'] / all_df['capacity_mw']

# ---- per-date CI (same as existing Table 2, now per date) ----
per_date_rows = []
for (country, day, tol), grp in all_df.groupby(['country', 'forecast_day', 'tol']):
    lo, hi = ci95(grp['ratio'])
    per_date_rows.append({
        'country': country, 'forecast_day': day, 'tol': tol,
        'ci_lower': lo, 'ci_upper': hi,
        'mean_sr': round(grp['ratio'].mean(), 3),
        'mean_abs_gain_k': round(grp['abs_gain'].mean(), 3),
        'mean_gain_per_mw': round(grp['gain_per_mw'].mean(), 2),
        'n': len(grp),
    })

per_date_df = pd.DataFrame(per_date_rows)
per_date_df.to_csv('battery_CI_rolling_per_date.csv', index=False)
print("Saved battery_CI_rolling_per_date.csv")

# ---- cross-date CI (40 obs per country × tol) ----
cross_date_rows = []
for (country, tol), grp in all_df.groupby(['country', 'tol']):
    lo, hi = ci95(grp['ratio'])
    cross_date_rows.append({
        'country': country, 'tol': tol,
        'ci_lower': lo, 'ci_upper': hi,
        'mean_sr': round(grp['ratio'].mean(), 3),
        'mean_abs_gain_k': round(grp['abs_gain'].mean(), 3),
        'mean_gain_per_mw': round(grp['gain_per_mw'].mean(), 2),
        'n': len(grp),
    })

cross_date_df = pd.DataFrame(cross_date_rows)
cross_date_df.to_csv('battery_CI_rolling_cross_date.csv', index=False)
print("Saved battery_CI_rolling_cross_date.csv")

# ---- consistency check: do per-date CIs overlap across dates? ----
print("\n--- Cross-date SR stability (mean ± std across 4 dates) ---")
stability = (all_df.groupby(['country', 'tol', 'forecast_day'])['ratio']
               .mean()
               .reset_index()
               .groupby(['country', 'tol'])['ratio']
               .agg(['mean', 'std'])
               .round(3))
print(stability.to_string())
