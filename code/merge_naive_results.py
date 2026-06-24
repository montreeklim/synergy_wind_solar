#!/usr/bin/env python3
"""
Merge per-country naive result CSVs produced by run_naive_model.slurm
into a single all_countries_naive_results_rolling.csv.

Run from the project root after all 28 array tasks complete:
    python code/merge_naive_results.py
"""

import glob
import pandas as pd

files = sorted(glob.glob('naive_results_*.csv'))
print(f"Found {len(files)} country files: {[f.split('_')[2].replace('.csv','') for f in files]}")

if not files:
    raise FileNotFoundError("No naive_results_*.csv files found in current directory.")

df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
df = df.sort_values(['forecast_day', 'tol', 'country', 'set_number']).reset_index(drop=True)

out = 'all_countries_naive_results_rolling.csv'
df.to_csv(out, index=False, float_format='%.5f')
print(f"Saved {len(df)} rows to {out}")
print(df.head())
