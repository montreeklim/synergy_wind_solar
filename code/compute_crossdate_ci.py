"""Compute cross-date 95% CIs (t_3) for battery 100 MW results and print
CROSS_CI_E05 / CROSS_CI_E01 dicts ready to paste into generate_ci_tables.py."""

import pandas as pd
import numpy as np
from scipy import stats

RESULTS = "../results"
DAYS = [271, 301, 332, 362]

def cross_date_ci(tol_tag):
    frames = []
    for day in DAYS:
        path = f"{RESULTS}/battery_CI_day{day}_tol{tol_tag}_100.csv"
        df = pd.read_csv(path, index_col=0)
        df = df[["mean_ratio"]].rename(columns={"mean_ratio": day})
        frames.append(df)
    wide = pd.concat(frames, axis=1)

    results = {}
    for country in wide.index:
        vals = wide.loc[country].dropna().values
        n = len(vals)
        if n < 2:
            results[country] = None
            continue
        m = vals.mean()
        se = vals.std(ddof=1) / np.sqrt(n)
        t = stats.t.ppf(0.975, df=n - 1)
        lo, hi = m - t * se, m + t * se
        results[country] = (lo, hi)
    return results

for varname, tol_tag in [("CROSS_CI_E05", "05"), ("CROSS_CI_E01", "01")]:
    ci = cross_date_ci(tol_tag)
    print(f"{varname} = {{")
    for country, v in sorted(ci.items()):
        if v is None:
            print(f'    "{country}": None,')
        else:
            lo, hi = v
            print(f'    "{country}": "({lo:.3f},\\\\ {hi:.3f})",')
    print("}")
    print()
