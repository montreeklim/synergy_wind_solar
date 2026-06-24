"""
Generate updated LaTeX table bodies for Tables 2-5, replacing per-date mean SR
with 95% CI over 10 runs (t-distribution, nu=9).

Tables 2-3: Naive model (section_6_absolute_profits.tex)
Tables 4-5: Battery model (section_5_rolling_window.tex)

Outputs LaTeX row content to stdout and to writing/ci_table_output.txt.
"""
import pandas as pd
import numpy as np
import scipy.stats as st
import os
import sys

BASE = r"C:\Users\montr\OneDrive - University of Southampton\wind_pv_synergy"

COUNTRIES_ISO = ["AT","BE","BG","CH","CZ","DE","DK","EE","ES","FI","FR","EL",
                 "HR","HU","IE","IT","LT","LU","LV","NL","NO","PL","PT","RO",
                 "SI","SK","SE","UK"]

DAYS = [271, 301, 332, 362]

# Country order in battery tables (sorted by descending cross-date mean SR at eps=0.05)
COUNTRIES_BATTERY = ["BE","DE","LT","LU","BG","RO","NL","UK","FR","EL",
                     "AT","HR","CZ","ES","DK","IT","HU","PT","CH","LV",
                     "SE","EE","PL","SI","FI","NO","SK","IE"]

# Current cross-date CIs (keep as-is, computed from 4 per-date means via t_3)
CROSS_CI_E05 = {
    "BE":"(1.306,\\ 1.507)", "DE":"(1.284,\\ 1.453)", "LT":"(1.262,\\ 1.459)",
    "LU":"(1.145,\\ 1.522)", "BG":"(1.288,\\ 1.376)", "RO":"(1.253,\\ 1.346)",
    "NL":"(1.169,\\ 1.424)", "UK":"(1.164,\\ 1.349)", "FR":"(1.128,\\ 1.372)",
    "EL":"(1.168,\\ 1.301)", "AT":"(1.139,\\ 1.321)", "HR":"(1.179,\\ 1.275)",
    "CZ":"(1.058,\\ 1.373)", "ES":"(1.163,\\ 1.267)", "DK":"(1.040,\\ 1.346)",
    "IT":"(1.076,\\ 1.287)", "HU":"(1.143,\\ 1.201)", "PT":"(1.061,\\ 1.162)",
    "CH":"(1.057,\\ 1.109)", "LV":"(1.022,\\ 1.113)", "SE":"(1.012,\\ 1.090)",
    "EE":"(1.011,\\ 1.082)", "PL":"(1.020,\\ 1.068)", "SI":"(1.010,\\ 1.026)",
    "FI":"(1.001,\\ 1.028)", "NO":"(1.001,\\ 1.015)", "SK":"(1.004,\\ 1.011)",
    "IE": None,
}
CROSS_CI_E01 = {
    "BE":"(1.194,\\ 1.742)", "DE":"(1.158,\\ 1.604)", "LT":"(1.320,\\ 1.451)",
    "LU":"(1.101,\\ 1.545)", "BG":"(1.235,\\ 1.349)", "RO":"(1.273,\\ 1.383)",
    "NL":"(1.322,\\ 1.476)", "UK":"(1.285,\\ 1.524)", "FR":"(1.128,\\ 1.377)",
    "EL":"(1.141,\\ 1.254)", "AT":"(1.264,\\ 1.388)", "HR":"(1.136,\\ 1.265)",
    "CZ":"(1.068,\\ 1.354)", "ES":"(1.147,\\ 1.245)", "DK":"(1.141,\\ 1.313)",
    "IT":"(1.068,\\ 1.230)", "HU":"(1.158,\\ 1.204)", "PT":"(1.125,\\ 1.214)",
    "CH":"(1.053,\\ 1.091)", "LV":"(1.062,\\ 1.093)", "SE":"(1.028,\\ 1.112)",
    "EE":"(1.030,\\ 1.081)", "PL":"(1.038,\\ 1.063)", "SI":"(1.009,\\ 1.031)",
    "FI":"(1.005,\\ 1.041)", "NO":"(1.008,\\ 1.029)", "SK":"(1.005,\\ 1.012)",
    "IE": None,
}

def ci95(values):
    """95% CI using t-distribution (nu = n-1). Returns (lo, hi, mean)."""
    arr = np.array(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    n = len(arr)
    if n < 2:
        return (np.nan, np.nan, np.nan)
    mean = arr.mean()
    sem = st.sem(arr)
    if sem == 0.0:
        # zero variance: degenerate point CI
        return (mean, mean, mean)
    lo, hi = st.t.interval(0.95, df=n-1, loc=mean, scale=sem)
    return (lo, hi, mean)


# ---------- 1. Load naive CIs from pre-computed CSV files ----------
TOL_FILE = {0.05: '05', 0.01: '01'}
naive_ci = {}  # (country, day, tol) -> (lo, hi, mean) or None
for tol, tol_str in TOL_FILE.items():
    for day in DAYS:
        fname = os.path.join(BASE, 'results', f'naive_CI_day{day}_tol{tol_str}.csv')
        df = pd.read_csv(fname)
        for _, row in df.iterrows():
            c = row['country']
            mean_r = float(row['mean_ratio'])
            lo = row['ci_lower']
            hi = row['ci_upper']
            if pd.isna(lo) or mean_r == 0.0:
                naive_ci[(c, day, tol)] = None
            else:
                naive_ci[(c, day, tol)] = (float(lo), float(hi), mean_r)


# ---------- 2. Load battery results and compute per-date CIs ----------
battery_ci = {}  # (country, day, tol) -> (lo, hi, mean)
for tol in [0.01, 0.05]:
    for day in DAYS:
        for c in COUNTRIES_ISO:
            fname = os.path.join(
                BASE, 'battery_results', '100_installed_capacity',
                f'battery_{c}_day_{day}_tol_{tol}_100_installed_capacity.csv'
            )
            if not os.path.exists(fname):
                print(f"MISSING: {fname}", file=sys.stderr)
                continue
            df_b = pd.read_csv(fname)
            lo, hi, mean_r = ci95(df_b['ratio'].values)
            battery_ci[(c, day, tol)] = (lo, hi, mean_r)


# ---------- 3. Formatting helpers ----------

def fmt_naive(c, day, tol):
    """Format a naive-model CI cell (2 d.p.)."""
    key = (c, day, tol)
    entry = naive_ci.get(key)
    if entry is None:
        return '---'
    lo, hi, mean = entry
    if pd.isna(lo):
        return '---'
    flag = r'^\dagger' if mean >= 5 else ''
    return f'$({lo:.2f},\\ {hi:.2f}){flag}$'


def fmt_battery(c, day, tol):
    """Format a battery-model CI cell (3 d.p.).
    IE always returns SR=1.000 with zero variance — shown as --- per table convention."""
    key = (c, day, tol)
    entry = battery_ci.get(key)
    if entry is None:
        return '---'
    lo, hi, mean = entry
    if pd.isna(lo):
        return '---'
    # IE: degenerate 1.000 case, keep --- convention from current tables
    if c == 'IE' and abs(mean - 1.0) < 1e-6 and abs(hi - lo) < 1e-9:
        return r'\multicolumn{1}{c}{---}'
    return f'$({lo:.3f},\\ {hi:.3f})$'


# ---------- 4. Build table rows ----------

lines = []

def section(title):
    lines.append('')
    lines.append(f'% {"=" * 60}')
    lines.append(f'%  {title}')
    lines.append(f'% {"=" * 60}')
    lines.append('')

# ---- Table 2: Naive eps=0.05 ----
section('TABLE 2 (tab:naive_e05): Naive model, eps = 0.05')
lines.append(r'% \midrule rows — paste into tab:naive_e05')
for c in COUNTRIES_ISO:
    d271 = fmt_naive(c, 271, 0.05)
    d301 = fmt_naive(c, 301, 0.05)
    d332 = fmt_naive(c, 332, 0.05)
    d362 = fmt_naive(c, 362, 0.05)
    lines.append(f'    {c} & {d271} & {d301} & {d332} & {d362} \\\\')

# ---- Table 3: Naive eps=0.01 ----
section('TABLE 3 (tab:naive_e01): Naive model, eps = 0.01')
lines.append(r'% \midrule rows — paste into tab:naive_e01')
for c in COUNTRIES_ISO:
    d271 = fmt_naive(c, 271, 0.01)
    d301 = fmt_naive(c, 301, 0.01)
    d332 = fmt_naive(c, 332, 0.01)
    d362 = fmt_naive(c, 362, 0.01)
    lines.append(f'    {c} & {d271} & {d301} & {d332} & {d362} \\\\')

# ---- Table 4: Battery eps=0.05 ----
section('TABLE 4 (tab:rolling_e05): Battery, eps = 0.05')
lines.append(r'% \midrule rows — paste into tab:rolling_e05')
for c in COUNTRIES_BATTERY:
    d271 = fmt_battery(c, 271, 0.05)
    d301 = fmt_battery(c, 301, 0.05)
    d332 = fmt_battery(c, 332, 0.05)
    d362 = fmt_battery(c, 362, 0.05)
    xci = CROSS_CI_E05[c]
    if xci is None:
        xci_col = r'\multicolumn{1}{c}{---}'
    else:
        xci_col = xci
    lines.append(f'    {c} & {d271} & {d301} & {d332} & {d362} & {xci_col} \\\\')

# ---- Table 5: Battery eps=0.01 ----
section('TABLE 5 (tab:rolling_e01): Battery, eps = 0.01')
lines.append(r'% \midrule rows — paste into tab:rolling_e01')
for c in COUNTRIES_BATTERY:
    d271 = fmt_battery(c, 271, 0.01)
    d301 = fmt_battery(c, 301, 0.01)
    d332 = fmt_battery(c, 332, 0.01)
    d362 = fmt_battery(c, 362, 0.01)
    xci = CROSS_CI_E01[c]
    if xci is None:
        xci_col = r'\multicolumn{1}{c}{---}'
    else:
        xci_col = xci
    lines.append(f'    {c} & {d271} & {d301} & {d332} & {d362} & {xci_col} \\\\')

output = '\n'.join(lines)
print(output)

out_path = os.path.join(BASE, 'writing', 'ci_table_output.txt')
with open(out_path, 'w') as f:
    f.write(output)
print(f'\nOutput written to {out_path}', file=sys.stderr)
