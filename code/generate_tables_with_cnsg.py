"""
Generate updated LaTeX tables (Tables 2-7) with two sub-columns per forecast date:
  SR   : 95% CI of the Synergy Ratio  (existing content)
  CNSG : 95% CI of the net cooperation gain / total installed capacity  (new column)
         = CI of (combined - wind - solar) / (wind_cap + pv_cap)  [×10^-3]

Tables 2-4: Naive model (section_6_absolute_profits.tex)
Tables 5-7: Storage-Enhanced model (section_5_rolling_window.tex)
"""
import pandas as pd
import numpy as np
import scipy.stats as st
import os

BASE = r"C:\Users\montr\OneDrive - University of Southampton\wind_pv_synergy"

# Country order matching ISO alphabetical order of display codes
# (UK→GB between FR and GR; EL→GR after GB; SE before SI/SK)
COUNTRIES_ISO = ["AT","BE","BG","CH","CZ","DE","DK","EE","ES","FI","FR","UK","EL",
                 "HR","HU","IE","IT","LT","LU","LV","NL","NO","PL","PT","RO",
                 "SE","SI","SK"]
DISPLAY = {"EL": "GR", "UK": "GB"}  # display remapping for paper
DAYS = [271, 301, 332, 362]
TOLS = {"01": 0.01, "05": 0.05, "10": 0.10}

INSTALLED_CAP = {
    "AT": 1981+404,  "BE": 2172+3068, "BG": 701+1041,  "CH": 60+756,
    "CZ": 277+2067,  "DE": 43429+38411,"DK": 5082+781,  "EE": 301+6,
    "ES": 23003+6967,"FI": 1082+11,   "FR": 10312+6192, "EL": 1775+2444,
    "HR": 384+44,    "HU": 328+29,    "IE": 2400+1,     "IT": 8750+19100,
    "LT": 290+69,    "LU": 60+116,    "LV": 70+2,       "NL": 3641+1429,
    "NO": 860+14,    "PL": 5186+87,   "PT": 4826+429,   "RO": 2923+1249,
    "SI": 3+263,     "SK": 3+532,     "SE": 3029+104,   "UK": 13563+9000,
}

MIN_VALID = 6  # fewer than this → CI reported as ---

# ---------- helpers ----------

def ci95(values):
    arr = np.array(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    n = len(arr)
    if n < MIN_VALID:
        return None
    mean = arr.mean()
    sem = st.sem(arr)
    if sem == 0.0:
        return (mean, mean, mean)
    lo, hi = st.t.interval(0.95, df=n-1, loc=mean, scale=sem)
    return (lo, hi, mean)


def fmt_sr_naive(c, day, tol_str):
    key = (c, day, tol_str)
    entry = naive_sr.get(key)
    if entry is None:
        return "---"
    lo, hi, mean = entry
    flag = r"^\ast" if mean >= 5 else ""
    return f"$({lo:.2f},\\ {hi:.2f}){flag}$"


def fmt_sr_battery(c, day, tol_str):
    key = (c, day, tol_str)
    entry = batt_sr.get(key)
    if entry is None:
        return "---"
    lo, hi, mean = entry
    flag = r"^\ast" if mean >= 5 else ""
    return f"$({lo:.2f},\\ {hi:.2f}){flag}$"


def fmt_cnsg(entry):
    """Format CNSG 95% CI ×1000 with 2 d.p.  entry is (lo, hi, mean) or None."""
    if entry is None:
        return "---"
    lo, hi, _ = entry
    return f"$({lo*1000:.2f},\\ {hi*1000:.2f})$"


# ---------- 1. Load naive SR CIs ----------
naive_sr = {}
for tag, tol in TOLS.items():
    for day in DAYS:
        fname = os.path.join(BASE, "results", f"naive_CI_day{day}_tol{tag}.csv")
        df = pd.read_csv(fname)
        for _, row in df.iterrows():
            c = row["country"]
            lo, hi = row["ci_lower"], row["ci_upper"]
            try:
                mean_r = float(row["mean_ratio"])
            except (ValueError, TypeError):
                naive_sr[(c, day, tag)] = None
                continue
            try:
                lo_f, hi_f = float(lo), float(hi)
            except (ValueError, TypeError):
                naive_sr[(c, day, tag)] = None
                continue
            if pd.isna(lo_f) or mean_r == 0.0:
                naive_sr[(c, day, tag)] = None
            else:
                naive_sr[(c, day, tag)] = (lo_f, hi_f, mean_r)


# ---------- 2. Load battery SR CIs ----------
batt_sr = {}
for tag, tol in TOLS.items():
    for day in DAYS:
        for c in COUNTRIES_ISO:
            fname = os.path.join(
                BASE, "battery_results", "100_installed_capacity",
                f"battery_{c}_day_{day}_tol_{tol}_100_installed_capacity.csv",
            )
            if not os.path.exists(fname):
                continue
            df_b = pd.read_csv(fname)
            result = ci95(df_b["ratio"].values)
            if result is not None:
                batt_sr[(c, day, tag)] = result


# ---------- 3. Compute CNSG ----------
naive_raw = pd.read_excel(os.path.join(BASE, "results", "all_countries_naive_results.xlsx"))
batt_raw  = pd.read_excel(os.path.join(BASE, "results", "all_countries_battery_results.xlsx"))

naive_raw["net_gain"] = (naive_raw["combined objective"]
                         - naive_raw["wind objective"]
                         - naive_raw["solar objective"])
batt_raw["net_gain"]  = (batt_raw["combine objective"]
                         - batt_raw["wind objective"]
                         - batt_raw["solar objective"])

def build_cnsg(df):
    """Return dict (country, day, tol) -> (lo, hi, mean) CI of CNSG across sets."""
    idx = {}
    for (country, day, tol), grp in df.groupby(["country", "forecast_day", "tol"]):
        cap = INSTALLED_CAP[country]
        vals = grp["net_gain"].values / cap
        result = ci95(vals)
        idx[(country, int(day), round(tol, 2))] = result
    return idx

naive_cnsg = build_cnsg(naive_raw)
batt_cnsg  = build_cnsg(batt_raw)


# ---------- 4. Build row strings ----------

def naive_row(c, tag):
    tol = TOLS[tag]
    disp = DISPLAY.get(c, c)
    cells = []
    for day in DAYS:
        sr   = fmt_sr_naive(c, day, tag)
        cnsg = fmt_cnsg(naive_cnsg.get((c, day, tol)))
        cells.append(f"{sr} & {cnsg}")
    return f"    {disp} & " + " & ".join(cells) + r" \\"


def battery_row(c, tag):
    tol = TOLS[tag]
    disp = DISPLAY.get(c, c)
    cells = []
    for day in DAYS:
        sr   = fmt_sr_battery(c, day, tag)
        cnsg = fmt_cnsg(batt_cnsg.get((c, day, tol)))
        cells.append(f"{sr} & {cnsg}")
    return f"    {disp} & " + " & ".join(cells) + r" \\"


# ---------- 5. Emit rows for all 6 tables ----------
lines = []

for model, tol_tag in [("Naive", "01"), ("Naive", "05"), ("Naive", "10"),
                        ("Battery", "01"), ("Battery", "05"), ("Battery", "10")]:
    lines.append(f"\n% ---- Table: {model} eps={TOLS[tol_tag]} ----")
    for c in COUNTRIES_ISO:
        if model == "Naive":
            lines.append(naive_row(c, tol_tag))
        else:
            lines.append(battery_row(c, tol_tag))

output = "\n".join(lines)
print(output)

out_path = os.path.join(BASE, "writing", "ci_cnsg_table_output.txt")
with open(out_path, "w") as f:
    f.write(output)
print(f"\nRows written to {out_path}")
