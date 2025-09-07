import pandas as pd
import os
import numpy as np
from scipy import stats

# This dictionary seems correct.
installed_capacity_2015 = {
    "AT": {"wind": 1981,    "solar":  404},
    "BE": {"wind": 2172,    "solar": 3068},
    "BG": {"wind":  701,    "solar": 1041},
    "CH": {"wind":   60,    "solar":  756},
    "CZ": {"wind":  277,    "solar": 2067},
    "DE": {"wind": 43429,   "solar": 38411},
    "DK": {"wind": 5082,    "solar":  781},
    "EE": {"wind":  301,    "solar":    6},
    "ES": {"wind": 23003,   "solar": 6967},
    "FI": {"wind": 1082,    "solar":   11},
    "FR": {"wind": 10312,   "solar": 6192},
    "EL": {"wind": 1775,    "solar": 2444}, # Note: 'EL' is often used for Greece
    "HR": {"wind":  384,    "solar":   44},
    "HU": {"wind":  328,    "solar":   29},
    "IE": {"wind": 2400,    "solar":    1},
    "IT": {"wind": 8750,    "solar": 19100},
    "LT": {"wind":  290,    "solar":   69},
    "LU": {"wind":   60,    "solar":  116},
    "LV": {"wind":   70,    "solar":    2},
    "NL": {"wind": 3641,    "solar": 1429},
    "NO": {"wind":  860,    "solar":   14},
    "PL": {"wind": 5186,    "solar":   87},
    "PT": {"wind": 4826,    "solar":  429},
    "RO": {"wind": 2923,    "solar": 1249},
    "SI": {"wind":    3,    "solar":  263},
    "SK": {"wind":    3,    "solar":  532},
    "SE": {"wind": 3029,    "solar":  104},
    "UK": {"wind": 13563,   "solar": 9000},
}

# List of country codes from the dictionary keys.
countries = installed_capacity_2015.keys()

# Path to the directory containing your files.
path = "."

# Empty list to store the results.
results = []

print("Starting to process files...")

# --- Main Loop ---
for c in countries:
    filename = os.path.join(path, f"battery_{c}_all_sets_tol_0.05_installed_capacity.csv")

    try:
        df = pd.read_csv(filename)

        if 'ratio' not in df.columns:
            print(f"   Skipping {c}: 'ratio' column not found in {filename}")
            continue

        ratio_data = df['ratio'].dropna()

        # A CI is not meaningful for a single data point, but we can still proceed.
        if len(ratio_data) < 1:
            print(f"   Skipping {c}: No data points available.")
            continue

        # --- Corrected CI Calculation ---
        mean = np.mean(ratio_data)
        sem = stats.sem(ratio_data)

        # Check if sem is 'Not a Number' (from a single data point) or zero.
        if np.isnan(sem) or sem == 0:
            # If so, set the confidence interval to (mean, mean) as requested.
            ci = (mean, mean)
        else:
            # Otherwise, calculate the interval normally.
            ci = stats.t.interval(confidence=0.95,
                                  df=len(ratio_data)-1,
                                  loc=mean,
                                  scale=sem)

        ci_str = f"({ci[0]:.2f}, {ci[1]:.2f})"
        results.append({'country': c, '95% CI for ratio': ci_str})
        print(f"   Processed {c}: CI = {ci_str}")

    except FileNotFoundError:
        print(f"   ERROR: File not found for {c} at '{filename}'")
        continue

# --- Save Results to a New CSV File ---
if results:
    final_df = pd.DataFrame(results)
    output_filename = 'battery_CI_5_installed_capacity.csv'
    final_df.to_csv(output_filename, index=False)

    print("\n------------------")
    print(f"Results saved to '{output_filename}'")
    print("--- Final Data ---")
    print(final_df.to_string())
    print("------------------")
else:
    print("\nNo data was processed. Please check file paths and contents.")