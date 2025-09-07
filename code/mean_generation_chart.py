import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from matplotlib.ticker import ScalarFormatter

# This script assumes EMHIRES_PV_2015.csv and EMHIRES_wind_2015.csv are in the same directory.

# 1. Load data
try:
    pv = pd.read_csv('EMHIRES_PV_2015.csv', parse_dates=['Date'])
    wind = pd.read_csv('EMHIRES_wind_2015.csv', parse_dates=['Date'])
except FileNotFoundError:
    print("Could not find data files. Please ensure they are in the correct directory.")
    sys.exit()

# Your installed‐capacity dict
installed_capacity_2015 = {
    "AT": {"wind": 1981, "solar": 404}, "BE": {"wind": 2172, "solar": 3068},
    "BG": {"wind": 701, "solar": 1041}, "CH": {"wind": 60, "solar": 756},
    "CZ": {"wind": 277, "solar": 2067}, "DE": {"wind": 43429, "solar": 38411},
    "DK": {"wind": 5082, "solar": 781}, "EE": {"wind": 301, "solar": 6},
    "ES": {"wind": 23003, "solar": 6967}, "FI": {"wind": 1082, "solar": 11},
    "FR": {"wind": 10312, "solar": 6192}, "EL": {"wind": 1775, "solar": 2444},
    "HR": {"wind": 384, "solar": 44}, "HU": {"wind": 328, "solar": 29},
    "IE": {"wind": 2400, "solar": 1}, "IT": {"wind": 8750, "solar": 19100},
    "LT": {"wind": 290, "solar": 69}, "LU": {"wind": 60, "solar": 116},
    "LV": {"wind": 70, "solar": 2}, "NL": {"wind": 3641, "solar": 1429},
    "NO": {"wind": 860, "solar": 14}, "PL": {"wind": 5186, "solar": 87},
    "PT": {"wind": 4826, "solar": 429}, "RO": {"wind": 2923, "solar": 1249},
    "SI": {"wind": 3, "solar": 263}, "SK": {"wind": 3, "solar": 532},
    "SE": {"wind": 3029, "solar": 104}, "UK": {"wind": 13563, "solar": 9000},
}

# 2. Data cleaning
drop_cols = ['Date','Year','Month','Day','Hour','CY']
pv = pv.drop(columns=drop_cols, errors='ignore')
wind = wind.drop(columns=drop_cols, errors='ignore')

# 3. Filter valid countries
if 'EL' in installed_capacity_2015:
    installed_capacity_2015['GR'] = installed_capacity_2015.pop('EL')
    pv.rename(columns={'EL': 'GR'}, inplace=True)
    wind.rename(columns={'EL': 'GR'}, inplace=True)

valid = [c for c, caps in installed_capacity_2015.items() if caps['wind']>0 and caps['solar']>0]
pv = pv[valid]
wind = wind[valid]

# 4. Calculate generation in MW
pv_gen = pv.mul({c: installed_capacity_2015[c]['solar'] for c in valid}, axis=1)
wind_gen = wind.mul({c: installed_capacity_2015[c]['wind'] for c in valid}, axis=1)

# 5. Calculate mean generation
mean_pv = pv_gen.mean()
mean_wind = wind_gen.mean()

# --- DATA PREPARATION FOR PLOTTING ---
# 6. Combine into a single DataFrame
df = pd.DataFrame({'Wind': mean_wind, 'Solar': mean_pv})

# 7. Sort by wind generation for a cleaner plot
df_sorted = df.sort_values('Wind', ascending=True)

# --- NEW STEP: Map ISO codes to full country names for the plot ---
iso_to_name = {
    'AT':'Austria', 'BE':'Belgium', 'BG':'Bulgaria', 'CH':'Switzerland',
    'CZ':'Czech Republic', 'DE':'Germany', 'DK':'Denmark', 'EE':'Estonia',
    'ES':'Spain', 'FI':'Finland', 'FR':'France', 'GR':'Greece', 'HR':'Croatia',
    'HU':'Hungary', 'IE':'Ireland', 'IT':'Italy', 'LT':'Lithuania',
    'LU':'Luxembourg', 'LV':'Latvia', 'NL':'Netherlands', 'NO':'Norway',
    'PL':'Poland', 'PT':'Portugal', 'RO':'Romania', 'SI':'Slovenia',
    'SK':'Slovakia', 'SE':'Sweden', 'UK':'United Kingdom'
}
df_sorted.rename(index=iso_to_name, inplace=True)
# --- END OF NEW STEP ---


# --- IMPROVED FACETED PLOTTING ---
# 8. Create two subplots side-by-side that share the y-axis
fig, axes = plt.subplots(1, 2, figsize=(15, 10), sharey=True)
fig.suptitle('Mean Hourly Generation by Country (2015, MW, log scale)', fontsize=18, fontweight='bold')

# --- Plot 1: Wind Generation ---
axes[0].barh(df_sorted.index, df_sorted['Wind'], color='#3498db')
axes[0].set_xscale('log')
axes[0].set_title('Wind Generation', fontsize=14)
axes[0].set_xlabel('Average Hourly Generation (MW, log scale)', fontsize=12)
axes[0].grid(axis='x', linestyle='--', alpha=0.7)
axes[0].tick_params(axis='y', labelsize=11) # Adjust y-tick font size if needed

# --- Plot 2: Solar Generation ---
axes[1].barh(df_sorted.index, df_sorted['Solar'], color='#f1c40f')
axes[1].set_xscale('log')
axes[1].set_title('PV Generation', fontsize=14)
axes[1].set_xlabel('Average Hourly Generation (MW, log scale)', fontsize=12)
axes[1].grid(axis='x', linestyle='--', alpha=0.7)

# --- General Formatting ---
# Format the x-axis ticks to be more readable than scientific notation
for ax in axes:
    formatter = ScalarFormatter()
    formatter.set_scientific(False)
    ax.xaxis.set_major_formatter(formatter)

plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make room for the suptitle
plt.savefig('faceted_barchart_fullnames.png', dpi=300)
print("Successfully generated and saved 'faceted_barchart_fullnames.png'")
plt.show()