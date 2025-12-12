# 1. IMPORTS
# -------------------------------
import pandas as pd
import numpy as np
import warnings
import os
from statsmodels.tsa.arima.model import ARIMA

# Ignore convergence warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)


# 2. LOAD DATA AND MODEL PARAMETERS
# -------------------------------
print("--- Loading Data and Model Parameters ---")

# File paths
model_params_file = '../results/arma_best_model_wind.csv'
wind_data_file = '../data/df_wind.csv'

# Load the model parameters saved from the previous step
results_df = pd.read_csv(model_params_file)

# Load the original wind generation data
df = pd.read_csv(wind_data_file)
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y %H:%M')

installed_capacity_2015 = {
    "AT": {"wind": 1981,    "solar":   404}, "BE": {"wind": 2172,    "solar": 3068},
    "BG": {"wind":  701,    "solar": 1041}, "CH": {"wind":    60,    "solar":  756},
    "CZ": {"wind":  277,    "solar": 2067}, "DE": {"wind":43429,    "solar":38411},
    "DK": {"wind": 5082,    "solar":  781}, "EE": {"wind":  301,    "solar":    6},
    "ES": {"wind":23003,    "solar": 6967}, "FI": {"wind": 1082,    "solar":   11},
    "FR": {"wind":10312,    "solar": 6192}, "EL": {"wind": 1775,    "solar": 2444},
    "HR": {"wind":  384,    "solar":   44}, "HU": {"wind":  328,    "solar":   29},
    "IE": {"wind": 2400,    "solar":    1}, "IT": {"wind": 8750,    "solar":19100},
    "LT": {"wind":  290,    "solar":   69}, "LU": {"wind":    60,    "solar":  116},
    "LV": {"wind":   70,    "solar":    2}, "NL": {"wind": 3641,    "solar": 1429},
    "NO": {"wind":  860,    "solar":   14}, "PL": {"wind": 5186,    "solar":   87},
    "PT": {"wind": 4826,    "solar":  429}, "RO": {"wind": 2923,    "solar": 1249},
    "SI": {"wind":     3,    "solar":  263}, "SK": {"wind":     3,    "solar":  532},
    "SE": {"wind": 3029,    "solar":  104}, "UK": {"wind":13563,    "solar": 9000},
}

cols = [c for c in df.columns if c not in ['Date', 'hour']]
valid_countries = [c for c in cols if c in installed_capacity_2015]

for country in valid_countries:
    wind_capacity = installed_capacity_2015[country]['wind']
    df[country] *= wind_capacity
    

# SCENARIO GENERATION FOR MULTIPLE SETS
# ------------------------------------------
# Define the number of scenario sets to generate
num_sets = 11

# Each subsequent set will forecast for the next day
initial_train_date = 270 * 24

# Create the main output directory once
output_dir = 'scenario_results'
os.makedirs(output_dir, exist_ok=True)


# --- Main loop to generate 11 sets of scenarios ---
for set_num in range(num_sets):
# for set_num in range(11, 12):

    current_set = set_num + 1
    print(f"     GENERATING SCENARIO SET {current_set}/{num_sets}")
    print("==============================================")

    # Dictionary to hold scenarios for the current set
    all_forecast_scenarios = {}

    # Slide the training window forward by 1 day (24 hours) for each set
    train_date = initial_train_date + (set_num * 24)

    # --- Inner loop to generate scenarios for each country ---
    for index, row in results_df.iterrows():
        country = row['Country']
        p = int(row['p'])
        q = int(row['q'])

        n_scenarios = 100
        forecast_steps = 24

        print(f"\n--- Processing {country} for set {current_set} ---")

        # Isolate the time series data slice for the current set
        forecast_train_data = df[country].iloc[train_date - 60*24:train_date].astype(float)
        model_fit = ARIMA(forecast_train_data, order=(p, 0, q)).fit()
        simulated_values = model_fit.simulate(nsimulations=forecast_steps, repetitions=n_scenarios)
        scenarios = simulated_values.values

        # Wind generation cannot be negative, so clip values at 0
        scenarios[scenarios < 0] = 0

        # Store the final scenarios in the dictionary for this set
        all_forecast_scenarios[country] = scenarios
        print(f"Generated scenarios for {country}.")


    # 4. SAVE SCENARIOS TO CSV FILES FOR THE CURRENT SET
    # --------------------------------------------------
    if all_forecast_scenarios:
        print("\n--- Saving scenarios to CSV files for this set ---")
        # Loop through the dictionary of stored scenarios
        for country, scenarios in all_forecast_scenarios.items():
            # Create a DataFrame for the scenarios
            column_names = [f'scenario_{i+1}' for i in range(n_scenarios)]
            scenarios_df = pd.DataFrame(scenarios, columns=column_names)

            # Add a 'Forecast_Hour' index, starting from 1
            scenarios_df.index = np.arange(1, len(scenarios_df) + 1)
            scenarios_df.index.name = 'Forecast_Hour'

            # Define a unique output filename for the current set
            filename = os.path.join(output_dir, f'wind_arma_{country}_set_{current_set}.csv')

            # Save to CSV
            scenarios_df.to_csv(filename)
            print(f"   -> Scenarios for {country} saved to '{filename}'")
    else:
        print(f"\nNo scenarios were generated for set {current_set}.")

print("\n\n==============================================")
print("          ALL SCENARIO SETS GENERATED")
print("==============================================")