import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import argparse
import os

# --- 1. Constants and Initial Setup ---

installed_capacity_2015 = {
    "AT": {"wind": 1981, "solar": 404},
    "BE": {"wind": 2172, "solar": 3068},
    "BG": {"wind": 701, "solar": 1041},
    "CH": {"wind": 60, "solar": 756},
    "CZ": {"wind": 277, "solar": 2067},
    "DE": {"wind": 43429, "solar": 38411},
    "DK": {"wind": 5082, "solar": 781},
    "EE": {"wind": 301, "solar": 6},
    "ES": {"wind": 23003, "solar": 6967},
    "FI": {"wind": 1082, "solar": 11},
    "FR": {"wind": 10312, "solar": 6192},
    "EL": {"wind": 1775, "solar": 2444},
    "HR": {"wind": 384, "solar": 44},
    "HU": {"wind": 328, "solar": 29},
    "IE": {"wind": 2400, "solar": 1},
    "IT": {"wind": 8750, "solar": 19100},
    "LT": {"wind": 290, "solar": 69},
    "LU": {"wind": 60, "solar": 116},
    "LV": {"wind": 70, "solar": 2},
    "NL": {"wind": 3641, "solar": 1429},
    "NO": {"wind": 860, "solar": 14},
    "PL": {"wind": 5186, "solar": 87},
    "PT": {"wind": 4826, "solar": 429},
    "RO": {"wind": 2923, "solar": 1249},
    "SI": {"wind": 3, "solar": 263},
    "SK": {"wind": 3, "solar": 532},
    "SE": {"wind": 3029, "solar": 104},
    "UK": {"wind": 13563, "solar": 9000},
}

# Define a list of countries and get country from command-line argument
countries = list(installed_capacity_2015.keys())
parser = argparse.ArgumentParser(description='Process a specific country across 10 datasets.')
parser.add_argument('--index', type=int, required=True, help='Index of the country in the list')
args = parser.parse_args()
country = countries[args.index]

# Hourly electricity prices/costs
wind_costs = np.array([
    0.0189, 0.0172, 0.0155, 0.0148, 0.0146, 0.0151, 0.0173, 0.0219, 0.0227, 
    0.0226, 0.0235, 0.0242, 0.0250, 0.0261, 0.0285, 0.0353, 0.0531, 0.0671, 
    0.0438, 0.0333, 0.0287, 0.0268, 0.0240, 0.0211
])

# Model parameters
T = range(24)  # hours in a day
tol = 0.05

# --- 2. Optimization Function ---

def run_optimization_and_collect_results(name, data_matrix, capacity = 1000):
    """
    Builds, optimizes, and returns key results for a given scenario.
    """
    # --- Battery Parameters ---
    X_ub = capacity  # Battery capacity upper bound
    P_ub = X_ub / 2
    Q_ub = X_ub / 2
    X_lb = X_ub / 5
    # BigM = X_ub * 10
    
    charge_cost = 0.0256
    discharge_cost = 0.0256
    charge_coef = 0.9
    initial_battery = X_ub / 2

    data_matrix = data_matrix.T
    W = range(data_matrix.shape[0])
    
    # Calculate BigM
    min_data = pd.Series(index=T, dtype=float)

    for t in T:
        sorted_data = data_matrix.iloc[:, t].sort_values()
        min_data[t] = sorted_data.iloc[int(np.floor(len(W) * tol))]

    BigM = pd.DataFrame(min(Q_ub, charge_coef*(X_ub - X_lb)) - data_matrix.values + min_data.values, index=W, columns=T)
    
    # --- Gurobi Model ---
    model = gp.Model()
    W = range(data_matrix.shape[0])

    # Variables
    y = model.addVars(T, lb=0.0, vtype=GRB.CONTINUOUS, name="y")
    z = model.addVars(W, vtype=GRB.BINARY, name="z")
    p = model.addVars(W, T, lb=0, ub=P_ub, vtype=GRB.CONTINUOUS, name="p")
    q = model.addVars(W, T, lb=0, ub=Q_ub, vtype=GRB.CONTINUOUS, name="q")
    x = model.addVars(W, T, lb=X_lb, ub=X_ub, vtype=GRB.CONTINUOUS, name="x")

    # Objective Function
    prob = 1.0 / data_matrix.shape[0]
    cost = (gp.quicksum(y[t] * wind_costs[t] for t in T)
            - prob * gp.quicksum(charge_cost*p[w,t] + discharge_cost*q[w,t]
                                for w in W for t in T))
    model.setObjective(cost, GRB.MAXIMIZE)

    # Constraints
    model.addConstrs((y[t] - q[w,t] + p[w,t] <= data_matrix.iloc[w, t] + BigM.iloc[w, t] * z[w]
                     for w in W for t in T), name="capacity")
    model.addConstr(gp.quicksum(z[w] for w in W) <= np.floor(len(W) * tol), name="z_tolerance")
    
    # State of Charge Constraints
    model.addConstrs((x[w, t+1] == x[w,t] + charge_coef*p[w,t] - (1/charge_coef)*q[w,t]
                     for w in W for t in T[:-1]), name="soc")
    model.addConstrs((x[w, 0] == initial_battery for w in W), name="init")

    # Optimizer settings
    model.Params.TimeLimit = 3600
    model.Params.MIPGap = 0.01
    model.Params.OutputFlag = 0 # Suppress Gurobi output in the console
    
    model.optimize()

    return {
        "Method": name,
        "Objective": model.objVal,
    }

# --- 3. Main Processing Loop ---

print(f"Starting battery optimization for country: {country}")

# A list to store the results from each of the 10 dataset runs
all_run_results = []
set_numbers = range(1, 11) # Loop from 1 to 10

for i in set_numbers:
    print(f"  > Processing dataset set_{i}...")
    try:
        # Load the specific dataset for the current set number
        wind = pd.read_csv(f'scenario_results/wind_arma_{country}_set_{i}.csv', index_col=0)
        solar = pd.read_csv(f'scenario_results/pv_arma_{country}_set_{i}.csv', index_col=0)

        # Align data and create combined set
        solar.columns = wind.columns
        solar.index = wind.index
        wind = np.round(wind, 3)
        solar = np.round(solar, 3)
        combined = wind + solar

        # Run optimization for each method
        res_wind = run_optimization_and_collect_results("wind only", wind, capacity = 100)
        res_solar = run_optimization_and_collect_results("pv only", solar, capacity = 100)
        res_combined = run_optimization_and_collect_results("combined", combined, capacity = 100)

        # Collect results for this set into a DataFrame
        results_this_set = [res_wind, res_solar, res_combined]
        df_this_set = pd.DataFrame(results_this_set)
        
        # Add the set number to identify which run these results came from
        df_this_set['set_number'] = i
        
        # Add the results to our master list
        all_run_results.append(df_this_set)

    except FileNotFoundError:
        print(f"    - SKIPPING: File not found for set_{i}. Please check the path.")
        continue

# --- 4. Aggregate and Save Final Results ---

if all_run_results:
    # Concatenate all the individual DataFrames into a single one
    df = pd.concat(all_run_results, ignore_index=True)
    
    # Reorder columns for clarity
    df = df[['set_number', 'Method', 'Objective']]
    wide_df = df.pivot(index='set_number', columns='Method', values='Objective')


    # 3. Rename the columns to match your desired output
    wide_df = wide_df.rename(columns={
        'wind only': 'wind objective',
        'pv only': 'solar objective',
        'combined': 'combine objective'
        })


    # 4. Calculate the 'ratio' column
    # The ratio is the combined result divided by the sum of the individual results.
    wide_df['ratio'] = wide_df['combine objective'] / (wide_df['wind objective'] + wide_df['solar objective'])


    # 5. Round all values to 3 decimal places, and the ratio to 2, as in your example
    numerical_cols = ['wind objective', 'solar objective', 'combine objective']
    wide_df[numerical_cols] = wide_df[numerical_cols].round(3)
    wide_df['ratio'] = wide_df['ratio'].round(2)


    # 6. Reset the index to turn 'set_number' back into a column
    final_df = wide_df.reset_index()

    # Clean up the column axis name left by the pivot operation
    final_df.columns.name = None

    # Display the aggregated results
    print("\n--- All runs complete. Aggregated Results: ---")
    print(final_df)

    # Save the final DataFrame to a single CSV file
    output_filename = f'battery_{country}_all_sets_tol_{tol}_100.csv'
    final_df.to_csv(output_filename, index=False)
    
    print(f"\nSuccessfully saved all results to '{output_filename}'")
else:
    print("\nNo results were generated. Please check file paths and ensure data exists.")