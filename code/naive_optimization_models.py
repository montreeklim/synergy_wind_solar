import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB

# --- 1. Constants and Initial Setup ---

# Hourly electricity prices/costs
wind_costs = np.array([
    0.0189, 0.0172, 0.0155, 0.0148, 0.0146, 0.0151, 0.0173, 0.0219, 0.0227, 
    0.0226, 0.0235, 0.0242, 0.0250, 0.0261, 0.0285, 0.0353, 0.0531, 0.0671, 
    0.0438, 0.0333, 0.0287, 0.0268, 0.0240, 0.0211
])

# Installed capacity data for each country
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

# --- 2. Experimental Parameters ---
T = range(24)
BigM = 50000
tolerances = [0.01, 0.05]
set_numbers = range(1, 11)  
countries = installed_capacity_2015.keys()

# --- 3. Updated Optimization Function ---

def run_optimization_and_collect_results(name, data_matrix, current_tol):
    """
    Build, optimize, and return key results for a given scenario.
    The tolerance for discarded days is now passed as a parameter.
    """
    data_matrix = data_matrix.T
    model = gp.Model()
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = 2100

    x = model.addVars(T, lb=0.0, vtype=GRB.CONTINUOUS, name="x")
    W = range(data_matrix.shape[0])
    z = model.addVars(W, vtype=GRB.BINARY, name="z")

    profit = gp.quicksum(x[t] * wind_costs[t] for t in T)
    model.setObjective(profit, GRB.MAXIMIZE)

    model.addConstrs((x[t] <= data_matrix.iloc[w, t] + z[w] * BigM
                      for w in W for t in T), name="capacity")
    
    # Use the 'current_tol' parameter passed into the function
    model.addConstr(gp.quicksum(z[w] for w in W) <= np.floor(len(W) * current_tol),
                    name="z_tolerance")

    model.optimize()

    return {
        "Method": name,
        "Objective": model.objVal if model.status == GRB.OPTIMAL else 0,
    }

# --- 4. Main Processing Loop for All Experiments ---

# Master list to store results from every single experimental run
all_experiments_results = []

for tol in tolerances:
    for set_i in set_numbers:
        for country in countries:
            print(f"Processing: tol={tol}, set={set_i}, country={country}")
            
            try:
                # Load the specific dataset for the current set number
                wind = pd.read_csv(f'../data/scenarios/wind_arma_{country}_set_{set_i}.csv', index_col=0)
                solar = pd.read_csv(f'../data/scenarios/pv_arma_{country}_set_{set_i}.csv', index_col=0)
                
                solar.columns = wind.columns
                solar.index = wind.index
                
                wind = np.round(wind, 3)
                solar = np.round(solar, 3)
                combined = wind + solar

                # Run optimization, passing the current 'tol' value
                res_wind = run_optimization_and_collect_results("wind only", wind, tol)
                res_solar = run_optimization_and_collect_results("pv only", solar, tol)
                res_combined = run_optimization_and_collect_results("combined", combined, tol)
                
                # Store the parameters for this run in each result dictionary
                for res in [res_wind, res_solar, res_combined]:
                    res['country'] = country
                    res['set_number'] = set_i
                    res['tol'] = tol
                
                all_experiments_results.extend([res_wind, res_solar, res_combined])

            except FileNotFoundError:
                print(f"  -> SKIPPING: File not found for tol={tol}, set={set_i}, country={country}")
                continue

# --- 5. Data Aggregation and Transformation ---

print("\n--- All experiments complete. Transforming final data... ---")

# Create the initial "long" format DataFrame from the list of all results
all_results_long = pd.DataFrame(all_experiments_results)

# Pivot the table, using the experiment parameters as the new index
wide_df = all_results_long.pivot(index=['tol', 'set_number', 'country'], columns='Method', values='Objective')

# Rename the columns for clarity
wide_df = wide_df.rename(columns={
    'wind only': 'wind objective',
    'pv only': 'solar objective',
    'combined': 'combined objective'
})

# Calculate the synergy ratio
denominator = wide_df['wind objective'] + wide_df['solar objective']
wide_df['ratio'] = wide_df['combined objective'] / denominator
wide_df.replace([np.inf, -np.inf], 0, inplace=True)
wide_df.fillna(0, inplace=True)

# Reset the index to turn 'tol', 'set_number', and 'country' back into columns
final_df = wide_df.reset_index()

# Reorder columns to place experiment parameters first
final_df.columns.name = None
final_df = final_df[[
    'tol', 'set_number', 'country', 'wind objective', 'solar objective', 
    'combined objective', 'ratio'
]]

# --- 6. Display and Save the Final Output ---
print(final_df.head(10))

output_filename = 'all_countries_naive_results.csv'
final_df.to_csv(output_filename, index=False, float_format='%.5f')

print(f"\nSuccessfully saved the final results to {output_filename}")