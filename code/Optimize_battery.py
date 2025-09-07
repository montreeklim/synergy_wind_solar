import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import time

wind_costs = np.array([0.0189, 0.0172, 0.0155, 0.0148, 0.0146, 0.0151, 0.0173, 0.0219, 0.0227, 0.0226, 0.0235, 0.0242, 0.0250, 0.0261, 0.0285, 0.0353, 0.0531, 0.0671, 0.0438, 0.0333, 0.0287, 0.0268, 0.0240, 0.0211])

# countries = ['AL']
countries = ['AL', 'AT', 'BA', 'BE', 'BG', 'CH', 'CZ', 'DE', 'DK', 'EE', 'EL', 'ES', 'FI', 'FR', 'HR', 'HU', 'IE', 'IT', 'LT', 'LU', 'LV', 'ME', 'MK', 'NL', 'NO', 'PL', 'PT', 'RO', 'RS', 'SE', 'SI', 'SK', 'UK', 'XK']

# Dictionary to hold DataFrames per country
country_dfs = {}

# Set up parameters for the optimization model
T = range(24)  # hours in a day
tol = 0.01
BigM = 50000

# Example optimization function (as provided)
def run_optimization_and_collect_results(name, data_matrix):
    """
    Build, optimize, and return key results for a given scenario.
    data_matrix should be a 2D numpy array with shape (#days, 24).
    """
    # Transport to shape (N, 24)
    data_matrix = data_matrix.T
    
    # Create a Gurobi model
    model = gp.Model()
    P_ub = 480
    Q_ub = 480
    X_lb = 192
    X_ub = 960
    charge_cost = 0.0256
    discharge_cost = 0.0256
    charge_coef = 0.9
    initial_battery = 480
    
    # Create decision variables: x for each hour, and binary z for each day block
    y = model.addVars(T, lb=0.0, vtype=GRB.CONTINUOUS, name="y")
    W = range(data_matrix.shape[0])
    z = model.addVars(W, vtype=GRB.BINARY, name="z")
    p = model.addVars(W, T, lb = 0, ub = P_ub, vtype=GRB.CONTINUOUS, name="p")
    q = model.addVars(W, T, lb = 0, ub = Q_ub, vtype=GRB.CONTINUOUS, name="q")
    x = model.addVars(W, T, lb = X_lb, ub = X_ub, vtype=GRB.CONTINUOUS, name="x")
    
    # Define a dummy cost objective (replace wind_costs with your cost data)
    # Here we assume wind_costs is a DataFrame with index matching hours and column 'REW'
    prob = 1/data_matrix.shape[0]
    # cost = gp.quicksum(y[t] * wind_costs['REW'].iloc[t] for t in T) - prob*(gp.quicksum(charge_cost*p[w,t] + discharge_cost*q[w,t] for w in W for t in T))
    cost = gp.quicksum(y[t] * wind_costs[t] for t in T) - prob*(gp.quicksum(charge_cost*p[w,t] + discharge_cost*q[w,t] for w in W for t in T))
    model.setObjective(cost, GRB.MAXIMIZE)

    # Capacity constraints: x[t] must be less than available capacity plus BigM if day is "switched"
    model.addConstrs((y[t] - q[w,t] + p[w,t] <= data_matrix.iloc[w, t] + z[w] * BigM
                      for w in W for t in T), name="capacity")
    model.addConstr(gp.quicksum(z[w] for w in W) <= np.floor(len(W) * tol),
                    name="z_tolerance")
    
    # State of charge
    model.addConstrs(x[w, t+1] == x[w,t] + charge_coef*p[w,t] - 1/charge_coef*q[w,t] for w in W for t in T[: -1])
    model.addConstrs(x[w, 0] == initial_battery for w in W)

    # Set a time limit (optional)
    model.Params.TimeLimit = 2100
    # model.Params.OutputFlag = 0
    model.Params.MIPGap = 0.01  # sets relative optimality gap to 1%

    # Optimize the model and collect results
    start_time = time.time()
    model.optimize()
    end_time = time.time()
    
    obj_val = model.objVal
    y_values = np.array([y[t].X for t in T])
    zero_count = np.sum((y_values >= -1e-6) & (y_values <= 1e-6))
    q1 = np.percentile(y_values, 25)
    q2 = np.percentile(y_values, 50)
    q3 = np.percentile(y_values, 75)
    comp_time = end_time - start_time

    return {
        "Method": name,
        "Objective": obj_val,
        "Q1": q1,
        "Q2": q2,
        "Q3": q3,
        "Number of zeros": zero_count,
        "computation time": comp_time
    }



# Assume you already have a dictionary of country-specific DataFrames named country_dfs,
# where each DataFrame has columns: 'Date', 'Hour', 'wind_generation', 'pv_generation', 'combined'.

# Dictionary to store optimization results for each country
country_results = {}
n_scenarios = 500

for country in countries:
    print(f"Processing optimization for country: {country}")
    
    # Here, we assume that the data covers complete days.
    wind = pd.read_csv(f'wind_arima_{country}_{n_scenarios}.csv', skiprows=1, header=None)
    solar = pd.read_csv(f'solar_arma_{country}_{n_scenarios}.csv', header=None)
    wind.columns = solar.columns
    combined = wind + solar
    
    # Run the optimization for each scenario
    res_wind = run_optimization_and_collect_results("wind only", wind)
    res_solar = run_optimization_and_collect_results("pv only", solar)
    res_combined = run_optimization_and_collect_results("combined", combined)
    
    # Collect results in a DataFrame
    results = [res_wind, res_solar, res_combined]
    df_results = pd.DataFrame(results)
    
    # Optionally, print or save the result for each country
    print(f"Results for {country}:")
    print(df_results)
    
    # Save the results DataFrame in the dictionary
    country_results[country] = df_results

    # Optionally, save each country's results to a CSV file
    # df_results.to_csv(f'results_{country}.csv', index=False)


# # If needed, you can also combine all results into a single DataFrame
# all_results = pd.concat([df.assign(country=country) for country, df in country_results.items()])
# all_results.to_csv(f'all_countries_results_battery_{tol}.csv', index=False)

# Print or display the results
# print(df_results)