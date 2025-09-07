import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import time


wind_costs = pd.read_csv('wind_costs.csv', index_col=0)

# Read the CSV files using the appropriate index (e.g., Date or another column)
solar_df = pd.read_csv('df_pv_.csv', index_col=0)
wind_df = pd.read_csv('df_wind.csv', index_col=0)

# Identify columns to multiply (i.e., exclude "Hour")
cols_to_multiply_solar = [col for col in solar_df.columns if col.lower() != 'hour']
cols_to_multiply_wind  = [col for col in wind_df.columns if col.lower() != 'hour']

# Multiply only the selected columns by 1000
# solar_df[cols_to_multiply_solar] = solar_df[cols_to_multiply_solar] * 1000
# wind_df[cols_to_multiply_wind]   = wind_df[cols_to_multiply_wind] * 1000

# Create combined dataframe by summing solar and wind values, excluding 'Date' and 'Hour'
combined_df = solar_df.copy()
countries = solar_df.columns.difference(['Date', 'Hour'])

# Add the values from wind_df to solar_df
combined_df[countries] = solar_df[countries] + wind_df[countries]

# Dictionary to hold DataFrames per country
country_dfs = {}

# Loop through each country and create the corresponding dataframe
for country in countries:
    country_df = pd.DataFrame({
        'Solar': solar_df[country],
        'Wind': wind_df[country],
        'Combined': combined_df[country]
    })
    
    # Optionally save as separate CSV if needed
    # country_df.to_csv(f'df_{country}.csv', index=False)

    # Save in dictionary for easy access
    country_dfs[country] = country_df
    
# Compute wind variance dictionary (grouping by the 'Hour' column)
wind_variance_dict = {}
for country in countries:
    # Group by 'Hour' and compute variance for the current country
    wind_variance = wind_df.groupby('Hour')[country].var()
    wind_variance_dict[country] = wind_variance

# Create a DataFrame from the wind variance dictionary
wind_variance_df = pd.DataFrame(wind_variance_dict)
wind_variance_df.index.name = 'Hour'

# Compute solar variance dictionary (grouping by the 'Hour' column)
solar_variance_dict = {}
for country in countries:
    solar_variance = solar_df.groupby('Hour')[country].var()
    solar_variance_dict[country] = solar_variance

# Create a DataFrame from the solar variance dictionary
solar_variance_df = pd.DataFrame(solar_variance_dict)
solar_variance_df.index.name = 'Hour'

# Plotting the variance for each country
# for country in countries:
#     plt.figure(figsize=(10, 6))
#     plt.plot(wind_variance_df.index, wind_variance_df[country], marker='o', label='Wind Variance')
#     plt.plot(solar_variance_df.index, solar_variance_df[country], marker='s', label='Solar Variance')
#     plt.xlabel('Hour of Day')
#     plt.ylabel('Variance (Energy Units$^2$)')
#     plt.title(f'Hourly Variance of Energy Generation for {country}')
#     plt.legend()
#     plt.grid(True)
#     plt.show()
    
# Loop through each country's DataFrame in the dictionary
for country, df in country_dfs.items():
    # Convert the index to datetime using dayfirst=True
    df.index = pd.to_datetime(df.index, dayfirst=True)
    
    # Group by hour (extract hour from the datetime index) and compute the average
    df_avg = df.groupby(df.index.hour)[['Solar', 'Wind', 'Combined']].mean()
    
    # Create a plot for this country
    plt.figure(figsize=(10, 6))
    plt.plot(df_avg.index, df_avg['Solar'], marker='o', label='Solar')
    plt.plot(df_avg.index, df_avg['Wind'], marker='s', label='Wind')
    plt.plot(df_avg.index, df_avg['Combined'], marker='^', label='Combined')
    
    plt.xlabel('Hour of Day')
    plt.ylabel('Average Energy Generation')
    plt.title(f'Average Energy Generation by Hour for {country}')
    plt.legend()
    plt.grid(True)
    plt.show()
    
# Set up parameters for the optimization model
T = range(24)  # hours in a day
tol = 0.01
BigM = 10000

# Example optimization function (as provided)
def run_optimization_and_collect_results(name, data_matrix):
    """
    Build, optimize, and return key results for a given scenario.
    data_matrix should be a 2D numpy array with shape (#days, 24).
    """
    # print(f'shape of data is {data_matrix.shape}')
    # Create a Gurobi model
    model = gp.Model()

    # Create decision variables: x for each hour, and binary z for each day block
    x = model.addVars(T, lb=0.0, vtype=GRB.CONTINUOUS, name="x")
    W = range(data_matrix.shape[0])
    z = model.addVars(W, vtype=GRB.BINARY, name="z")

    # Define a dummy cost objective (replace wind_costs with your cost data)
    # Here we assume wind_costs is a DataFrame with index matching hours and column 'REW'
    # For illustration, we use a constant cost per hour.
    cost_per_hour = 1.0
    cost = gp.quicksum(x[t] * cost_per_hour for t in T)
    model.setObjective(cost, GRB.MAXIMIZE)

    # Capacity constraints: x[t] must be less than available capacity plus BigM if day is "switched"
    model.addConstrs((x[t] <= data_matrix[w, t] + z[w] * BigM
                      for w in W for t in T), name="capacity")
    model.addConstr(gp.quicksum(z[w] for w in W) <= np.floor(len(W) * tol),
                    name="z_tolerance")

    # Set a time limit (optional)
    model.Params.TimeLimit = 2100
    model.Params.OutputFlag = 0

    # Optimize the model and collect results
    start_time = time.time()
    model.optimize()
    end_time = time.time()
    
    obj_val = model.objVal
    x_values = np.array([x[t].X for t in T])
    zero_count = np.sum((x_values >= -1e-6) & (x_values <= 1e-6))
    q1 = np.percentile(x_values, 25)
    q2 = np.percentile(x_values, 50)
    q3 = np.percentile(x_values, 75)
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

for country, df_country in country_dfs.items():
    print(f"Processing optimization for country: {country}")
    
    # print(df_country)
    # Ensure the data is sorted by 'Date' 
    df_sorted = df_country.sort_values(['Date']).reset_index(drop=True)
    
    # Reshape data so that each row corresponds to one day (24 hours)
    # Here, we assume that the data covers complete days.
    wind = np.array(df_sorted['Wind']).reshape(-1, 24)
    solar = np.array(df_sorted['Solar']).reshape(-1, 24)
    combined = np.array(df_sorted['Combined']).reshape(-1, 24)
    
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

# If needed, you can also combine all results into a single DataFrame
all_results = pd.concat([df.assign(country=country) for country, df in country_results.items()])
all_results.to_csv(f'Historical_data_results_{tol}.csv', index=False)

# Print or display the results
print(df_results)