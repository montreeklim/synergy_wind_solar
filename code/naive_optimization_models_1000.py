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
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--index', type=int, default=None,
                    help='Country index 0-27 (omit to run all countries sequentially)')
args = parser.parse_args()

T = range(24)
tolerances = [0.01, 0.05, 0.10]
set_numbers = range(1, 11)
forecast_days = [271, 301, 332, 362]

all_countries = list(installed_capacity_2015.keys())
if args.index is not None:
    countries = [all_countries[args.index]]
else:
    countries = all_countries

# --- 3. Optimization Function with Tight Per-(w,t) Big-M ---

def run_optimization_and_collect_results(name, data_matrix, current_tol):
    """
    Build, optimize, and return key results for a given scenario matrix.
    Uses scenario-specific Big-M: M[w,t] = max_gen[t] - data[w,t],
    which tightens the LP relaxation compared to a global constant.
    """
    data_matrix = data_matrix.T          # shape: (n_scenarios, 24)
    model = gp.Model()
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = 3600

    x = model.addVars(T, lb=0.0, vtype=GRB.CONTINUOUS, name="x")
    W = range(data_matrix.shape[0])
    z = model.addVars(W, vtype=GRB.BINARY, name="z")

    profit = gp.quicksum(x[t] * wind_costs[t] for t in T)
    model.setObjective(profit, GRB.MAXIMIZE)

    # Per-hour maximum generation across all scenarios (upper bound on x[t])
    max_gen = data_matrix.max(axis=0)   # shape: (24,)

    # Tight Big-M: when z[w]=1 (scenario discarded), x[t] is relaxed only to max_gen[t]
    model.addConstrs(
        (x[t] <= data_matrix.iloc[w, t]
                 + z[w] * float(max_gen.iloc[t] - data_matrix.iloc[w, t])
         for w in W for t in T),
        name="capacity"
    )

    model.addConstr(
        gp.quicksum(z[w] for w in W) <= np.floor(len(W) * current_tol),
        name="z_tolerance"
    )

    model.optimize()

    if model.status == GRB.OPTIMAL:
        obj = model.objVal
    else:
        status_codes = {
            GRB.INFEASIBLE: "INFEASIBLE",
            GRB.TIME_LIMIT: "TIME_LIMIT",
            GRB.SUBOPTIMAL: "SUBOPTIMAL",
            GRB.UNBOUNDED: "UNBOUNDED",
        }
        status_str = status_codes.get(model.status, f"STATUS_{model.status}")
        print(f"  WARNING: [{name}] solver did not reach optimality — {status_str}")
        obj = model.objVal if model.status == GRB.SUBOPTIMAL else 0

    return {
        "Method": name,
        "Objective": obj,
    }

# --- 4. Main Processing Loop: Pool All 10 Sets per Country ---

all_experiments_results = []

for forecast_day in forecast_days:
    scen_dir = f'scenario_results/day_{forecast_day}'
    for tol in tolerances:
        for country in countries:
            print(f"Processing: day={forecast_day}, tol={tol}, country={country}")

            try:
                wind_parts, solar_parts = [], []
                for set_i in set_numbers:
                    w = pd.read_csv(f'{scen_dir}/wind_arma_{country}_set_{set_i}.csv', index_col=0)
                    s = pd.read_csv(f'{scen_dir}/pv_arma_{country}_set_{set_i}.csv', index_col=0)
                    s.columns = w.columns
                    s.index = w.index
                    wind_parts.append(w)
                    solar_parts.append(s)

                wind = pd.concat(wind_parts, axis=1)    # 24 rows × 1000 columns
                solar = pd.concat(solar_parts, axis=1)
                wind.columns = range(len(wind.columns)) # deduplicate column names
                solar.columns = wind.columns

                wind = np.round(wind, 3)
                solar = np.round(solar, 3)
                combined = wind + solar

                res_wind     = run_optimization_and_collect_results("wind only", wind, tol)
                res_solar    = run_optimization_and_collect_results("pv only",   solar, tol)
                res_combined = run_optimization_and_collect_results("combined",  combined, tol)

                for res in [res_wind, res_solar, res_combined]:
                    res['country'] = country
                    res['tol'] = tol
                    res['forecast_day'] = forecast_day

                all_experiments_results.extend([res_wind, res_solar, res_combined])

            except FileNotFoundError:
                print(f"  -> SKIPPING: day={forecast_day}, tol={tol}, country={country}")
                continue

# --- 5. Data Aggregation and Transformation ---

print("\n--- All experiments complete. Transforming final data... ---")

all_results_long = pd.DataFrame(all_experiments_results)

wide_df = all_results_long.pivot(
    index=['forecast_day', 'tol', 'country'],
    columns='Method', values='Objective'
)

wide_df = wide_df.rename(columns={
    'wind only': 'wind objective',
    'pv only':   'solar objective',
    'combined':  'combined objective'
})

denominator = wide_df['wind objective'] + wide_df['solar objective']
wide_df['ratio'] = wide_df['combined objective'] / denominator
wide_df.replace([np.inf, -np.inf], 0, inplace=True)
wide_df.fillna(0, inplace=True)

final_df = wide_df.reset_index()
final_df.columns.name = None
final_df = final_df[[
    'forecast_day', 'tol', 'country',
    'wind objective', 'solar objective', 'combined objective', 'ratio'
]]

# --- 6. Save Output ---

print(final_df.head(10))

if args.index is not None:
    output_filename = f'naive_results_1000_{all_countries[args.index]}.csv'
else:
    output_filename = 'all_countries_naive_results_1000.csv'
final_df.to_csv(output_filename, index=False, float_format='%.5f')

print(f"\nSuccessfully saved the final results to {output_filename}")
