import pandas as pd
import numpy as np
import argparse
from statsmodels.tsa.arima.model import ARIMA
import warnings, logging, os

# Suppress warnings and configure logging
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s — %(message)s")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--forecast_day", type=int, default=271,
    help="1-indexed day of year to forecast (must be > WINDOW=270). "
         "E.g. 271=Sep 28, 301=Oct 28, 332=Nov 28, 362=Dec 28."
)
args = parser.parse_args()
if args.forecast_day <= 270:
    raise ValueError("forecast_day must be > 270 so a full 270-day PV training window fits within the data.")

df = pd.read_csv("data/df_pv.csv")
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
df["hour"] = df["Date"].dt.hour

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

# Filter and scale the data
cols = [c for c in df.columns if c not in ["Date", "hour"]]
valid_countries = [c for c in cols if c in installed_capacity_2015]
df = df[["Date", "hour"] + valid_countries]
for c in valid_countries:
    df[c] *= installed_capacity_2015[c]["solar"]

# -------- Load and clean best-ARIMA parameters --------

params = (pd.read_csv(f"results/arma_best_model_pv_day_{args.forecast_day}.csv")
          .assign(hour=lambda x: x["hour"].astype(int))
          .set_index(["country", "hour"]))
params["notes"] = params["notes"].fillna("")

# -------- Scenario generation --------
WINDOW = 270
N_SCEN = 100
N_SETS = 10
out_dir = os.path.join("scenario_results", f"day_{args.forecast_day}")
os.makedirs(out_dir, exist_ok=True)

ordered = {
    c: {
        h: (df.loc[df.hour == h, ["Date", c]]
              .sort_values("Date")[c]
              .reset_index(drop=True))
        for h in range(24)
    }
    for c in valid_countries
}

for c in valid_countries:
    for set_no in range(1, N_SETS + 1):
        hourly_scenarios = {}
        for h in range(24):
            try:
                param_row = params.loc[(c, h)]

                if param_row["notes"] == "Zero-production rule applied":
                    hourly_scenarios[h] = np.zeros(N_SCEN)
                    logging.info(f"{c} H{h:02d} — Zero-production rule applied, using all 0s.")
                    continue  # Skip to the next hour

                p, d, q = param_row[["p", "d", "q"]].astype(int)
                start = args.forecast_day - 1 - WINDOW
                train = ordered[c][h].iloc[start : args.forecast_day - 1]
                    
                if d == 1:
                    # RETRANSFORM PROCESS for differenced data
                    diff_data = train.diff().dropna()
                    model_fit = ARIMA(diff_data, order=(p, 0, q)).fit()
                    simulated_diffs = model_fit.simulate(nsimulations=1, repetitions=N_SCEN)
                    last_known_values = train.iloc[-1:].values
                    scenarios = simulated_diffs.values + last_known_values[:, np.newaxis]
                else:
                    # STANDARD FORECAST for non-differenced data
                    model_fit = ARIMA(train, order=(p, 0, q)).fit()
                    simulated_values = model_fit.simulate(nsimulations=1, repetitions=N_SCEN)
                    scenarios = simulated_values.values.copy()

                # PV generation cannot be negative, so clip values at 0
                scenarios[scenarios < 0] = 0
                
                hourly_scenarios[h] = scenarios.squeeze()

            except KeyError:
                hourly_scenarios[h] = np.zeros(N_SCEN).squeeze()
            except Exception as e:
                logging.warning(f"{c} H{h:02d} — Fallback to 0s due to error: {e}")
                hourly_scenarios[h] = np.zeros(N_SCEN).squeeze()

        # Save the generated scenarios to a CSV file
        (pd.DataFrame(hourly_scenarios, index=range(N_SCEN)).T
           .rename_axis("hour")
           .add_prefix("scenario_")
           .to_csv(f"{out_dir}/pv_arma_{c}_set_{set_no}.csv"))

logging.info("Finished — all %d sets created for every country.", N_SETS)