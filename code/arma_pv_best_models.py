import pandas as pd
import numpy as np
import warnings
import logging
import os # <-- NEW: For creating directories
import matplotlib.pyplot as plt # <-- NEW: For plotting
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf # <-- NEW
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller

def plot_time_series_diagnostics(series, country, hour, save_dir='plots'):
    """
    Plots the time series, ACF, and PACF for a given series
    and saves the combined plot to a file.

    Args:
        series (pd.Series): The time series data to plot.
        country (str): The country code (e.g., 'AT').
        hour (int): The hour of the day.
        save_dir (str): The directory to save plots in.
    """
    # Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Create a figure with 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle(f'Diagnostic Plots for {country} - Hour {hour}', fontsize=16)

    # 1. Time Series Plot
    series.plot(ax=axes[0])
    axes[0].set_title('Training Data Time Series')
    axes[0].set_ylabel('Production (MW)')
    axes[0].grid(True)

    # 2. ACF Plot
    plot_acf(series, ax=axes[1], lags=40)
    axes[1].set_title('Autocorrelation Function (ACF)')
    axes[1].grid(True)

    # 3. PACF Plot
    plot_pacf(series, ax=axes[2], lags=40)
    axes[2].set_title('Partial Autocorrelation Function (PACF)')
    axes[2].grid(True)

    # Adjust layout and save the figure
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
    file_path = os.path.join(save_dir, f'{country}_hour_{hour}_diagnostics.png')
    plt.savefig(file_path)
    plt.close(fig)


def find_best_arima_model_bic_ljungbox(train_data, p_range, q_range, d_order, ljung_box_lags=[5, 10, 15], alpha=0.05):
    """
    Finds the best ARIMA(p,d,q) model based on the lowest BIC,
    considering only models where residuals pass the Ljung-Box test.

    The Ljung-Box null hypothesis is that the residuals are independently distributed.
    A p-value < alpha suggests the residuals are random (white noise), which is desired.

    Args:
        train_data (pd.Series): The training time series data.
        p_range (list or range): The range of p values to check.
        q_range (list or range): The range of q values to check.
        d_order (int): The differencing order 'd'.
        ljung_box_lags (list): The lags to use for the Ljung-Box test.
        alpha (float): The significance level for the Ljung-Box test.

    Returns:
        dict: A dictionary with the best model, its BIC, and its order,
              or None if no model passes the diagnostic checks.
    """
    best_bic = float('inf')
    best_model_info = None

    for p in p_range:
        for q in q_range:
            order = (p, d_order, q)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                try:
                    
                    # Fit the ARIMA model
                    model = ARIMA(train_data, order=order).fit()
                    
                    # --- Ljung-Box Test for residual autocorrelation ---
                    lb_results = acorr_ljungbox(model.resid, lags=ljung_box_lags, return_df=True)
                    p_values = lb_results['lb_pvalue']
                    
                    # Check if all p-values are below the significance level (pass the test)
                    if (p_values < alpha).all():
                        current_bic = model.bic
                        # Check if this model is better than the previous best
                        if current_bic < best_bic:
                            best_bic = current_bic
                            best_model_info = {'model': model, 'bic': current_bic, 'order': order}

                except Exception:
                    continue
                    
    return best_model_info

# -------------------------------
# Setup logging
# -------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("arma_predictions.log", mode="w", encoding="utf-8")
    ]
)

# -------------------------------
# 1. Load & preprocess
# -------------------------------
df = pd.read_csv('../data/df_pv.csv')
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y %H:%M')
df.rename(columns={'Hour': 'hour'}, inplace=True)

# -------------------------------
# 2. Installed capacities (MW)
# -------------------------------
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


# -------------------------------
# 3. Filter & convert to MW
# -------------------------------
cols = [c for c in df.columns if c not in ['Date','hour']]
valid_countries = [c for c in cols if c in installed_capacity_2015]
df = df[['Date','hour'] + valid_countries]
for c in valid_countries:
    df[c] = df[c] * installed_capacity_2015[c]['solar']
    
# -------------------------------
# 4. Define modeling parameters
# -------------------------------
p_to_check = range(0, 6) # AR order p
q_to_check = range(0, 6) # MA order q

train_size = 270
forecast_date = 270
hours = range(24)

# Initialize a dictionary to store all forecasts
all_forecasts = {}

# -------------------------------
# 5. MAIN PROCESSING LOOP
# -------------------------------
best_params_list = []

ZERO_CHECK_WINDOW = 7

for country in valid_countries:
    print(f"\n{'='*50}\nProcessing Country: {country}\n{'='*50}")
    hour_sums = df.groupby('hour')[country].sum()
    production_hours = [h for h in hours if hour_sums.get(h, 0) > 0]

    for h in production_hours:
        print(f"\n--- Hour: {h} ---")
        series_h = df[df['hour']==h][country].reset_index(drop=True)
        train_data = series_h.iloc[forecast_date - train_size:forecast_date]

        # Handle series with null values or no variance
        if train_data.isnull().any() or train_data.var() == 0:
            if train_data.var() == 0:
                best_params_list.append({'country': country, 'hour': h, 'p': 0, 'd': 0, 'q': 0, 'bic': np.nan, 'notes': 'Zero variance'})
            continue
        
        # We use a small tolerance (1e-6) for floating point comparisons.
        if len(train_data) >= ZERO_CHECK_WINDOW and (train_data.tail(ZERO_CHECK_WINDOW) < 1e-6).all():
            print(f"INFO: Recent production is zero for {country}, hour {h}. Applying zero-forecast rule.")
            logging.info(f"SUCCESS: Zero-production rule applied for {country}, hour {h}. Forecasting 0.")
            best_params_list.append({
                'country': country,
                'hour': h,
                'p': 0, 'd': 0, 'q': 0,
                'bic': np.nan,
                'notes': 'Zero-production rule applied'
            })
            continue
        
        print(f"Generating diagnostic plots for {country}, hour {h}...")
        plot_time_series_diagnostics(train_data, country, h, save_dir='diagnostic_plots')

        # --- Stationarity Check to determine 'd' ---
        d = 0 # Default differencing order
        p_value = adfuller(train_data)[1]
        print(f"Initial ADF Test p-value: {p_value:.4f}")

        if p_value >= 0.05:
            print("Series is not stationary. Applying first difference and re-testing.")
            diff_train_data = train_data.diff().dropna()
            if not diff_train_data.empty and diff_train_data.var() > 1e-9:
                diff_p_value = adfuller(diff_train_data)[1]
                print(f"ADF Test p-value after 1st difference: {diff_p_value:.4f}")
                if diff_p_value < 0.05:
                    print("Series is stationary after one difference. Setting d=1.")
                    d = 1
                else:
                    diff_2_train_data = diff_train_data.diff().dropna()
                    if not diff_2_train_data.empty and diff_2_train_data.var() > 1e-9 and adfuller(diff_2_train_data)[1] < 0.05:
                        d = 2
                    else:
                        logging.warning(f"Series for {country} hour {h} is still non-stationary. Proceeding with d=2.")
                        d = 2
            else:
                logging.warning(f"Could not perform ADF test on differenced series for {country} hour {h}. Defaulting to d=1.")
                d = 1
        else:
            print("Series is stationary. Setting d=0.")
            d = 0
            
        # --- Find Best Model using BIC and Ljung-Box ---
        best_model_info = find_best_arima_model_bic_ljungbox(
            train_data=train_data, 
            p_range=p_to_check,
            q_range=q_to_check,
            d_order=d
        )

        if best_model_info:
            best_order = best_model_info['order']
            best_bic = best_model_info['bic']
            
            best_params_list.append({
                'country': country, 'hour': h, 'p': best_order[0],
                'd': best_order[1], 'q': best_order[2], 'bic': best_bic,
                'notes': 'Passes Ljung-Box'
            })
            logging.info(f"SUCCESS: Best params for {country}, hour {h} are {best_order} with BIC: {best_bic:.2f}")
        else:
            best_params_list.append({
                'country': country, 'hour': h, 'p': np.nan, 'd': d,
                'q': np.nan, 'bic': np.nan,
                'notes': 'No model passed Ljung-Box test'
            })
            logging.warning(f"FAILURE: Could not find a suitable ARIMA model for {country}, hour {h} that passed diagnostics.")

# ---------------------------------------------
# 6. Convert list to DataFrame and Save to CSV
# ---------------------------------------------
print("\n\n--- FINALIZING AND SAVING PARAMETERS ---")
params_df = pd.DataFrame(best_params_list)

params_df = params_df[['country', 'hour', 'p', 'd', 'q', 'bic', 'notes']]
params_df.to_csv("arma_best_model_pv.csv", index=False)

print("\n--- Sample of the output file ---")
print(params_df.head())