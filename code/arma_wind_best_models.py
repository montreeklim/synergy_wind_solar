import pandas as pd
import numpy as np
import warnings
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore", category=UserWarning)


def find_best_arma_model(data, p_range, q_range, significance_level=0.05):
    """
    Finds the best ARMA(p,q) model by selecting from models that pass the Ljung-Box test
    and then choosing the one with the lowest BIC.

    Args:
        data (pd.Series): The time series data.
        p_range (range): A range of values for the AR order 'p'.
        q_range (range): A range of values for the MA order 'q'.
        significance_level (float): The p-value threshold for the Ljung-Box test.

    Returns:
        dict: A dictionary containing the best model's order, BIC, and the fitted model object.
              Returns None if no model passes the Ljung-Box test.
    """
    passing_models = []
    print("--- Starting Model Search ---")

    for p in p_range:
        for q in q_range:
            if p == 0 and q == 0:
                continue

            order = (p, 0, q)  
            try:
                # Fit the ARMA model
                model = ARIMA(data, order=order).fit()

                # Perform Ljung-Box test on model residuals
                lb_test = acorr_ljungbox(model.resid, lags=[5, 10, 15], return_df=True)

                # Check if all p-values are above the significance level
                if (lb_test['lb_pvalue'] < significance_level).all():
                    passing_models.append({
                        'order': order,
                        'bic': model.bic,
                        'model': model
                    })
                    print(f"ARMA{order} PASSED. BIC: {model.bic:.2f}")

            except Exception as e:
                # Silently continue if a model fails to converge
                continue

    print("--- Search Complete ---")

    if not passing_models:
        return None

    # Select the best model (lowest BIC) from the passing ones
    best_model_info = min(passing_models, key=lambda x: x['bic'])
    
    print(f"\n Best Model Found: ARMA{best_model_info['order']} with BIC: {best_model_info['bic']:.2f}")
    return best_model_info

def find_best_arma_by_bic(data, p_range, q_range, significance_level=0.05):
    """
    Finds the ARMA(p,q) model with the lowest BIC. After finding the best model,
    it performs and prints a Ljung-Box test on its residuals as a diagnostic check.

    Args:
        data (pd.Series): The time series data.
        p_range (range): A range of values for the AR order 'p'.
        q_range (range): A range of values for the MA order 'q'.
        significance_level (float): The p-value threshold for interpreting the Ljung-Box test.

    Returns:
        dict: A dictionary containing the best model's order, BIC, the fitted model object,
              and its Ljung-Box test results. Returns None if no models could be fitted.
    """
    best_bic = np.inf
    best_model_info = {}

    print("--- Starting Model Search (Finding Lowest BIC) ---")

    for p in p_range:
        for q in q_range:
            if p == 0 and q == 0:
                continue

            order = (p, 0, q)
            try:
                # Fit the model and get its BIC
                model = ARIMA(data, order=order).fit()
                current_bic = model.bic

                # If current model is better, store its info
                if current_bic < best_bic:
                    best_bic = current_bic
                    best_model_info = {
                        'order': order,
                        'bic': current_bic,
                        'model': model
                    }
                    print(f"  > New best found: ARMA{order} with BIC: {current_bic:.2f}")

            except Exception:
                # Silently ignore models that fail to converge or have other issues
                continue

    print("--- Search Complete ---\n")

    # Check if any model was successfully fitted
    if not best_model_info:
        print("Could not fit any models in the given range.")
        return None

    # --- Perform Ljung-Box test ONLY on the single best model ---
    best_model = best_model_info['model']
    lb_test_results = acorr_ljungbox(best_model.resid, lags=[5, 10, 15], return_df=True)

    # Add the test results to our output dictionary
    best_model_info['lb_test'] = lb_test_results

    # --- Final Report ---
    print(f"Best Model by BIC: ARMA{best_model_info['order']} with BIC: {best_model_info['bic']:.2f}")
    
    print("\n--- Diagnostic: Ljung-Box Test on Best Model's Residuals ---")
    print("Null Hypothesis: The residuals are independently distributed (white noise).")
    print(f"P-values < {significance_level} support this, indicating a good fit.")
    print(lb_test_results)

    return best_model_info


# DATA LOADING & PREP
# -------------------------------
try:
    df = pd.read_csv('df_wind.csv')
except FileNotFoundError:
    print("Error: 'df_wind.csv' not found. Please make sure the data file is in the correct directory.")
    exit() # Exit the script if the data file is missing

df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y %H:%M')
df.rename(columns={'Hour': 'hour'}, inplace=True)

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

df = df[['Date', 'hour'] + valid_countries]
for country in valid_countries:
    wind_capacity = installed_capacity_2015[country]['wind']
    df[country] *= wind_capacity
    
# -------------------------------
# Perform Stationarity Check for Each Country
# -------------------------------
last_train_date = 270
train_day = 60
for country in valid_countries:
    print(f"--- Processing Country: {country} ---")
    wind_data = df[country].astype(float)
    train_data = wind_data.iloc[last_train_date*24 - train_day*24:last_train_date*24]

    # Perform the Augmented Dickey-Fuller (ADF) test
    adf_result = adfuller(train_data)
    p_value = adf_result[1]

    print(f"ADF Statistic: {adf_result[0]:.4f}")
    print(f"P-value: {p_value:.4f}")

    # Interpret the result based on the p-value
    if p_value <= 0.05:
        print(f"Result: The series for {country} is likely STATIONARY.\n")
    else:
        print(f"Result: The series for {country} is likely NON-STATIONARY.\n")

# MAIN PROCESSING LOOP
# -------------------------------
# Define the search space for p and q orders
p_to_check = range(0, 6) # Check p from 0 to 5
q_to_check = range(0, 6) # Check q from 0 to 5

final_params = {}
# Using a subset for demonstration. To run  for all, change to: for country in valid_countries:
for country in valid_countries: 
    print(f" PROCESSING COUNTRY: {country}")
    
    wind_data = df[country].astype(float)
    train_data = wind_data.iloc[last_train_date*24 - train_day*24:last_train_date*24]
    
    # First, try to find a model on the original data
    best_model_info = find_best_arma_model(train_data, p_to_check, q_to_check)
    
    if best_model_info:
        p = best_model_info['order'][0]
        q = best_model_info['order'][2]
        final_params[country] = {
            'p': p,
            'q': q,
            'bic': best_model_info['bic']
        }
    else:
        print(f"\n No model passed for {country} on original data.")
        best_model_info = find_best_arma_by_bic(train_data, p_to_check, q_to_check)
        p = best_model_info['order'][0]
        q = best_model_info['order'][2]
        final_params[country] = {
            'p': p,
            'q': q,
            'bic': best_model_info['bic']
        }


# SUMMARY OF RESULTS
# -------------------------------
print("\n\n==============================================")
print("       FINAL MODEL SELECTION SUMMARY")
print("==============================================")
if not final_params:
    print("No valid models were found for any country.")
else:
    for country, params in final_params.items():
        p, q = params['p'], params['q']
        bic = params['bic']
        print(f"  - {country}: Best model is ARMA({p}, {q}) | BIC: {bic:.2f}")

# SAVE RESULTS TO CSV
# -------------------------------
if final_params:
    # Convert the results dictionary to a pandas DataFrame
    results_df = pd.DataFrame.from_dict(final_params, orient='index')
    results_df.reset_index(inplace=True)
    results_df.rename(columns={'index': 'Country'}, inplace=True)
    
    # Save the DataFrame to a CSV file
    output_filename = 'arma_best_model_wind.csv'
    results_df.to_csv(output_filename, index=False)
    
    print("\n----------------------------------------------")
    print(f" Results successfully saved to '{output_filename}'")
    print("----------------------------------------------")