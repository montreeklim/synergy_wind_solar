import pandas as pd
import numpy as np
import scipy.stats as st

# --- 1. Load Data ---
try:
    # Load the dataset from the file you uploaded
    df = pd.read_csv('all_countries_naive_results.csv')
    print("Successfully loaded 'all_countries_naive_results.csv'")
except FileNotFoundError:
    print("Error: 'all_countries_naive_results.csv' not found. Please ensure the file is uploaded.")
    # Exit or create a dummy df if you want the script to run for demonstration
    df = pd.DataFrame() # Create empty df to prevent crash

if not df.empty:
    # --- 2. Define Confidence Interval Functions ---
    # These functions calculate the lower and upper bounds of the 95% CI.
    # They use the t-distribution, which is accurate for small sample sizes (like n=10).
    
    def ci_lower_bound(data):
        """Calculates the lower bound of the 95% CI."""
        if len(data) < 2:
            return np.nan
        return st.t.interval(0.95, len(data)-1, loc=np.mean(data), scale=st.sem(data))[0]

    def ci_upper_bound(data):
        """Calculates the upper bound of the 95% CI."""
        if len(data) < 2:
            return np.nan
        return st.t.interval(0.95, len(data)-1, loc=np.mean(data), scale=st.sem(data))[1]

    # --- 3. Group and Aggregate ---
    print("Calculating confidence intervals for the 'ratio' column...")
    
    # Group by the experimental parameters
    grouped = df.groupby(['country', 'tol'])

    # Define the aggregations to perform ONLY on the 'ratio' column
    aggregations_for_ratio = {
        'ratio': ['mean', ci_lower_bound, ci_upper_bound]
    }

    # Apply the aggregation to the grouped data
    ratio_ci_df = grouped.agg(aggregations_for_ratio)

    # --- 4. Format the Output ---
    
    # The aggregation creates multi-level columns (e.g., ('ratio', 'mean')).
    # This line flattens them into a single level (e.g., 'ratio_mean').
    ratio_ci_df.columns = ['_'.join(col).strip() for col in ratio_ci_df.columns.values]

    # Turn the 'country' and 'tol' group keys from an index back into columns
    ratio_ci_df = ratio_ci_df.reset_index()

    # --- 5. Display and Save Results ---
    
    # Save the results to a new CSV file
    output_filename = 'ratio_confidence_intervals.csv'
    ratio_ci_df.to_csv(output_filename, index=False, float_format='%.5f')

    print(f"\nResults have been saved to '{output_filename}'")
    print("\nHere is a preview of the final data:")
    print(ratio_ci_df.head())