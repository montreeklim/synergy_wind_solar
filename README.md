# Measuring the Economic Value of Wind–Solar Complementarity in Europe Using Chance Constraints

This repository accompanies the working paper titled "Measuring the Economic Value of Wind–Solar Complementarity in Europe Using Chance Constraints" by Montree Jaidee and Bismark Singh. We provide a framework for quantifying the synergistic effects of combining wind and solar power generation across 28 European countries, using two chance-constrained stochastic optimization models — a **Naive** model without storage and a **Storage-Enhanced** model with battery dynamics — under three reliability tolerances (ε = 0.01, 0.05, 0.10) and four seasonal forecast dates.

## Repository content

The repository contains the following content:

- `data`: contains the two historical data files of wind and solar generation in 2015 for European countries, adapted from the EMHIRES dataset. All required features are selected and saved as `df_wind` and `df_pv`. `scenario_results/day_{271,301,332,362}` holds the generated ARIMA scenario sets (10 independent sets of 100 samples per country) for each of the four forecast dates used in the analysis.
- `code`: contains scripts for the Naive optimization model and the Storage-Enhanced optimization model, scenario generation, statistical post-processing, and visualization.
  - Visualization: `mean_generation_chart.py`, `heat_map_correlation.py`, `heatmap_mean_log_scale.py` (exploratory data analysis plots); `boxplot_by_region.py`, `boxplot_by_region_naive.py`, `boxplot_battery_regional_all_tols.py`, `boxplot_naive_regional_all_tols.py`, `synergy_heatmap_day362.py` (Synergy Ratio plots)
  - Forecasting: `arma_pv_best_models.py`, `arma_wind_best_models.py` (select ARMA(p,q) lags minimizing BIC with white-noise residuals)
  - Scenario generation: `pv_scenarios_generation.py`, `wind_scenarios_generation.py` (generate scenarios for the optimization models)
  - Optimization: `naive_optimization_models.py` (Naive model); `parallel_battery.py` / `parallel_battery_model_cap.py` (Storage-Enhanced model, parallel execution across countries, tolerances, forecast dates, and installed-capacity scalings); `parallel_battery_cap_sensitivity.py` (battery-sizing sensitivity)
  - Statistics: `naive_CI_calculation.py`, `fix_naive_ratios.py`, `compute_crossdate_ci.py` (95% confidence intervals for the Synergy Ratio via the t-distribution); `merge_naive_results.py`, `merge_naive_results_1000.py`, `combine_battery_results.py` (merge per-country outputs)
  - Comparison: `compare_days_100.py`, `compare_days_naive.py`, `compare_tolerances.py`, `compare_1000_vs_100_naive.py`, `compare_naive_vs_battery.py`, `compare_storage_sizes_day271.py` (cross-date, cross-tolerance, cross-model, and cross-capacity comparisons)
  - Explanatory analysis: `random_forest.py`, `decision_tree.py`, `regression.py`, `pv_balance_vs_synergy_battery.py`, `pv_balance_vs_synergy_naive.py`, `pv_share_gen_vs_synergy_battery.py` (what predicts the Synergy Ratio across countries)
  - SLURM job scripts for running the above on HPC (`run_arma_pv.slurm`, `run_arma_wind.slurm`, `generate_scenarios.slurm`, `run_naive_model.slurm`, `run_naive_model_1000.slurm`, `run_parallel_battery_model.slurm`, `run_cap_sensitivity_day271.slurm`)
- `naive_results` / `battery_results`: per-country raw optimization output (one CSV per country) for the Naive and Storage-Enhanced models respectively. `battery_results` is split into `50_installed_capacity/`, `100_installed_capacity/` (baseline), and `200_installed_capacity/` subfolders, supporting the battery-sizing sensitivity analysis.
- `results`: contains the confidence-interval tables, comparison tables, and figures (boxplots, heatmaps, decision trees, regression plots) produced from `naive_results` and `battery_results`, broken down by forecast date and reliability tolerance, plus the consolidated Excel tables (`all_countries_naive_results.xlsx`, `all_countries_battery_results.xlsx`) used in the article.

## The Synergy Ratio

Each model reports the Synergy Ratio (SR) for every country: the combined (wind + solar) profit divided by the sum of the two standalone profits. SR > 1.0 indicates that operating wind and solar jointly is more profitable than operating them independently. Uncertainty is characterized by 95% confidence intervals over ten independent scenario sets per (country, forecast date, tolerance) combination, with robustness confirmed by a rolling-window test across four seasonal forecast dates and by a pooled 1000-scenario variant of the Naive model.

## Parameter Values

### 1. Hourly Profit ($R$)
The profit of electricity generation for each hour of the day is:

| Hour | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Profit** | 0.0189 | 0.0172 | 0.0155 | 0.0148 | 0.0146 | 0.0151 | 0.0173 | 0.0219 | 0.0227 | 0.0226 | 0.0235 | 0.0242 |

| Hour | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 | 24 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Profit** | 0.0250 | 0.0261 | 0.0285 | 0.0353 | 0.0531 | 0.0671 | 0.0438 | 0.0333 | 0.0287 | 0.0268 | 0.0240 | 0.0211 |

### 2. Battery and Operational Costs
* **Charge/Discharge Costs ($C_c, C_d$):** 0.0256
* **Efficiency Coefficient ($\eta$):** 0.9
* **Upper Bound Capacity ($X_{ub}$):** installed capacity of each country, scaled to 50%, 100% (baseline), 200%, or 384% for the sensitivity analysis
* **Lower Bound Capacity ($X_{lb}$):** $0.2 X_{ub}$
* **Maximum Charge/Discharge Rate:** $0.5 X_{ub}$
* **Initial Battery State:** $0.5 X_{ub}$

### 3. Reliability Tolerance ($\varepsilon$)
The chance-constrained models are solved at three reliability tolerances: 0.01, 0.05, and 0.10, and across four rolling-window forecast dates (calendar days 271, 301, 332, 362 — one per season) to test seasonal robustness.

## Requirements to run the code
The code uses some open-source Python packages. The ones that the reader may be most unfamiliar with are:
- [Gurobi](https://www.gurobi.com/) – for solving mixed-integer optimization problems
- [GeoPandas](https://geopandas.org/) and [Cartopy](https://scitools.org.uk/cartopy/) – for producing geospatial plots
- [statsmodels](https://www.statsmodels.org/) – for ARMA model fitting
- [scikit-learn](https://scikit-learn.org/) – for the random forest, decision tree, and regression explanatory analysis

## Reproducing the Results
The scenario sets are already provided in `scenario_results/day_{271,301,332,362}/`, so reproduction can start directly from the optimization step (`--index` selects a country, 0–27, in the fixed country order listed in `code/`):

```bash
# Naive (no-storage) model — single country
python code/naive_optimization_models.py --index 0

# Storage-Enhanced (battery) model — single country
python code/parallel_battery.py --index 0

# Merge per-country outputs, then compute 95% confidence intervals
python code/merge_naive_results.py
python code/combine_battery_results.py
python code/naive_CI_calculation.py
```

On HPC, the battery model across all 28 countries is submitted as a SLURM array via `sbatch code/run_parallel_battery_model.slurm`.
