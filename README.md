# Measuring the Economic Value of Wind–Solar Complementarity in Europe Using Chance Constraints

This repository accompanies the working-paper titled "Measuring the Economic Value of Wind–Solar Complementarity in Europe Using Chance Constraints" by Montree Jaidee and Bismark Singh. We provide a framework for quantifying the synergistic effects of combining wind and solar power generation.

## Repository content

The repository contains the following content:

- `data`: contains the two historical data files of wind and solar generation in 2015 for European countries, adapted from the EMHIRES dataset. All required features are selected and saved as `df_wind` and `df_pv`. Generated scenarios are contained in the `scenarios` folder, which includes 10 independent sets of 100 sampled scenarios for wind and solar generation of each country.
- `code`: contains scripts for the Naive optimization model and the Storage-Enhanced optimization model, as well as visualization of the data.  
  - Visualization: `mean_generation_chart.py`, `heat_map_correlation.py`, `heatmap_mean_log_scale.py` (exploratory data analysis plots)  
  - Forecasting: `arma_pv_best_models.py`, `arma_wind_best_models.py` (select ARMA(p,q) lags minimizing BIC with white-noise residuals)  
  - Scenario generation: `pv_scenarios_generation.py`, `wind_scenarios_generation.py` (generate scenarios for the optimization models)  
  - Optimization: `naive_optimization_models.py` (Naive model), `parallel_battery.py` (Storage-Enhanced model, parallel execution)
- `results`: contains the Excel tables for the results produced by the optimization models in the article.

## Parameter Values

### 1. Hourly Wind Profit ($R$)
The profit of wind generation for each hour of the day is defined as:

| Hour | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Profit** | 0.0189 | 0.0172 | 0.0155 | 0.0148 | 0.0146 | 0.0151 | 0.0173 | 0.0219 | 0.0227 | 0.0226 | 0.0235 | 0.0242 |

| Hour | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 | 24 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Profit** | 0.0250 | 0.0261 | 0.0285 | 0.0353 | 0.0531 | 0.0671 | 0.0438 | 0.0333 | 0.0287 | 0.0268 | 0.0240 | 0.0211 |

### 2. Battery and Operational Costs
* **Charge/Discharge Costs ($C_c, C_d$):** 0.0256
* **Efficiency Coefficient ($\eta$):** 0.9
* **Upper Bound Capacity ($X_{ub}$):** installed capacity of each country
* **Lower Bound Capacity ($X_{lb}$):** $0.2 X_{ub}$
* **Maximum Charge/Discharge Rate:** $0.5 X_{ub}$
* **Initial Battery State:** $0.5 X_{ub}$

## Requirements to run the code
The code uses some open-source Python packages. The ones that the reader may be most unfamiliar with are:
- [Gurobi](https://www.gurobi.com/) – for solving mixed-integer optimization problems  
- [GeoPandas](https://geopandas.org/) and [Cartopy](https://scitools.org.uk/cartopy/) – for producing geospatial plots  
