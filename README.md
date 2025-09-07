# Synergy between wind and solar in energy generation with chance-constrained models

This project is based on the paper Synergy between wind and solar in energy generation with chance-constrained models. We provide a framework for quantifying the synergistic effects of combining wind and solar power generation.

## Repository content
The repository contains the following content:
- `data` contains the two historical data files that contains wind and solar generation in 2015 for European countries from EMHIRES dataset. All required features are selected and saved as `df_wind` and `df_pv`. Generated scenarios are contained in `scenarios` folder inside which contain 10 sets of 100 generated scenarios for wind and solar generation of each country.
- `code` contains scripts for the optimization of the Naive model and Storage-Enhanced model, and visualization of the data: `mean_generation_chart.py` `heat_map_correlation.py` and `heatmap_mean_log_scale.py` produce plots for explrationary data analysis.  `arma_pv_best_models.py` and `arma_wind_best_models.py` find the lag of ARMA(p,q) that minimize the BIC and the residuals are white noise to use as a forecasting model. `pv_scenarios_generation.py` and `wind_scenarios_generation.py` generate scenarios used in optimization model from the forecasting model. `naive_optimization_models.py` solve the Naive optimization and `parallel_battery.py` solve the Storage-Enhanced model in array which can be run in parallel.
- `results` contains the excel tables for optimization results discussed in the paper.
## Requirements to run code
The code uses some open-source Python packages. The ones that the reader may be most unfamiliar with are:
- Gurobi, a software well-equiped for solving mixed-integer programming models.
- Geopandas and Cartopy, Python packages for create plots related to real-world locations. 
