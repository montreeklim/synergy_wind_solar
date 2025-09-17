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

## Requirements to run the code
The code uses some open-source Python packages. The ones that the reader may be most unfamiliar with are:
- [Gurobi](https://www.gurobi.com/) – for solving mixed-integer optimization problems  
- [GeoPandas](https://geopandas.org/) and [Cartopy](https://scitools.org.uk/cartopy/) – for producing geospatial plots  
