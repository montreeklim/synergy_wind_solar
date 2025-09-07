# Synergy between wind and solar in energy generation with chance-constrained models

This project is based on the paper Synergy between wind and solar in energy generation with chance-constrained models.

We provide a framework for quantifying the synergistic effects of combining wind and solar power generation. The goal is to analyze how a mixed portfolio of these intermittent renewable sources can lead to a more stable and reliable aggregate power output. 

We provide code to calculate a **synergy ratio** based on time-series generation data.

## Repository content
The repository contains the following content:
- `catchment_population` presents an efficient algorithm to compute the "catchment population" of each recycling center that was used to estimate the capacities. An implementation of this algorithm is contained in `catchment_population.py`. Further, this directory contains two corresponding input data files in csv format: `bavaria_grid_population.csv` is a file containing the latitude and longitude of the centroid of each 100m x 100m grid in Bavaria as well as the residing population. `rc_locations.csv` contains the latitude and longitude of each recycling center in Bavaria.
- `data` contains the two input data files that are used in the MIQP model: `users_and_facilities.xlsx` contains all ZIP codes and recycling centers related data like the population, centroid and regional spatial type (rural/urban) of each ZIP code as well as the capacity, centroid and regional spatial type of each recycling center. `travel_dict.json.pbz2` is a compressed json file that contains the travel probabilities from each ZIP code to each recycling center.  
- `model` contains scripts for the optimization of the MIQP model and visualization of the results: `model.py` contains functions for building and optimizing the MIQP model, while `greedy_heuristic.py` applies a greedy heuristic to achieve a feasible solution. `plotting.py` and `results.py` contain functions tasked with visualizing results through various different plots as well as excel tables. They also contain superordinate functions that create the corresponding results first by running functions from `model.py` and/or `greedy_heuristic.py` before creating the corresponding visualization. These functions are called by functions in `tables_and_figures.py` to create the exact same figures and tables that are included in the paper from scratch. Lastly, `utils.py` contains helper and utility functions.
- `subsequent_work`contains a copy of the files in `model`that have been further expanded on for the application of this topic.
- `results` contains the excel tables and figures visualizing the results that are included in the paper.
- `appendix.pdf` that supplments the main text. This appendix includes proofs, sources of data, and figures and tables. 
## Requirements to run code
The code uses some open-source Python packages. The ones that the reader may be most unfamiliar with are:
- Pyomo, a Python-based optimization modeling language that allows building optimization models.
- Gurobi, a software well-equiped for solving complex optimization models such as MIQPs.
- Geopy, which was used for calculating geodesic distances (i.e. shortest distances on the surface of the earth) between two locations.
