#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=arma_pv
#SBATCH --output=output.%j.arma_pv
#SBATCH --time=08:00:00
#SBATCH --mail-user=mj1v21@soton.ac.uk
#SBATCH --mail-type=END,FAIL

# Fit best ARIMA(p,d,q) models for PV across all 28 countries and all
# production hours. Heavier than wind (270-day window, per-hour fitting,
# diagnostic plots) so uses a longer walltime.
#
# Submit from the project root with FORECAST_DAY set, e.g.:
#   sbatch --export=ALL,FORECAST_DAY=271 code/run_arma_pv.slurm
#   sbatch --export=ALL,FORECAST_DAY=301 code/run_arma_pv.slurm
#   sbatch --export=ALL,FORECAST_DAY=332 code/run_arma_pv.slurm
#   sbatch --export=ALL,FORECAST_DAY=362 code/run_arma_pv.slurm
#
# Outputs:
#   results/arma_best_model_pv_day_<FORECAST_DAY>.csv
#   arma_predictions_pv_day_<FORECAST_DAY>.log
#   diagnostic_plots/<country>_hour_<h>_diagnostics.png
#
# To submit all four rolling-window days at once:
#   for DAY in 271 301 332 362; do
#     sbatch --export=ALL,FORECAST_DAY=$DAY code/run_arma_pv.slurm
#   done

# Activate your conda environment if not loaded via .bashrc, e.g.:
# module load conda
# conda activate <your_env>

FORECAST_DAY=${FORECAST_DAY:-271}

cd "$SLURM_SUBMIT_DIR"
mkdir -p results diagnostic_plots

# Prevent matplotlib from trying to open a display on the compute node.
export MPLBACKEND=Agg

python code/arma_pv_best_models.py --forecast_day "$FORECAST_DAY"
