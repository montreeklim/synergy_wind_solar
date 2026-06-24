#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=arma_wind
#SBATCH --output=output.%j.arma_wind
#SBATCH --time=02:00:00
#SBATCH --mail-user=mj1v21@soton.ac.uk
#SBATCH --mail-type=END,FAIL

# Fit best ARMA(p,q) models for wind across all 28 countries.
# Submit from the project root with FORECAST_DAY set, e.g.:
#   sbatch --export=ALL,FORECAST_DAY=271 code/run_arma_wind.slurm
#   sbatch --export=ALL,FORECAST_DAY=301 code/run_arma_wind.slurm
#   sbatch --export=ALL,FORECAST_DAY=332 code/run_arma_wind.slurm
#   sbatch --export=ALL,FORECAST_DAY=362 code/run_arma_wind.slurm
#
# Output: results/arma_best_model_wind_day_<FORECAST_DAY>.csv
#
# To submit all four rolling-window days at once:
#   for DAY in 271 301 332 362; do
#     sbatch --export=ALL,FORECAST_DAY=$DAY code/run_arma_wind.slurm
#   done

# Activate your conda environment if not loaded via .bashrc, e.g.:
# module load conda
# conda activate <your_env>

FORECAST_DAY=${FORECAST_DAY:-271}

cd "$SLURM_SUBMIT_DIR"
mkdir -p results

python code/arma_wind_best_models.py --forecast_day "$FORECAST_DAY"
