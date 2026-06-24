#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=gen_scenarios
#SBATCH --output=output.%A.gen_scenarios
#SBATCH --time=02:00:00
#SBATCH --mail-user=mj1v21@soton.ac.uk
#SBATCH --mail-type=END,FAIL

# Generate PV and wind scenarios for a single forecast day.
# Submit with FORECAST_DAY set, e.g.:
#   sbatch --export=ALL,FORECAST_DAY=271 generate_scenarios.slurm
#
# Use --dependency=afterok:<JOBID> on the battery array job so
# optimisation only starts after scenarios are written.

FORECAST_DAY=${FORECAST_DAY:-271}

cd "$SLURM_SUBMIT_DIR"
mkdir -p scenario_results

python code/pv_scenarios_generation.py   --forecast_day $FORECAST_DAY
python code/wind_scenarios_generation.py --forecast_day $FORECAST_DAY
