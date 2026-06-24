#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=cap_sens_d271
#SBATCH --output=output.%A_%a.cap_sensitivity
#SBATCH --time=12:00:00
#SBATCH --mem=50G
#SBATCH --mail-user=mj1v21@soton.ac.uk
#SBATCH --mail-type=END,FAIL
#SBATCH --array=0-27

# 28 tasks = one per country, day 271 only.
# CAP_SCALE and TOL are passed via --export when submitting.
#
# Submit all 6 combinations (run from the project root on Iridis):
#
#   sbatch --export=ALL,TOL=0.01,CAP_SCALE=0.5  code/run_cap_sensitivity_day271.slurm
#   sbatch --export=ALL,TOL=0.05,CAP_SCALE=0.5  code/run_cap_sensitivity_day271.slurm
#   sbatch --export=ALL,TOL=0.10,CAP_SCALE=0.5  code/run_cap_sensitivity_day271.slurm
#   sbatch --export=ALL,TOL=0.01,CAP_SCALE=2.0  code/run_cap_sensitivity_day271.slurm
#   sbatch --export=ALL,TOL=0.05,CAP_SCALE=2.0  code/run_cap_sensitivity_day271.slurm
#   sbatch --export=ALL,TOL=0.10,CAP_SCALE=2.0  code/run_cap_sensitivity_day271.slurm
#
# Output files: battery_{COUNTRY}_day_271_tol_{TOL}_{50|200}_installed_capacity.csv
# Move to battery_results/50_installed_capacity/ or battery_results/200_installed_capacity/
# before running analysis scripts.

cd /iridisfs/home/mj1v21/Wind_pv_synergy

TOL=${TOL:-0.05}
CAP_SCALE=${CAP_SCALE:-1.0}
COUNTRY_IDX=$SLURM_ARRAY_TASK_ID

echo "Task $SLURM_ARRAY_TASK_ID -> country_idx=$COUNTRY_IDX, forecast_day=271, tol=$TOL, cap_scale=$CAP_SCALE"

# Sanity check: confirm scenario files exist before launching Gurobi
echo "--- Checking scenario_results/day_271/ ---"
ls scenario_results/day_271/ | head -5

python code/parallel_battery_cap_sensitivity.py \
    --index $COUNTRY_IDX \
    --tol $TOL \
    --forecast_day 271 \
    --cap_scale $CAP_SCALE
