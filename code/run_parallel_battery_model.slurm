#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=battery_model
#SBATCH --output=output.%A_%a.battery_model
#SBATCH --time=24:00:00
#SBATCH --mem=50G
#SBATCH --mail-user=mj1v21@soton.ac.uk
#SBATCH --mail-type=END,FAIL
#SBATCH --array=0-111

# 112 tasks = 28 countries x 4 forecast days, all running in parallel.
# Task ID layout: task = day_idx * 28 + country_idx
#   day_idx    0->271, 1->301, 2->332, 3->362
#   country_idx 0->AT ... 27->UK

cd /iridisfs/home/mj1v21/Wind_pv_synergy

DAYS=(271 301 332 362)
COUNTRY_IDX=$((SLURM_ARRAY_TASK_ID % 28))
DAY_IDX=$((SLURM_ARRAY_TASK_ID / 28))
FORECAST_DAY=${DAYS[$DAY_IDX]}

# TOL is passed via --export when submitting, e.g.:
#   sbatch --export=ALL,TOL=0.01 run_parallel_battery_model.slurm
#   sbatch --export=ALL,TOL=0.05 run_parallel_battery_model.slurm
#   sbatch --export=ALL,TOL=0.10 run_parallel_battery_model.slurm
TOL=${TOL:-0.05}

echo "Task $SLURM_ARRAY_TASK_ID -> country_idx=$COUNTRY_IDX, forecast_day=$FORECAST_DAY, tol=$TOL"

python code/parallel_battery_model_cap.py --index $COUNTRY_IDX --tol $TOL --forecast_day $FORECAST_DAY
