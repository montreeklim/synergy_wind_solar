#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=naive_model_1000
#SBATCH --output=output.%A_%a.naive_model_1000
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --mail-user=mj1v21@soton.ac.uk
#SBATCH --mail-type=END,FAIL
#SBATCH --array=0-27

# 28 tasks = one per country.
# Each task runs all 4 forecast days x 3 tolerances = 12 optimizations,
# each on a pooled 1000-scenario matrix (10 sets x 100 scenarios).

cd /iridisfs/home/mj1v21/Wind_pv_synergy

echo "Task $SLURM_ARRAY_TASK_ID -> country_idx=$SLURM_ARRAY_TASK_ID"

python code/naive_optimization_models_1000.py --index $SLURM_ARRAY_TASK_ID
