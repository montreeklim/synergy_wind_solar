#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=naive_model
#SBATCH --output=output.%A_%a.naive_model
#SBATCH --time=08:00:00
#SBATCH --mem=16G
#SBATCH --mail-user=mj1v21@soton.ac.uk
#SBATCH --mail-type=END,FAIL
#SBATCH --array=0-27

# 28 tasks = one per country.
# Each task runs all 4 forecast days x 3 tolerances x 10 sets = 120 optimizations.

cd /iridisfs/home/mj1v21/Wind_pv_synergy

echo "Task $SLURM_ARRAY_TASK_ID -> country_idx=$SLURM_ARRAY_TASK_ID"

python code/naive_optimization_models.py --index $SLURM_ARRAY_TASK_ID
