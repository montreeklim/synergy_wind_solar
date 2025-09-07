#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=40
#SBATCH --job-name=battery_model
#SBATCH --output=output.%A_%a.battery_model
#SBATCH --time=24:00:00
#SBATCH --mail-user=mj1v21@soton.ac.uk
#SBATCH --mail-type=END,FAIL
#SBATCH --array=0-27

python parallel_battery_model_cap.py --index $SLURM_ARRAY_TASK_ID