#!/bin/bash
#SBATCH --job-name=lynx_hare_sim
#SBATCH --output=../BatchOutput/finalSimulation_%A_%a.out
#SBATCH --error=../BatchOutput/finalSimulation_%A_%a.err
#SBATCH --account=pi-seppekuehn
#SBATCH --partition=caslake
#SBATCH --nodes=1
#SBATCH --array=0-69
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --time=36:00:00

module load python

source /software/python-miniforge-24.1.2-el8-x86_64/etc/profile.d/conda.sh
conda activate /home/adomenig/.conda/envs/solitary_mammals

PARAM_FILE="/project/jnirody/Alissa/Data/fitParameters.csv"

echo "Running task ${SLURM_ARRAY_TASK_ID}"

python ../MC_Simulations/finalSimulations.py \
    --parameters "$PARAM_FILE"

