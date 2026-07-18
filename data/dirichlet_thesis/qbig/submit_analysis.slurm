#!/bin/bash -l
#SBATCH --job-name=dir-analysis
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --array=0-2
#SBATCH --output=/qbigwork2/negro/dirichlet/logs/analysis-%A_%a.out

set -euo pipefail

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

thicknesses=(2 6 12)
nt="${thicknesses[SLURM_ARRAY_TASK_ID]}"
exec /qbigwork2/negro/dirichlet/project/data/dirichlet_thesis/qbig/analyze_ensemble.sh \
    analyse "${nt}" 500
