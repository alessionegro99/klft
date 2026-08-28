#!/bin/bash -l
#SBATCH --job-name=bntf-analysis
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-9
#SBATCH --output=/qbigwork2/negro/dirichlet/logs/bntf-analysis-%A_%a.out

set -euo pipefail
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
exec bash /qbigwork2/negro/dirichlet/project/data/beta_nt_fixed/production/cadence25/qbig/analyse_ensemble.sh \
    analyse "${SLURM_ARRAY_TASK_ID}"
