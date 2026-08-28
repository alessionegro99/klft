#!/bin/bash -l
#SBATCH --job-name=bntf-reanalysis
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --array=0-4
#SBATCH --output=/qbigwork2/negro/dirichlet/logs/bntf-reanalysis-%A_%a.out

set -euo pipefail
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
exec /qbigwork2/negro/dirichlet/project/data/beta_nt_fixed/qbig/reanalyse_existing.sh \
    "${SLURM_ARRAY_TASK_ID}"
