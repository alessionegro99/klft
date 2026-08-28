#!/bin/bash -l
#SBATCH --job-name=bntf-diag
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --array=0-7%4
#SBATCH --output=/qbigwork2/negro/dirichlet/logs/bntf-diag-%A_%a.out

set -euo pipefail
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
exec /qbigwork2/negro/dirichlet/project/data/beta_nt_fixed/pilot/qbig/analyze_ensemble.sh \
    diagnose "${SLURM_ARRAY_TASK_ID}"
