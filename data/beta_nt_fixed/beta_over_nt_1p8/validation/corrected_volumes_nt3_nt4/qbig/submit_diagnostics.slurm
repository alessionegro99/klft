#!/bin/bash -l
#SBATCH --job-name=bntf-cv-diag
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=04:00:00
#SBATCH --array=0-1
#SBATCH --output=/qbigwork2/negro/dirichlet/logs/bntf-cv-diag-%A_%a.out

set -euo pipefail
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
exec /qbigwork2/negro/dirichlet/project/data/beta_nt_fixed/beta_over_nt_1p8/corrected_volumes_nt3_nt4/qbig/analyze.sh \
    diagnose "${SLURM_ARRAY_TASK_ID}"
