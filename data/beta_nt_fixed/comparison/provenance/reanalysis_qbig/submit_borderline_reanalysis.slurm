#!/bin/bash -l
#SBATCH --job-name=bntf-fit10k
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --array=3-4
#SBATCH --output=/qbigwork2/negro/dirichlet/logs/bntf-fit10k-%A_%a.out

set -euo pipefail
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
exec bash /qbigwork2/negro/dirichlet/project/data/beta_nt_fixed/qbig/reanalyse_borderline.sh \
    "${SLURM_ARRAY_TASK_ID}"
