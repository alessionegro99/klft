#!/bin/bash -l
#SBATCH --job-name=bntf-pcm-fit
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --array=0-1
#SBATCH --output=/qbigwork2/negro/dirichlet/logs/bntf-pcm-fit-%A_%a.out

set -euo pipefail
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
exec bash /qbigwork2/negro/dirichlet/project/data/beta_nt_fixed/qbig/reanalyse_pcm.sh \
    "${SLURM_ARRAY_TASK_ID}"
