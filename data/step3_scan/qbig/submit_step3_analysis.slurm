#!/bin/bash -l
#SBATCH --job-name=step3-analysis
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=08:00:00
#SBATCH --array=0-7%4
#SBATCH --output=/qbigwork2/negro/dirichlet/logs/step3-analysis-%A_%a.out

set -euo pipefail
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
exec /qbigwork2/negro/dirichlet/project/data/step3_scan/qbig/analyze_step3_ensemble.sh \
    analyse "${SLURM_ARRAY_TASK_ID}"
