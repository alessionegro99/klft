#!/bin/bash -l
#SBATCH --job-name=mgnt-analysis
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --array=0-24
#SBATCH --output=/qbigwork2/negro/dirichlet/logs/mgnt-analysis-%A_%a.out

set -euo pipefail
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
exec /qbigwork2/negro/dirichlet/project/data/mg_nt_fixed/pilot/qbig/analyze_pilot.sh \
    analyse "${SLURM_ARRAY_TASK_ID}"
