#!/bin/bash -l
#SBATCH --job-name=mgnt-final-analysis
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --array=0-2
#SBATCH --output=/qbigwork2/negro/dirichlet/logs/mgnt-final-analysis-%A_%a.out

set -euo pipefail
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
exec bash /qbigwork2/negro/dirichlet/campaigns/mg_nt_fixed/final_checks/qbig/analyze_replacements.sh analyse "${SLURM_ARRAY_TASK_ID}"
