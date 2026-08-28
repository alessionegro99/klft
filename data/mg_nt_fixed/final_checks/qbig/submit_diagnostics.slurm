#!/bin/bash -l
#SBATCH --job-name=mgnt-final-diag
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --array=0-2
#SBATCH --output=/qbigwork2/negro/dirichlet/logs/mgnt-final-diag-%A_%a.out

set -euo pipefail
exec bash /qbigwork2/negro/dirichlet/campaigns/mg_nt_fixed/final_checks/qbig/analyze_replacements.sh diagnose "${SLURM_ARRAY_TASK_ID}"
