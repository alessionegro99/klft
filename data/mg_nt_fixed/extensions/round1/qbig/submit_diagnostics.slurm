#!/bin/bash -l
#SBATCH --job-name=mgnt-ext1-diag
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --array=0-1
#SBATCH --output=/qbigwork2/negro/dirichlet/logs/mgnt-ext1-diag-%A_%a.out

set -euo pipefail
exec bash /qbigwork2/negro/dirichlet/campaigns/mg_nt_fixed/extensions/round1/qbig/analyze_extension.sh \
    diagnose "${SLURM_ARRAY_TASK_ID}"
