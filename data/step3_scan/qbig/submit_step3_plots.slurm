#!/bin/bash -l
#SBATCH --job-name=step3-plots
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=00:30:00
#SBATCH --output=/qbigwork2/negro/dirichlet/logs/step3-plots-%j.out

set -euo pipefail
remote_root=/qbigwork2/negro/dirichlet
export UV_CACHE_DIR="${remote_root}/.uv-cache"
cd "${remote_root}/project"
exec "${remote_root}/bin/uv" run --frozen python scripts/plot_step3_scan.py \
    --results "${remote_root}/analysis/step3_scan" \
    --output "${remote_root}/analysis/step3_scan/plots"
