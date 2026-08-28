#!/bin/bash -l
#SBATCH --job-name=mgnt-plots
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=/qbigwork2/negro/dirichlet/logs/mgnt-plots-%j.out

set -euo pipefail
remote_root=/qbigwork2/negro/dirichlet
project="${remote_root}/project"
tuning="${remote_root}/analysis/mg_nt_fixed/tuning"
export UV_CACHE_DIR="${remote_root}/.uv-cache"
cd "${project}"
exec "${remote_root}/bin/uv" run --frozen python scripts/mg_nt_fixed_plot.py \
    --summary "${tuning}/tuning_summary.json" \
    --output "${remote_root}/analysis/mg_nt_fixed/plots"
