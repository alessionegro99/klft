#!/bin/bash -l
#SBATCH --job-name=bntf-plots
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/qbigwork2/negro/dirichlet/logs/bntf-plots-%j.out

set -euo pipefail
remote_root=/qbigwork2/negro/dirichlet
project="${remote_root}/project"
output="${remote_root}/analysis/beta_nt_fixed/continuum"
uv="${remote_root}/bin/uv"
export UV_CACHE_DIR="${remote_root}/.uv-cache"
export MPLCONFIGDIR="${remote_root}/.matplotlib"
mkdir -p "${output}/plots" "${MPLCONFIGDIR}"
cd "${project}"
"${uv}" run --frozen python scripts/beta_nt_fixed/plot_continuum.py \
    --summary "${output}/continuum_summary.json" \
    --output "${output}/plots"
