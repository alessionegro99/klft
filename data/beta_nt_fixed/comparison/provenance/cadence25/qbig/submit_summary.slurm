#!/bin/bash -l
#SBATCH --job-name=bntf-summary
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/qbigwork2/negro/dirichlet/logs/bntf-summary-%j.out

set -euo pipefail
remote_root=/qbigwork2/negro/dirichlet
project="${remote_root}/project"
output="${remote_root}/analysis/beta_nt_fixed/continuum"
uv="${remote_root}/bin/uv"
export UV_CACHE_DIR="${remote_root}/.uv-cache"
mkdir -p "${output}"
[[ ! -e "${output}/continuum_summary.json" ]]
cd "${project}"
"${uv}" run --frozen python scripts/beta_nt_fixed/continuum_summary.py \
    --selection data/beta_nt_fixed/continuum_points.tsv \
    --analysis-root "${remote_root}/analysis" \
    --allow-incomplete \
    --output "${output}/continuum_summary.json" \
    --covariance-output "${output}/continuum_bootstrap.npz" \
    > "${output}/summary.log"
