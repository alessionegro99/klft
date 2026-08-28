#!/bin/bash -l
#SBATCH --job-name=bntf-first-stage
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/qbigwork2/negro/dirichlet/logs/bntf-first-stage-%j.out

set -euo pipefail
remote_root=/qbigwork2/negro/dirichlet
project="${remote_root}/project"
output="${remote_root}/analysis/beta_nt_fixed/first_stage_clean_20260728"
uv="${remote_root}/bin/uv"
export UV_CACHE_DIR="${remote_root}/.uv-cache"
export MPLCONFIGDIR="${remote_root}/.matplotlib"
if [[ -e "${output}" ]]; then
    echo "Refusing to overwrite ${output}" >&2
    exit 1
fi
mkdir -p "${output}" "${MPLCONFIGDIR}"
cd "${project}"
"${uv}" run --frozen python scripts/beta_nt_fixed/plot_first_stage.py \
    --manifest data/beta_nt_fixed/comparison/first_stage_manifest.tsv \
    --output-root "${output}"
