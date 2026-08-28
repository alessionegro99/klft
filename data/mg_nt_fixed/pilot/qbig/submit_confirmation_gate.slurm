#!/bin/bash -l
#SBATCH --job-name=mgnt-conf-g
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=/qbigwork2/negro/dirichlet/logs/mgnt-conf-g-%j.out

set -euo pipefail
remote_root=/qbigwork2/negro/dirichlet
project="${remote_root}/project"
confirmation_root="${remote_root}/analysis/mg_nt_fixed/confirmation"
confirmation=$(find "${confirmation_root}" -mindepth 2 -maxdepth 2 -name analysis.json)
[[ "$(wc -w <<< "${confirmation}")" -eq 1 ]] || {
    echo "expected exactly one confirmation result" >&2
    exit 1
}
export UV_CACHE_DIR="${remote_root}/.uv-cache"
cd "${project}"
exec "${remote_root}/bin/uv" run --frozen python scripts/mg_nt_fixed_confirmation.py \
    --tuning-summary "${remote_root}/analysis/mg_nt_fixed/tuning_final/tuning_summary.json" \
    --confirmation "${confirmation}" \
    --output "${confirmation_root}"
