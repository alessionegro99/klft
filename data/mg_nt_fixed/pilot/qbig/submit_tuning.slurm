#!/bin/bash -l
#SBATCH --job-name=mgnt-tuning
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/qbigwork2/negro/dirichlet/logs/mgnt-tuning-%j.out

set -euo pipefail
remote_root=/qbigwork2/negro/dirichlet
project="${remote_root}/project"
campaign="${remote_root}/campaigns/mg_nt_fixed"
results="${remote_root}/analysis/mg_nt_fixed/pilot"
audit="${remote_root}/analysis/mg_nt_fixed/audit"
output="${remote_root}/analysis/mg_nt_fixed/tuning"
export UV_CACHE_DIR="${remote_root}/.uv-cache"

[[ "$(find "${results}" -mindepth 2 -maxdepth 2 -name analysis.json | wc -l)" -eq 25 ]] || {
    echo "expected 25 pilot analysis results" >&2
    exit 1
}
[[ "$(find "${audit}" -mindepth 2 -maxdepth 2 -name analysis.json | wc -l)" -eq 2 ]] || {
    echo "expected two prior-information audit results" >&2
    exit 1
}
cd "${project}"
exec "${remote_root}/bin/uv" run --frozen python scripts/mg_nt_fixed_tuning.py \
    --manifest "${campaign}/manifest.tsv" \
    --prior-points "${campaign}/prior_points.tsv" \
    --results "${results}" \
    --audit-results "${audit}" \
    --output "${output}"
