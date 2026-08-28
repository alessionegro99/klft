#!/bin/bash -l
#SBATCH --job-name=mgnt-ext2-tuning
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/qbigwork2/negro/dirichlet/logs/mgnt-ext2-tuning-%j.out

set -euo pipefail
remote_root=/qbigwork2/negro/dirichlet
project="${remote_root}/project"
campaign="${remote_root}/campaigns/mg_nt_fixed"
round1="${campaign}/extensions/round1"
round2="${campaign}/extensions/round2"
results="${remote_root}/analysis/mg_nt_fixed/pilot"
round1_results="${remote_root}/analysis/mg_nt_fixed/extensions/round1"
round2_results="${remote_root}/analysis/mg_nt_fixed/extensions/round2"
audit="${remote_root}/analysis/mg_nt_fixed/audit"
output="${remote_root}/analysis/mg_nt_fixed/tuning_round2"
expected_code_sha=3c760e2eedfc95adaae5b9267648b53e9a8e43b3b6e8fae410c8b22e4eb22fba
export UV_CACHE_DIR="${remote_root}/.uv-cache"
[[ "$(find "${round2_results}" -mindepth 2 -maxdepth 2 -name analysis.json | wc -l)" -eq 1 ]]
[[ ! -e "${output}/tuning_summary.json" ]]
[[ "$(sha256sum "${project}/scripts/mg_nt_fixed_tuning.py" | awk '{print $1}')" == "${expected_code_sha}" ]]
cd "${project}"
exec "${remote_root}/bin/uv" run --frozen python scripts/mg_nt_fixed_tuning.py \
    --manifest "${campaign}/manifest.tsv" --prior-points "${campaign}/prior_points.tsv" \
    --results "${results}" --audit-results "${audit}" \
    --extension-manifest "${round1}/manifest.tsv" --extension-results "${round1_results}" \
    --extension-manifest "${round2}/manifest.tsv" --extension-results "${round2_results}" \
    --output "${output}"
