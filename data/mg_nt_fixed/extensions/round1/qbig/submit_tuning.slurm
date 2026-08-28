#!/bin/bash -l
#SBATCH --job-name=mgnt-ext1-tuning
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/qbigwork2/negro/dirichlet/logs/mgnt-ext1-tuning-%j.out

set -euo pipefail
remote_root=/qbigwork2/negro/dirichlet
project="${remote_root}/project"
campaign="${remote_root}/campaigns/mg_nt_fixed"
extension="${campaign}/extensions/round1"
results="${remote_root}/analysis/mg_nt_fixed/pilot"
extension_results="${remote_root}/analysis/mg_nt_fixed/extensions/round1"
audit="${remote_root}/analysis/mg_nt_fixed/audit"
output="${remote_root}/analysis/mg_nt_fixed/tuning_round1"
expected_code_sha=2ed9f91bdccf49bbb506eee982852d08d59e0ab1646802d2f9df62b89491e8a0
export UV_CACHE_DIR="${remote_root}/.uv-cache"

[[ "$(find "${extension_results}" -mindepth 2 -maxdepth 2 -name analysis.json | wc -l)" -eq 2 ]]
[[ ! -e "${output}/tuning_summary.json" ]]
[[ "$(sha256sum "${project}/scripts/mg_nt_fixed_tuning.py" | awk '{print $1}')" == "${expected_code_sha}" ]]
cd "${project}"
exec "${remote_root}/bin/uv" run --frozen python scripts/mg_nt_fixed_tuning.py \
    --manifest "${campaign}/manifest.tsv" \
    --prior-points "${campaign}/prior_points.tsv" \
    --results "${results}" \
    --audit-results "${audit}" \
    --extension-manifest "${extension}/manifest.tsv" \
    --extension-results "${extension_results}" \
    --output "${output}"
