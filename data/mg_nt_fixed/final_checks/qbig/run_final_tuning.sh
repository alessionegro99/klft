#!/usr/bin/env bash
set -euo pipefail

root=/qbigwork2/negro/dirichlet
project="${root}/project"
campaign="${root}/campaigns/mg_nt_fixed"
checks="${campaign}/final_checks"
pilot_results="${root}/analysis/mg_nt_fixed/pilot"
replacement_results="${root}/analysis/mg_nt_fixed/final_checks"
bootstrap_results="${replacement_results}/bootstrap_10000"
summary="${replacement_results}/final_checks_summary.json"
tuning="${root}/analysis/mg_nt_fixed/tuning_final"
uv="${root}/bin/uv"
export UV_CACHE_DIR="${root}/.uv-cache"

[[ "$(find "${replacement_results}" -mindepth 2 -maxdepth 2 -name analysis.json | wc -l)" -eq 3 ]]
[[ "$(find "${bootstrap_results}" -mindepth 2 -maxdepth 2 -name analysis.json | wc -l)" -eq 3 ]]
[[ ! -e "${summary}" && ! -e "${tuning}/tuning_summary.json" ]]
while read -r name expected; do
    [[ "$(sha256sum "${project}/${name}" | awk '{print $1}')" == "${expected}" ]]
done < "${checks}/code_sha256.tsv"
cd "${project}"
"${uv}" run --frozen python scripts/mg_nt_fixed_final_checks.py \
    --pilot-results "${pilot_results}" --replacement-results "${replacement_results}" \
    --output "${summary}" > "${replacement_results}/final_checks_summary.log"
"${uv}" run --frozen python -c \
    'import json,sys; paths=sys.argv[1:]; assert all(json.load(open(path))["bootstrap_samples"] == 10000 for path in paths)' \
    "${bootstrap_results}"/*/analysis.json
"${uv}" run --frozen python scripts/mg_nt_fixed_tuning.py \
    --manifest "${campaign}/manifest.tsv" \
    --prior-points "${campaign}/prior_points.tsv" \
    --results "${pilot_results}" --audit-results "${root}/analysis/mg_nt_fixed/audit" \
    --extension-manifest "${campaign}/extensions/round1/manifest.tsv" \
    --extension-results "${root}/analysis/mg_nt_fixed/extensions/round1" \
    --extension-manifest "${campaign}/extensions/round2/manifest.tsv" \
    --extension-results "${root}/analysis/mg_nt_fixed/extensions/round2" \
    --extension-manifest "${checks}/manifest.tsv" \
    --extension-results "${replacement_results}" \
    --extension-manifest "${checks}/reanalysis_manifest.tsv" \
    --extension-results "${bootstrap_results}" \
    --output "${tuning}"
"${uv}" run --frozen python -c \
    'import json,sys; d=json.load(open(sys.argv[1])); assert d["all_nine_tunings_accepted"]' \
    "${tuning}/tuning_summary.json"
