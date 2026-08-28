#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 1 || ! "$1" =~ ^[0-2]$ ]]; then
    echo "usage: $0 ENSEMBLE_INDEX" >&2
    exit 2
fi
index="$1"
root=/qbigwork2/negro/dirichlet
project="${root}/project"
manifest="${root}/campaigns/mg_nt_fixed/final_checks/reanalysis_manifest.tsv"
uv="${root}/bin/uv"
export UV_CACHE_DIR="${root}/.uv-cache"
line=$(sed -n "$((4 * index + 2))p" "${manifest}")
IFS=$'\t' read -r _ campaign_type target_c run _ purpose physical_nt _ ns beta \
    _ _ _ _ _ _ _ _ _ <<< "${line}"
ensemble=$(basename "$(dirname "${run}")")
diagnostics="${root}/analysis/mg_nt_fixed/pilot/${ensemble}/thermalization_diagnostics.json"
accepted="${root}/analysis/mg_nt_fixed/pilot/${ensemble}/analysis.json"
output="${root}/analysis/mg_nt_fixed/final_checks/bootstrap_10000/${ensemble}"
[[ -s "${diagnostics}" && -s "${accepted}" && ! -e "${output}/analysis.json" && ! -e "${output}/covariance.npz" ]]
chain_arguments=()
for offset in 0 1 2 3; do
    chain_run=$(sed -n "$((4 * index + offset + 2))p" "${manifest}" | cut -f4)
    [[ "$(basename "$(dirname "${chain_run}")")" == "${ensemble}" ]]
    chain="${root}/data/${chain_run}"
    for file in holonomy_summary.out holonomy_correlator.out plaquette_profile.out; do
        [[ -f "${chain}/${file}" ]]
    done
    chain_arguments+=(--chain-directory "${chain}")
done
mkdir -p "${output}"
cd "${project}"
"${uv}" run --frozen python scripts/mg_nt_fixed_analysis.py \
    --ensemble-dir "${root}/data/mg_nt_fixed/${ensemble}" \
    "${chain_arguments[@]}" --nt "${physical_nt}" --ns "${ns}" --beta "${beta}" \
    --purpose "${purpose}" --campaign-type "${campaign_type}" --target-c "${target_c}" \
    --minimum-cut 500 --thermalization-diagnostics "${diagnostics}" \
    --bootstrap-samples 10000 --bootstrap-seed "$((20261020 + index))" \
    --output "${output}/analysis.json" --covariance-output "${output}/covariance.npz" \
    > "${output}/analysis.log"
