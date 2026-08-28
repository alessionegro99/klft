#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 1 || ! "$1" =~ ^(diagnose|analyse)$ ]]; then
    echo "usage: $0 diagnose|analyse" >&2
    exit 2
fi
mode="$1"
remote_root=/qbigwork2/negro/dirichlet
project="${remote_root}/project"
campaign="${remote_root}/campaigns/mg_nt_fixed/extensions/round2"
manifest="${campaign}/manifest.tsv"
uv="${remote_root}/bin/uv"
export UV_CACHE_DIR="${remote_root}/.uv-cache"
line=$(sed -n '2p' "${manifest}")
IFS=$'\t' read -r _ campaign_type target_c run _ purpose physical_nt _ ns beta \
    _ _ _ _ _ _ _ _ _ <<< "${line}"
ensemble=$(basename "$(dirname "${run}")")
output="${remote_root}/analysis/mg_nt_fixed/extensions/round2/${ensemble}"
diagnostics="${output}/thermalization_diagnostics.json"
chain_arguments=()
for offset in 0 1 2 3; do
    chain_run=$(sed -n "$((offset + 2))p" "${manifest}" | cut -f4)
    [[ "$(basename "$(dirname "${chain_run}")")" == "${ensemble}" ]]
    chain="${remote_root}/data/${chain_run}"
    [[ -f "${chain}/run_complete.txt" ]]
    chain_arguments+=(--chain-directory "${chain}")
done
mkdir -p "${output}"
cd "${project}"
common=(
    --ensemble-dir "${remote_root}/data/mg_nt_fixed/extensions/round2/${ensemble}"
    "${chain_arguments[@]}" --nt "${physical_nt}" --ns "${ns}" --beta "${beta}"
    --purpose "${purpose}" --campaign-type "${campaign_type}" --target-c "${target_c}"
    --minimum-cut 500
)
if [[ "${mode}" == diagnose ]]; then
    [[ ! -e "${diagnostics}" ]]
    "${uv}" run --frozen python scripts/mg_nt_fixed_analysis.py \
        "${common[@]}" --diagnostics-only --output "${diagnostics}" \
        > "${output}/thermalization_diagnostics.log"
else
    [[ -s "${diagnostics}" && ! -e "${output}/analysis.json" && ! -e "${output}/covariance.npz" ]]
    "${uv}" run --frozen python scripts/mg_nt_fixed_analysis.py \
        "${common[@]}" --thermalization-diagnostics "${diagnostics}" \
        --bootstrap-samples 1000 --bootstrap-seed 20261001 \
        --output "${output}/analysis.json" --covariance-output "${output}/covariance.npz" \
        > "${output}/analysis.log"
fi
