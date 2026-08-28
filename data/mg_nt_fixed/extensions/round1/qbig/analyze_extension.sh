#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 2 || ! "$1" =~ ^(diagnose|analyse)$ || ! "$2" =~ ^[01]$ ]]; then
    echo "usage: $0 diagnose|analyse ENSEMBLE_INDEX" >&2
    exit 2
fi
mode="$1"
ensemble_index="$2"
remote_root=/qbigwork2/negro/dirichlet
project="${remote_root}/project"
campaign="${remote_root}/campaigns/mg_nt_fixed/extensions/round1"
manifest="${campaign}/manifest.tsv"
uv="${remote_root}/bin/uv"
export UV_CACHE_DIR="${remote_root}/.uv-cache"

line=$(sed -n "$((4 * ensemble_index + 2))p" "${manifest}")
IFS=$'\t' read -r _ campaign_type target_c run _ purpose physical_nt _ ns beta \
    _ _ sweeps _ _ _ _ _ _ <<< "${line}"
ensemble=$(basename "$(dirname "${run}")")
output="${remote_root}/analysis/mg_nt_fixed/extensions/round1/${ensemble}"
diagnostics="${output}/thermalization_diagnostics.json"
chain_arguments=()
for offset in 0 1 2 3; do
    chain_run=$(sed -n "$((4 * ensemble_index + offset + 2))p" "${manifest}" | cut -f4)
    [[ "$(basename "$(dirname "${chain_run}")")" == "${ensemble}" ]]
    chain="${remote_root}/data/${chain_run}"
    [[ -f "${chain}/run_complete.txt" ]]
    chain_arguments+=(--chain-directory "${chain}")
done

mkdir -p "${output}"
cd "${project}"
common=(
    --ensemble-dir "${remote_root}/data/mg_nt_fixed/extensions/round1/${ensemble}"
    "${chain_arguments[@]}"
    --nt "${physical_nt}"
    --ns "${ns}"
    --beta "${beta}"
    --purpose "${purpose}"
    --campaign-type "${campaign_type}"
    --target-c "${target_c}"
    --minimum-cut 500
)
if [[ "${mode}" == diagnose ]]; then
    if [[ -e "${diagnostics}" ]]; then
        [[ -s "${diagnostics}" ]]
        echo "keeping completed diagnostics ${diagnostics}"
        exit 0
    fi
    "${uv}" run --frozen python scripts/mg_nt_fixed_analysis.py \
        "${common[@]}" --diagnostics-only --output "${diagnostics}" \
        > "${output}/thermalization_diagnostics.log"
else
    [[ -s "${diagnostics}" ]]
    if [[ -e "${output}/analysis.json" || -e "${output}/covariance.npz" ]]; then
        [[ -s "${output}/analysis.json" && -s "${output}/covariance.npz" ]]
        echo "keeping completed analysis ${output}"
        exit 0
    fi
    "${uv}" run --frozen python scripts/mg_nt_fixed_analysis.py \
        "${common[@]}" --thermalization-diagnostics "${diagnostics}" \
        --bootstrap-samples 1000 --bootstrap-seed "$((20260750 + ensemble_index))" \
        --output "${output}/analysis.json" \
        --covariance-output "${output}/covariance.npz" \
        > "${output}/analysis.log"
fi
