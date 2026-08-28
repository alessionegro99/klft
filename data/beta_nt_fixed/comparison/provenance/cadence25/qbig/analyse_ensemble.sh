#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 2 || ! "$1" =~ ^(diagnose|analyse)$ || ! "$2" =~ ^[0-9]$ ]]; then
    echo "usage: $0 diagnose|analyse ENSEMBLE_INDEX" >&2
    exit 2
fi
mode="$1"
ensemble_index="$2"
[[ "${ensemble_index}" -le 9 ]]

remote_root=/qbigwork2/negro/dirichlet
project="${remote_root}/project"
campaign="${remote_root}/campaigns/beta_nt_fixed/production/cadence25"
manifest="${campaign}/manifest.tsv"
uv="${remote_root}/bin/uv"
export UV_CACHE_DIR="${remote_root}/.uv-cache"

first_array_index=$((4 * ensemble_index))
line=$(sed -n "$((first_array_index + 2))p" "${manifest}")
IFS=$'\t' read -r array_index run physical_nt stored_l2 ns beta \
    beta_over_nt g3_squared_l expected_xi boundary_max_r chain_index seed \
    sweeps base_interval profile_interval boundary_interval input_sha \
    <<< "${line}"
[[ "${array_index}" -eq "${first_array_index}" && "${chain_index}" -eq 0 ]]
ensemble="${run%%/*}"
output="${remote_root}/analysis/beta_nt_fixed/production/${ensemble}"
diagnostics="${output}/thermalization_diagnostics.json"

chain_arguments=()
for offset in 0 1 2 3; do
    chain_line=$(sed -n "$((first_array_index + offset + 2))p" "${manifest}")
    chain_run=$(printf '%s' "${chain_line}" | cut -f2)
    chain_ensemble="${chain_run%%/*}"
    [[ "${chain_ensemble}" == "${ensemble}" ]]
    chain="${remote_root}/data/beta_nt_fixed/production/${chain_run}"
    [[ -f "${chain}/run_complete.txt" ]]
    chain_arguments+=(--chain-directory "${chain}")
done

mkdir -p "${output}"
cd "${project}"
common=(
    --ensemble-dir "${remote_root}/data/beta_nt_fixed/production/${ensemble}"
    "${chain_arguments[@]}"
    --nt "${physical_nt}"
    --ns "${ns}"
    --beta "${beta}"
    --purpose "fixed-g3-squared-L continuum production"
    --minimum-cut 500
)
if [[ "${mode}" == diagnose ]]; then
    if [[ -e "${diagnostics}" ]]; then
        [[ -s "${diagnostics}" ]]
        echo "keeping completed diagnostics ${diagnostics}"
        exit 0
    fi
    [[ ! -e "${diagnostics}" ]]
    "${uv}" run --frozen python scripts/beta_nt_fixed/analysis.py \
        "${common[@]}" \
        --diagnostics-only \
        --output "${diagnostics}" \
        > "${output}/diagnostics.log"
else
    [[ -f "${diagnostics}" ]]
    if [[ -e "${output}/analysis.json" || -e "${output}/covariance.npz" ]]; then
        [[ -s "${output}/analysis.json" && -s "${output}/covariance.npz" ]]
        echo "keeping completed analysis ${output}"
        exit 0
    fi
    [[ ! -e "${output}/analysis.json" && ! -e "${output}/covariance.npz" ]]
    "${uv}" run --frozen python scripts/beta_nt_fixed/analysis.py \
        "${common[@]}" \
        --thermalization-diagnostics "${diagnostics}" \
        --bootstrap-samples 1000 \
        --bootstrap-seed "$((20260900 + ensemble_index))" \
        --output "${output}/analysis.json" \
        --covariance-output "${output}/covariance.npz" \
        > "${output}/analysis.log"
fi
