#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 1 || ! "$1" =~ ^(diagnose|analyse)$ ]]; then
    echo "usage: $0 diagnose|analyse" >&2
    exit 2
fi

mode="$1"
remote_root=/qbigwork2/negro/dirichlet
project="${remote_root}/project"
campaign="${remote_root}/campaigns/beta_nt_fixed/production/cadence25"
manifest="${campaign}/manifest.tsv"
output="${remote_root}/analysis/beta_nt_fixed/preliminary_snapshot_48300_v2/Nt_5_Ns_688_b_10p5"
diagnostics="${output}/thermalization_diagnostics.json"
maximum_sweep=48300
expected_analysis_sha=8bd3ae8041623ba579a75affca3b333d0fb823314c384c29c60a8eaa34a3deea
expected_loader_sha=8ac57279656c44fb7ec09dcd908243c95279ae783a2685234cf0da8c4a80da24
uv="${remote_root}/bin/uv"
export UV_CACHE_DIR="${remote_root}/.uv-cache"

[[ "$(sha256sum "${project}/scripts/beta_nt_fixed/analysis.py" | awk '{print $1}')" == "${expected_analysis_sha}" ]]
[[ "$(sha256sum "${project}/scripts/dirichlet_analysis.py" | awk '{print $1}')" == "${expected_loader_sha}" ]]

chain_arguments=()
for array_index in 36 37 38 39; do
    line=$(sed -n "$((array_index + 2))p" "${manifest}")
    IFS=$'\t' read -r _ run physical_nt _ ns beta _ _ _ _ _ _ _ _ _ _ _ <<< "${line}"
    [[ "${physical_nt}" -eq 5 && "${ns}" -eq 688 && "${beta}" == 10.5 ]]
    chain="${remote_root}/data/beta_nt_fixed/production/${run}"
    for filename in holonomy_summary.out holonomy_correlator.out \
        plaquette_profile.out boundary_wilson.out; do
        last_step=$(tail -n 1 "${chain}/${filename}" | awk '{print $1}')
        [[ "${last_step}" =~ ^[0-9]+$ && "${last_step}" -ge "${maximum_sweep}" ]]
    done
    chain_arguments+=(--chain-directory "${chain}")
done

mkdir -p "${output}"
cd "${project}"
common=(
    --ensemble-dir "${remote_root}/data/beta_nt_fixed/production/Nt_5_Ns_688_b_10p5"
    "${chain_arguments[@]}"
    --nt 5
    --ns 688
    --beta 10.5
    --purpose "PRELIMINARY fixed-g3-squared-L timeout snapshot at sweeps <= 48300"
    --campaign-type beta_nt_fixed_preliminary_snapshot
    --minimum-cut 500
    --maximum-sweep "${maximum_sweep}"
)

if [[ "${mode}" == diagnose ]]; then
    [[ ! -e "${diagnostics}" ]]
    "${uv}" run --frozen python scripts/beta_nt_fixed/analysis.py \
        "${common[@]}" \
        --diagnostics-only \
        --output "${diagnostics}" \
        > "${output}/diagnostics.log"
else
    [[ -s "${diagnostics}" ]]
    [[ ! -e "${output}/analysis.json" && ! -e "${output}/covariance.npz" ]]
    "${uv}" run --frozen python scripts/beta_nt_fixed/analysis.py \
        "${common[@]}" \
        --thermalization-diagnostics "${diagnostics}" \
        --bootstrap-samples 1000 \
        --bootstrap-seed 20260910 \
        --output "${output}/analysis.json" \
        --covariance-output "${output}/covariance.npz" \
        > "${output}/analysis.log"
fi
