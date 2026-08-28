#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 2 || ("$1" != diagnose && "$1" != analyse) || ! "$2" =~ ^[01]$ ]]; then
    echo "usage: $0 diagnose|analyse PRIOR_INDEX" >&2
    exit 2
fi
mode="$1"
prior_index="$2"

remote_root=/qbigwork2/negro/dirichlet
project="${remote_root}/project"
campaign="${remote_root}/campaigns/mg_nt_fixed"
prior_points="${campaign}/prior_points.tsv"
uv="${remote_root}/bin/uv"
export UV_CACHE_DIR="${remote_root}/.uv-cache"

line=$(sed -n "$((prior_index + 2))p" "${prior_points}")
IFS=$'\t' read -r recorded_index target_c physical_nt _stored_l2 ns beta \
    sweeps ensemble seeds source_campaign _source_analysis <<< "${line}"
[[ "${recorded_index}" -eq "${prior_index}" ]] || {
    echo "prior index mismatch" >&2
    exit 1
}
IFS=',' read -r -a seed_values <<< "${seeds}"
[[ "${#seed_values[@]}" -eq 4 ]] || {
    echo "${ensemble}: expected four prior seeds" >&2
    exit 1
}
data="${remote_root}/data/${ensemble}"
analysis="${remote_root}/analysis/mg_nt_fixed/audit/${ensemble}"
chain_arguments=()
for seed in "${seed_values[@]}"; do
    directory="${data}/seed_${seed}"
    chain_arguments+=(--chain-directory "${directory}")
    for file in holonomy_summary.out holonomy_correlator.out \
                plaquette_profile.out run_metadata.txt; do
        [[ -f "${directory}/${file}" ]] || {
            echo "missing ${directory}/${file}" >&2
            exit 1
        }
    done
    count=$(awk '!/^#/ {n++} END {print n+0}' "${directory}/holonomy_summary.out")
    [[ "${count}" -eq "${sweeps}" ]] || {
        echo "${directory}: incomplete prior history" >&2
        exit 1
    }
done

mkdir -p "${analysis}"
cd "${project}"
common=(
    --ensemble-dir "${data}"
    "${chain_arguments[@]}"
    --nt "${physical_nt}"
    --ns "${ns}"
    --beta "${beta}"
    --purpose prior_information_audit
    --campaign-type nonperturbative_mg_nt
    --target-c "${target_c}"
    --minimum-cut 500
)
if [[ "${mode}" == diagnose ]]; then
    "${uv}" run --frozen python scripts/mg_nt_fixed_analysis.py \
        "${common[@]}" \
        --diagnostics-only \
        --output "${analysis}/thermalization_diagnostics.json" \
        > "${analysis}/thermalization_diagnostics.log"
else
    diagnostics="${analysis}/thermalization_diagnostics.json"
    [[ -f "${diagnostics}" ]] || {
        echo "missing inspected diagnostics ${diagnostics}" >&2
        exit 1
    }
    "${uv}" run --frozen python scripts/mg_nt_fixed_analysis.py \
        "${common[@]}" \
        --thermalization-diagnostics "${diagnostics}" \
        --bootstrap-samples 1000 \
        --bootstrap-seed "$((20260824 + prior_index))" \
        --output "${analysis}/analysis.json" \
        --covariance-output "${analysis}/covariance.npz" \
        > "${analysis}/analysis.log"
fi
