#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 2 || ("$1" != diagnose && "$1" != analyse) || ! "$2" =~ ^([0-9]|1[0-9]|2[0-4])$ ]]; then
    echo "usage: $0 diagnose|analyse ENSEMBLE_INDEX" >&2
    exit 2
fi
mode="$1"
ensemble_index="$2"

remote_root=/qbigwork2/negro/dirichlet
project="${remote_root}/project"
campaign="${remote_root}/campaigns/mg_nt_fixed"
manifest="${campaign}/manifest.tsv"
uv="${remote_root}/bin/uv"
export UV_CACHE_DIR="${remote_root}/.uv-cache"

line=$(sed -n "$((4 * ensemble_index + 2))p" "${manifest}")
IFS=$'\t' read -r _array_index campaign_type target_c run _input purpose \
    physical_nt _stored_l2 ns beta _chain_index _seed sweeps _base _suite \
    _input_hash _binary_hash _commit _uv_hash <<< "${line}"
ensemble=$(basename "$(dirname "${run}")")
data="${remote_root}/data/mg_nt_fixed/${ensemble}"
analysis="${remote_root}/analysis/mg_nt_fixed/pilot/${ensemble}"

mapfile -t manifest_runs < <(
    awk -F '\t' -v first="$((4 * ensemble_index + 2))" \
        'NR >= first && NR < first + 4 {print $4}' "${manifest}"
)
[[ "${#manifest_runs[@]}" -eq 4 ]] || {
    echo "${ensemble}: expected four manifest runs" >&2
    exit 1
}
chain_arguments=()
for manifest_run in "${manifest_runs[@]}"; do
    [[ "$(basename "$(dirname "${manifest_run}")")" == "${ensemble}" ]] || {
        echo "${manifest_run}: manifest ensemble mismatch" >&2
        exit 1
    }
    directory="${remote_root}/data/${manifest_run}"
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
        echo "${directory}: incomplete holonomy history" >&2
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
    --purpose "${purpose}"
    --campaign-type "${campaign_type}"
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
        --bootstrap-seed "$((20260724 + ensemble_index))" \
        --output "${analysis}/analysis.json" \
        --covariance-output "${analysis}/covariance.npz" \
        > "${analysis}/analysis.log"
fi
