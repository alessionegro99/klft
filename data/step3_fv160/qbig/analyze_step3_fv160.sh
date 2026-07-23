#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 1 || ("$1" != diagnose && "$1" != analyse) ]]; then
    echo "usage: $0 diagnose|analyse" >&2
    exit 2
fi
mode="$1"

remote_root=/qbigwork2/negro/dirichlet
project="${remote_root}/project"
campaign="${remote_root}/campaigns/step3_fv160"
manifest="${campaign}/manifest.tsv"
uv="${remote_root}/bin/uv"
export UV_CACHE_DIR="${remote_root}/.uv-cache"

line=$(sed -n '2p' "${manifest}")
IFS=$'\t' read -r _array_index run _input purpose physical_nt _stored_l2 \
    ns beta _lambda _chain_index _seed sweeps _base _suite _hash <<< "${line}"
ensemble=${run%/seed_*}
data="${remote_root}/data/${ensemble}"
analysis="${remote_root}/analysis/step3_fv160/${ensemble}"

mapfile -t manifest_runs < <(awk -F '\t' 'NR >= 2 && NR <= 5 {print $2}' "${manifest}")
[[ "${#manifest_runs[@]}" -eq 4 ]] || {
    echo "${ensemble}: expected four manifest runs" >&2
    exit 1
}
chain_arguments=()
for manifest_run in "${manifest_runs[@]}"; do
    [[ "${manifest_run%/seed_*}" == "${ensemble}" ]] || {
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
    --minimum-cut 500
)
if [[ "${mode}" == diagnose ]]; then
    output="${analysis}/thermalization_diagnostics.json"
    log="${analysis}/thermalization_diagnostics.log"
    [[ ! -e "${output}" && ! -e "${log}" ]] || {
        echo "refusing to overwrite existing diagnostics" >&2
        exit 1
    }
    "${uv}" run --frozen python scripts/step3_analysis.py \
        "${common[@]}" \
        --diagnostics-only \
        --output "${output}" \
        > "${log}"
else
    diagnostics="${analysis}/thermalization_diagnostics.json"
    output="${analysis}/analysis.json"
    covariance="${analysis}/covariance.npz"
    log="${analysis}/analysis.log"
    [[ -f "${diagnostics}" ]] || {
        echo "missing inspected diagnostics ${diagnostics}" >&2
        exit 1
    }
    [[ ! -e "${output}" && ! -e "${covariance}" && ! -e "${log}" ]] || {
        echo "refusing to overwrite existing analysis products" >&2
        exit 1
    }
    "${uv}" run --frozen python scripts/step3_analysis.py \
        "${common[@]}" \
        --thermalization-diagnostics "${diagnostics}" \
        --bootstrap-samples 20000 \
        --bootstrap-seed 20260723 \
        --output "${output}" \
        --covariance-output "${covariance}" \
        > "${log}"
fi
