#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "usage: $0 diagnose|analyse ENSEMBLE_INDEX" >&2
    exit 2
}

[[ "$#" -eq 2 ]] || usage
mode="$1"
ensemble_index="$2"
[[ "${mode}" == diagnose || "${mode}" == analyse ]] || usage
[[ "${ensemble_index}" =~ ^[0-7]$ ]] || usage

remote_root=/qbigwork2/negro/dirichlet
project="${remote_root}/project"
campaign="${remote_root}/campaigns/step3_scan"
manifest="${campaign}/manifest.tsv"
uv="${remote_root}/bin/uv"
export UV_CACHE_DIR="${remote_root}/.uv-cache"

line=$(sed -n "$((4 * ensemble_index + 2))p" "${manifest}")
IFS=$'\t' read -r _array_index run _input purpose physical_nt _stored_l2 \
    ns beta _lambda _chain_index _seed sweeps _base _suite _hash <<< "${line}"
ensemble=${run%/seed_*}
data="${remote_root}/data/${ensemble}"
analysis="${remote_root}/analysis/step3_scan/${ensemble}"

chain_count=0
for directory in "${data}"/seed_*; do
    [[ -d "${directory}" ]] || continue
    ((chain_count+=1))
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
[[ "${chain_count}" -eq 4 ]] || {
    echo "${ensemble}: expected four complete chains" >&2
    exit 1
}

mkdir -p "${analysis}"
cd "${project}"
common=(
    --ensemble-dir "${data}"
    --nt "${physical_nt}"
    --ns "${ns}"
    --beta "${beta}"
    --purpose "${purpose}"
    --minimum-cut 500
)
if [[ "${mode}" == diagnose ]]; then
    "${uv}" run --frozen python scripts/step3_analysis.py \
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
    "${uv}" run --frozen python scripts/step3_analysis.py \
        "${common[@]}" \
        --thermalization-diagnostics "${diagnostics}" \
        --bootstrap-samples 20000 \
        --bootstrap-seed "$((20260721 + ensemble_index))" \
        --output "${analysis}/analysis.json" \
        --covariance-output "${analysis}/covariance.npz" \
        > "${analysis}/analysis.log"
fi
