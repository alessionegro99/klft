#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "usage: $0 diagnose NT | analyse NT THERMALIZATION_CUT" >&2
    exit 2
}

[[ "$#" -ge 2 && "$#" -le 3 ]] || usage
mode="$1"
nt="$2"
[[ "${nt}" =~ ^[0-9]+$ ]] || usage

remote_root=/qbigwork2/negro/dirichlet
project="${remote_root}/project"
ensemble="Nt_${nt}_Ns_64_b_12p15266"
data="${remote_root}/data/${ensemble}"
analysis="${remote_root}/analysis/${ensemble}"
uv="${remote_root}/bin/uv"
export UV_CACHE_DIR="${remote_root}/.uv-cache"

shopt -s nullglob
correlators=("${data}"/seed_*/holonomy_correlator.out)
summaries=("${data}"/seed_*/holonomy_summary.out)
profiles=("${data}"/seed_*/plaquette_profile.out)
wilson=("${data}"/seed_*/boundary_wilson.out)
if [[ "${#correlators[@]}" -ne 4 || "${#summaries[@]}" -ne 4 ||
      "${#profiles[@]}" -ne 4 || "${#wilson[@]}" -ne 4 ]]; then
    echo "${ensemble}: expected four complete observable sets" >&2
    exit 1
fi
for summary in "${summaries[@]}"; do
    measurements=$(awk '!/^#/ {count++} END {print count+0}' "${summary}")
    if [[ "${measurements}" -ne 8000 ]]; then
        echo "${summary}: expected 8000 measurements, found ${measurements}" >&2
        exit 1
    fi
done

mkdir -p "${analysis}"
cd "${project}"
if [[ "${mode}" == diagnose && "$#" -eq 2 ]]; then
    "${uv}" run --frozen python scripts/thermalization_diagnostics.py \
        --summary "${summaries[@]}" \
        --correlator "${correlators[@]}" \
        --minimum-cut 500 \
        --output "${analysis}/thermalization_diagnostics.json"
elif [[ "${mode}" == analyse && "$#" -eq 3 ]]; then
    cut="$3"
    [[ "${cut}" =~ ^[0-9]+$ ]] || usage
    output="${analysis}/cut_${cut}"
    mkdir -p "${output}"
    "${uv}" run --frozen python scripts/dirichlet_analysis.py \
        --correlator "${correlators[@]}" \
        --summary "${summaries[@]}" \
        --profile "${profiles[@]}" \
        --wilson "${wilson[@]}" \
        --ns 64 \
        --thermalization "${cut}" \
        --fit-rmin 2 3 4 5 6 \
        --fit-rmax 12 16 20 24 28 32 \
        --output "${output}/analysis.json" \
        --covariance-output "${output}/covariance.npz" \
        > "${output}/analysis.log"
else
    usage
fi
