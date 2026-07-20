#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "usage: $0 diagnose|analyse BETA_TAG" >&2
    exit 2
}

[[ "$#" -eq 2 ]] || usage
mode="$1"
beta_tag="$2"
[[ "${mode}" == diagnose || "${mode}" == analyse ]] || usage
[[ "${beta_tag}" == 1p9 || "${beta_tag}" == 2p0 || "${beta_tag}" == 2p1 ]] || usage

remote_root=/qbigwork2/negro/dirichlet
project="${remote_root}/project"
ensemble="Nt_1_Ns_96_b_${beta_tag}"
data="${remote_root}/data/${ensemble}"
analysis="${remote_root}/analysis/pcm_beta_pilot/${ensemble}"
uv="${remote_root}/bin/uv"
export UV_CACHE_DIR="${remote_root}/.uv-cache"

seeds=(32091 90517 177013 515029)
correlators=()
summaries=()
for seed in "${seeds[@]}"; do
    correlator="${data}/seed_${seed}/holonomy_correlator.out"
    summary="${data}/seed_${seed}/holonomy_summary.out"
    [[ -f "${correlator}" && -f "${summary}" ]] || {
        echo "missing observable files below ${data}/seed_${seed}" >&2
        exit 1
    }
    measurements=$(awk '!/^#/ {count++} END {print count+0}' "${summary}")
    if [[ "${measurements}" -ne 10000 ]]; then
        echo "${summary}: expected 10000 measurements, found ${measurements}" >&2
        exit 1
    fi
    correlators+=("${correlator}")
    summaries+=("${summary}")
done

mkdir -p "${analysis}"
cd "${project}"
if [[ "${mode}" == diagnose ]]; then
    "${uv}" run --frozen python scripts/thermalization_diagnostics.py \
        --summary "${summaries[@]}" \
        --correlator "${correlators[@]}" \
        --minimum-cut 500 \
        --output "${analysis}/thermalization_diagnostics.json"
else
    diagnostics="${analysis}/thermalization_diagnostics.json"
    [[ -f "${diagnostics}" ]] || {
        echo "run the thermalization diagnostics first: ${diagnostics}" >&2
        exit 1
    }
    "${uv}" run --frozen python scripts/pcm_beta_pilot_analysis.py \
        --correlator "${correlators[@]}" \
        --summary "${summaries[@]}" \
        --thermalization-diagnostics "${diagnostics}" \
        --ns 96 \
        --bootstrap-samples 20000 \
        --bootstrap-seed 20260720 \
        --output "${analysis}/analysis.json" \
        --covariance-output "${analysis}/covariance.npz" \
        > "${analysis}/analysis.log"
fi
