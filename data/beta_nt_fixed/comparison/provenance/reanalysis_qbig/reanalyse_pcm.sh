#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 1 || ! "$1" =~ ^[0-1]$ ]]; then
    echo "usage: $0 INDEX" >&2
    exit 2
fi

index="$1"
case "${index}" in
    0) beta_tag=2p0 ;;
    1) beta_tag=2p1 ;;
esac

remote_root=/qbigwork2/negro/dirichlet
project="${remote_root}/project"
uv="${remote_root}/bin/uv"
export UV_CACHE_DIR="${remote_root}/.uv-cache"
ensemble="Nt_1_Ns_96_b_${beta_tag}"
diagnostics="${remote_root}/analysis/pcm_beta_pilot/${ensemble}/thermalization_diagnostics.json"
output="${remote_root}/analysis/beta_nt_fixed/pcm_reanalysis_v2/${ensemble}"
[[ -f "${diagnostics}" ]]

cd "${project}"
mapfile -t chains < <(
    "${uv}" run --frozen python -c \
        'import json,sys; print("\n".join(x["chain"] for x in json.load(open(sys.argv[1]))["chains"]))' \
        "${diagnostics}"
)
[[ "${#chains[@]}" -eq 4 ]]
correlators=()
summaries=()
for chain in "${chains[@]}"; do
    [[ -f "${chain}/holonomy_correlator.out" ]]
    [[ -f "${chain}/holonomy_summary.out" ]]
    correlators+=("${chain}/holonomy_correlator.out")
    summaries+=("${chain}/holonomy_summary.out")
done

mkdir -p "${output}"
[[ ! -e "${output}/analysis.json" && ! -e "${output}/covariance.npz" ]]
"${uv}" run --frozen python scripts/pcm_beta_pilot_analysis.py \
    --correlator "${correlators[@]}" \
    --summary "${summaries[@]}" \
    --thermalization-diagnostics "${diagnostics}" \
    --ns 96 \
    --bootstrap-samples 1000 \
    --bootstrap-seed "$((20260800 + index))" \
    --fit-k0 \
    --output "${output}/analysis.json" \
    --covariance-output "${output}/covariance.npz" \
    > "${output}/analysis.log"
