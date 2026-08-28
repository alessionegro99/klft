#!/usr/bin/env bash
set -euo pipefail
if [[ "$#" -ne 1 || ("$1" != diagnose && "$1" != analyse) ]]; then
    echo "usage: $0 diagnose|analyse" >&2
    exit 2
fi
mode="$1"
remote_root=/qbigwork2/negro/dirichlet
project="${remote_root}/project"
campaign="${remote_root}/campaigns/mg_nt_fixed"
manifest="${campaign}/confirmation_manifest.tsv"
uv="${remote_root}/bin/uv"
export UV_CACHE_DIR="${remote_root}/.uv-cache"
line=$(sed -n '2p' "${manifest}")
IFS=$'\t' read -r _index campaign_type target_c run _input purpose physical_nt \
    _stored_l2 ns beta _chain _seed sweeps _base _suite _ih _bh _commit _uvh <<< "${line}"
ensemble=$(basename "$(dirname "${run}")")
data="${remote_root}/data/mg_nt_fixed/confirmation/${ensemble}"
analysis="${remote_root}/analysis/mg_nt_fixed/confirmation/${ensemble}"
mapfile -t runs < <(awk -F '\t' 'NR > 1 {print $4}' "${manifest}")
[[ "${#runs[@]}" -eq 4 ]] || exit 1
chain_arguments=()
for manifest_run in "${runs[@]}"; do
    directory="${remote_root}/data/${manifest_run}"
    chain_arguments+=(--chain-directory "${directory}")
    [[ "$(awk '!/^#/ {n++} END {print n+0}' "${directory}/holonomy_summary.out")" -eq "${sweeps}" ]] || exit 1
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
        "${common[@]}" --diagnostics-only \
        --output "${analysis}/thermalization_diagnostics.json" \
        > "${analysis}/thermalization_diagnostics.log"
else
    "${uv}" run --frozen python scripts/mg_nt_fixed_analysis.py \
        "${common[@]}" \
        --thermalization-diagnostics "${analysis}/thermalization_diagnostics.json" \
        --bootstrap-samples 1000 --bootstrap-seed 20260924 \
        --output "${analysis}/analysis.json" \
        --covariance-output "${analysis}/covariance.npz" \
        > "${analysis}/analysis.log"
fi
