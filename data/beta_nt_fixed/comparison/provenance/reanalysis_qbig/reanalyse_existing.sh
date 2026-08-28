#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 1 || ! "$1" =~ ^[0-4]$ ]]; then
    echo "usage: $0 INDEX" >&2
    exit 2
fi

index="$1"
remote_root=/qbigwork2/negro/dirichlet
project="${remote_root}/project"
uv="${remote_root}/bin/uv"
export UV_CACHE_DIR="${remote_root}/.uv-cache"

case "${index}" in
    0)
        ensemble=Nt_2_Ns_96_b_3p6
        nt=2
        ns=96
        beta=3.6
        source_analysis=step3_scan
        ;;
    1)
        ensemble=Nt_2_Ns_128_b_3p6
        nt=2
        ns=128
        beta=3.6
        source_analysis=step3_scan
        ;;
    2)
        ensemble=Nt_2_Ns_160_b_3p6
        nt=2
        ns=160
        beta=3.6
        source_analysis=step3_fv160
        ;;
    3)
        ensemble=Nt_3_Ns_176_b_5p4
        nt=3
        ns=176
        beta=5.4
        source_analysis=step3_corrected_volumes
        ;;
    4)
        ensemble=Nt_4_Ns_256_b_7p2
        nt=4
        ns=256
        beta=7.2
        source_analysis=step3_corrected_volumes
        ;;
esac

data="${remote_root}/data/${ensemble}"
diagnostics="${remote_root}/analysis/${source_analysis}/${ensemble}/thermalization_diagnostics.json"
output="${remote_root}/analysis/beta_nt_fixed/reanalysis_v2/${ensemble}"
[[ -f "${diagnostics}" ]] || {
    echo "missing ${diagnostics}" >&2
    exit 1
}

cd "${project}"
mapfile -t chains < <(
    "${uv}" run --frozen python -c \
        'import json,sys; print("\n".join(x["chain"] for x in json.load(open(sys.argv[1]))["chains"]))' \
        "${diagnostics}"
)
[[ "${#chains[@]}" -eq 4 ]] || {
    echo "${ensemble}: expected four recorded chains" >&2
    exit 1
}
chain_arguments=()
for chain in "${chains[@]}"; do
    [[ -d "${chain}" ]] || {
        echo "missing ${chain}" >&2
        exit 1
    }
    chain_arguments+=(--chain-directory "${chain}")
done

mkdir -p "${output}"
[[ ! -e "${output}/analysis.json" && ! -e "${output}/covariance.npz" ]] || {
    echo "refusing to overwrite ${output}" >&2
    exit 1
}
"${uv}" run --frozen python scripts/beta_nt_fixed/analysis.py \
    --ensemble-dir "${data}" \
    "${chain_arguments[@]}" \
    --nt "${nt}" \
    --ns "${ns}" \
    --beta "${beta}" \
    --purpose "fixed-beta/Nt fit-method reanalysis" \
    --minimum-cut 500 \
    --thermalization-diagnostics "${diagnostics}" \
    --bootstrap-samples 1000 \
    --bootstrap-seed "$((20260730 + index))" \
    --output "${output}/analysis.json" \
    --covariance-output "${output}/covariance.npz" \
    > "${output}/analysis.log"
