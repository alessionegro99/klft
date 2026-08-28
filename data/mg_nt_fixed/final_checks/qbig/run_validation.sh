#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 1 || ! "$1" =~ ^(smoke|timing)$ ]]; then
    echo "usage: $0 smoke|timing" >&2
    exit 2
fi
mode="$1"
index=0
[[ "${mode}" == smoke ]] || index=1
root=/qbigwork2/negro/dirichlet
campaign="${root}/campaigns/mg_nt_fixed/final_checks"
manifest="${campaign}/validation_manifest.tsv"
source_dir=/hadron/negro/src/klft
binary="${root}/build/klft/binaries/heatbath"
[[ "$(awk -F '\t' '$1 == "gate" {print $2}' "${root}/campaigns/mg_nt_fixed/timing_gate.tsv")" == PASS ]]
[[ "$(awk -F '\t' '$1 == "gate" {print $2}' "${root}/campaigns/mg_nt_fixed/audit_gate.tsv")" == PASS ]]
line=$(sed -n "$((index + 2))p" "${manifest}")
IFS=$'\t' read -r array_index campaign_type target_c run input_relative purpose \
    physical_nt stored_l2 ns beta chain_index seed sweeps base_interval \
    suite_interval expected_input_sha expected_binary_sha expected_commit \
    expected_uv_sha <<< "${line}"
[[ "${array_index}" -eq "${index}" && "${campaign_type}" == nonperturbative_mg_nt ]]
source_input="${campaign}/${input_relative}"
run_dir="${root}/data/${run}"
[[ -f "${source_input}" && -x "${binary}" && ! -e "${run_dir}" ]]
[[ "$(git -C "${source_dir}" rev-parse HEAD)" == "${expected_commit}" ]]
[[ "$(sha256sum "${binary}" | awk '{print $1}')" == "${expected_binary_sha}" ]]
[[ "$(sha256sum "${root}/project/uv.lock" | awk '{print $1}')" == "${expected_uv_sha}" ]]
[[ "$(sha256sum "${source_input}" | awk '{print $1}')" == "${expected_input_sha}" ]]
mkdir -p "${run_dir}"
cp "${source_input}" "${run_dir}/input.yaml"
{
    printf 'campaign_type %s\ntarget_c %s\npurpose %s\n' "${campaign_type}" "${target_c}" "${purpose}"
    printf 'job_id %s\nnode %s\nseed %s\n' "${SLURM_JOB_ID}" "${SLURMD_NODENAME}" "${seed}"
    printf 'physical_Nt %s\nstored_L2 %s\nNs %s\nbeta %s\nsweeps %s\n' \
        "${physical_nt}" "${stored_l2}" "${ns}" "${beta}" "${sweeps}"
    printf 'klft_commit %s\nheatbath_sha256 %s\ninput_sha256 %s\n' \
        "${expected_commit}" "${expected_binary_sha}" "${expected_input_sha}"
} > "${run_dir}/run_metadata.txt"
cd "${run_dir}"
/usr/bin/time -f '%e' -o elapsed_seconds.txt "${binary}" -f input.yaml
[[ "$(awk '!/^#/ {n++} END {print n+0}' holonomy_summary.out)" -eq "${sweeps}" ]]
[[ "$(awk '!/^#/ {n++} END {print n+0}' holonomy_correlator.out)" -eq $((sweeps * (ns / 2 + 1))) ]]
[[ "$(awk '!/^#/ {n++} END {print n+0}' plaquette_profile.out)" -eq $((sweeps / suite_interval * (2 * physical_nt + 1))) ]]
awk '!/^#/ && $2 == 0 && $3 != 1 {bad++} END {exit bad != 0}' holonomy_correlator.out
for disabled in boundary_wilson.out w_temp.out w_mu_nu.out bulk_horizontal_polyakov.out; do
    [[ ! -e "${disabled}" ]]
done
printf 'complete\t%s\n' "$(date -Is)" > run_complete.txt
if [[ "${mode}" == smoke ]]; then
    printf 'gate\tPASS\n' > "${campaign}/smoke_gate.tsv"
else
    elapsed=$(<elapsed_seconds.txt)
    projected_one=$(awk -v elapsed="${elapsed}" -v sweeps="${sweeps}" \
        'BEGIN {print elapsed / sweeps * 80000 / 3600}')
    projected_total=$(awk -F '\t' -v elapsed="${elapsed}" -v sweeps="${sweeps}" \
        'NR > 1 {total += elapsed / sweeps * $13 / 3600} END {print total}' \
        "${campaign}/manifest.tsv")
    awk -v one="${projected_one}" -v total="${projected_total}" \
        'BEGIN {exit !(one <= 2.0 && total <= 24.0)}'
    printf 'gate\tPASS\nelapsed_seconds\t%s\nprojected_longest_chain_hours\t%s\nprojected_total_gpu_hours\t%s\n' \
        "${elapsed}" "${projected_one}" "${projected_total}" > "${campaign}/timing_gate.tsv"
fi
