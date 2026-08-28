#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 1 || ! "$1" =~ ^(smoke|timing)$ ]]; then
    echo "usage: $0 smoke|timing" >&2
    exit 2
fi
mode="$1"
index=0
[[ "${mode}" == smoke ]] || index=1
remote_root=/qbigwork2/negro/dirichlet
campaign="${remote_root}/campaigns/mg_nt_fixed/extensions/round2"
manifest="${campaign}/validation_manifest.tsv"
source_dir=/hadron/negro/src/klft
binary="${remote_root}/build/klft/binaries/heatbath"

[[ "$(awk -F '\t' '$1 == "gate" {print $2}' "${remote_root}/campaigns/mg_nt_fixed/timing_gate.tsv")" == PASS ]]
[[ "$(awk -F '\t' '$1 == "gate" {print $2}' "${remote_root}/campaigns/mg_nt_fixed/audit_gate.tsv")" == PASS ]]
line=$(sed -n "$((index + 2))p" "${manifest}")
IFS=$'\t' read -r array_index campaign_type target_c run input_relative purpose \
    physical_nt stored_l2 ns beta chain_index seed sweeps base_interval \
    suite_interval expected_input_sha expected_binary_sha expected_commit \
    expected_uv_sha <<< "${line}"
[[ "${array_index}" -eq "${index}" && "${campaign_type}" == nonperturbative_mg_nt ]]
source_input="${campaign}/${input_relative}"
run_dir="${remote_root}/data/${run}"
[[ -f "${source_input}" && -x "${binary}" ]]
[[ "$(git -C "${source_dir}" rev-parse HEAD)" == "${expected_commit}" ]]
[[ "$(sha256sum "${binary}" | awk '{print $1}')" == "${expected_binary_sha}" ]]
[[ "$(sha256sum "${remote_root}/project/uv.lock" | awk '{print $1}')" == "${expected_uv_sha}" ]]
[[ "$(sha256sum "${source_input}" | awk '{print $1}')" == "${expected_input_sha}" ]]
[[ ! -e "${run_dir}" ]] || {
    echo "refusing existing validation path ${run_dir}" >&2
    exit 1
}
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
summary_rows=$(awk '!/^#/ {n++} END {print n+0}' holonomy_summary.out)
correlator_rows=$(awk '!/^#/ {n++} END {print n+0}' holonomy_correlator.out)
profile_rows=$(awk '!/^#/ {n++} END {print n+0}' plaquette_profile.out)
[[ "${summary_rows}" -eq "${sweeps}" ]]
[[ "${correlator_rows}" -eq $((sweeps * (ns / 2 + 1))) ]]
[[ "${profile_rows}" -eq $((sweeps / suite_interval * (2 * physical_nt + 1))) ]]
awk '!/^#/ && $2 == 0 && $3 != 1 {bad++} END {exit bad != 0}' holonomy_correlator.out
for disabled in boundary_wilson.out w_temp.out w_mu_nu.out bulk_horizontal_polyakov.out; do
    [[ ! -e "${disabled}" ]]
done
printf 'complete\t%s\n' "$(date -Is)" > run_complete.txt
if [[ "${mode}" == smoke ]]; then
    printf 'gate\tPASS\n' > "${campaign}/smoke_gate.tsv"
else
    elapsed=$(<elapsed_seconds.txt)
    projected_one=$(awk -v elapsed="${elapsed}" -v sweeps="${sweeps}" 'BEGIN {print elapsed / sweeps * 8000 / 3600}')
    projected_total=$(awk -v one="${projected_one}" 'BEGIN {print 4 * one}')
    awk -v one="${projected_one}" -v total="${projected_total}" 'BEGIN {exit !(one <= 2.0 && total <= 12.0)}'
    printf 'gate\tPASS\nelapsed_seconds\t%s\nprojected_chain_hours\t%s\nprojected_total_gpu_hours\t%s\n' \
        "${elapsed}" "${projected_one}" "${projected_total}" > "${campaign}/timing_gate.tsv"
fi
