#!/bin/bash -l
#SBATCH --job-name=mgnt-ext1-prod
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:pascal:1
#SBATCH --mem=5G
#SBATCH --time=02:00:00
#SBATCH --array=0-7
#SBATCH --output=/qbigwork2/negro/dirichlet/logs/mgnt-ext1-prod-%A_%a.out

set -euo pipefail
source /hadron/negro/.env/klft_env.sh
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OMP_PROC_BIND=close
export OMP_PLACES=cores

remote_root=/qbigwork2/negro/dirichlet
campaign="${remote_root}/campaigns/mg_nt_fixed/extensions/round1"
manifest="${campaign}/manifest.tsv"
source_dir=/hadron/negro/src/klft
build="${remote_root}/build/klft"
binary="${build}/binaries/heatbath"

[[ "$(awk -F '\t' '$1 == "gate" {print $2}' "${remote_root}/campaigns/mg_nt_fixed/timing_gate.tsv")" == PASS ]]
[[ "$(awk -F '\t' '$1 == "gate" {print $2}' "${remote_root}/campaigns/mg_nt_fixed/audit_gate.tsv")" == PASS ]]
line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "${manifest}")
IFS=$'\t' read -r array_index campaign_type target_c run input_relative purpose \
    physical_nt stored_l2 ns beta chain_index seed sweeps base_interval \
    suite_interval expected_input_sha expected_binary_sha expected_commit \
    expected_uv_sha <<< "${line}"
[[ "${array_index}" -eq "${SLURM_ARRAY_TASK_ID}" ]]

source_input="${campaign}/${input_relative}"
run_dir="${remote_root}/data/${run}"
input="${run_dir}/input.yaml"
[[ -f "${source_input}" && -x "${binary}" ]]
[[ "$(git -C "${source_dir}" rev-parse HEAD)" == "${expected_commit}" ]]
[[ "$(sha256sum "${binary}" | awk '{print $1}')" == "${expected_binary_sha}" ]]
[[ "$(sha256sum "${remote_root}/project/uv.lock" | awk '{print $1}')" == "${expected_uv_sha}" ]]
[[ "$(sha256sum "${source_input}" | awk '{print $1}')" == "${expected_input_sha}" ]]

mkdir -p "${run_dir}"
if [[ -e "${input}" ]]; then
    cmp --silent "${source_input}" "${input}"
else
    cp "${source_input}" "${input}"
fi
if compgen -G "${run_dir}/*.out" >/dev/null ||
   compgen -G "${run_dir}/*.klft" >/dev/null ||
   [[ -e "${run_dir}/run_metadata.txt" ]]; then
    echo "refusing to append to existing output in ${run_dir}" >&2
    exit 1
fi

[[ "${campaign_type}" == nonperturbative_mg_nt ]]
[[ "$(awk '$1 == "L2:" {print $2}' "${input}")" == "${stored_l2}" ]]
[[ "$(awk '$1 == "beta:" {print $2}' "${input}")" == "${beta}" ]]
[[ "$(awk '$1 == "measurement_interval:" {print $2}' "${input}")" == "${base_interval}" ]]
[[ "$(awk '$1 == "dirichlet_suite_measurement_interval:" {print $2}' "${input}")" == "${suite_interval}" ]]
grep -q '^  temporal_dirichlet: true$' "${input}"
grep -q '^  measure_dirichlet_plaquette_profiles: true$' "${input}"
grep -q '^  measure_boundary_wilson_loops: false$' "${input}"

{
    printf 'campaign_type %s\ntarget_c %s\npurpose %s\n' \
        "${campaign_type}" "${target_c}" "${purpose}"
    printf 'job_id %s_%s\nnode %s\ncuda_visible_devices %s\n' \
        "${SLURM_ARRAY_JOB_ID}" "${SLURM_ARRAY_TASK_ID}" \
        "${SLURMD_NODENAME}" "${CUDA_VISIBLE_DEVICES:-unset}"
    printf 'physical_Nt %s\nstored_L2 %s\nNs %s\nbeta %s\n' \
        "${physical_nt}" "${stored_l2}" "${ns}" "${beta}"
    printf 'chain_index %s\nseed %s\nsweeps %s\n' \
        "${chain_index}" "${seed}" "${sweeps}"
    printf 'measurement_interval %s\n' "${base_interval}"
    printf 'dirichlet_suite_measurement_interval %s\n' "${suite_interval}"
    printf 'klft_commit %s\nbuild_path %s\n' "${expected_commit}" "${build}"
    printf 'heatbath_sha256 %s\ninput_sha256 %s\nuv_lock_sha256 %s\n' \
        "${expected_binary_sha}" "${expected_input_sha}" "${expected_uv_sha}"
} > "${run_dir}/run_metadata.txt"

cd "${run_dir}"
/usr/bin/time -v "${binary}" -f input.yaml

summary_rows=$(awk '!/^#/ {count++} END {print count+0}' holonomy_summary.out)
correlator_rows=$(awk '!/^#/ {count++} END {print count+0}' holonomy_correlator.out)
profile_rows=$(awk '!/^#/ {count++} END {print count+0}' plaquette_profile.out)
[[ "${summary_rows}" -eq "${sweeps}" ]]
[[ "${correlator_rows}" -eq $((sweeps * (ns / 2 + 1))) ]]
[[ "${profile_rows}" -eq $((sweeps / suite_interval * (2 * physical_nt + 1))) ]]
awk '!/^#/ && $2 == 0 && $3 != 1 {bad++} END {exit bad != 0}' holonomy_correlator.out
for disabled in boundary_wilson.out w_temp.out w_mu_nu.out bulk_horizontal_polyakov.out; do
    [[ ! -e "${disabled}" ]]
done
printf 'complete\t%s\n' "$(date -Is)" > run_complete.txt
