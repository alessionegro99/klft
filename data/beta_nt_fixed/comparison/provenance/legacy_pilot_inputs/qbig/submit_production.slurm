#!/bin/bash -l
#SBATCH --job-name=bntf-prod
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:pascal:1
#SBATCH --mem=5G
#SBATCH --time=02:15:00
#SBATCH --array=0-31
#SBATCH --output=/qbigwork2/negro/dirichlet/logs/bntf-prod-%A_%a.out

set -euo pipefail

source /hadron/negro/.env/klft_env.sh
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OMP_PROC_BIND=close
export OMP_PLACES=cores

remote_root=/qbigwork2/negro/dirichlet
project="${remote_root}/project"
campaign="${remote_root}/campaigns/step3_scan"
manifest="${campaign}/manifest.tsv"
source_dir=/hadron/negro/src/klft
expected_commit=$(tr -d '[:space:]' < "${project}/data/beta_nt_fixed/pilot/klft_commit.txt")
build="${remote_root}/build/klft"

[[ "$(awk -F '\t' '$1 == "gate" {print $2}' "${campaign}/timing_gate.tsv")" == PASS ]] || {
    echo "timing gate did not pass" >&2
    exit 1
}
line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "${manifest}")
[[ -n "${line}" ]] || {
    echo "no manifest row for task ${SLURM_ARRAY_TASK_ID}" >&2
    exit 1
}
IFS=$'\t' read -r array_index run input_relative purpose physical_nt stored_l2 \
    ns beta lambda_squared chain_index seed sweeps base_interval suite_interval \
    expected_input_sha <<< "${line}"
[[ "${array_index}" -eq "${SLURM_ARRAY_TASK_ID}" ]] || {
    echo "manifest array index mismatch" >&2
    exit 1
}

source_input="${campaign}/${input_relative}"
run_dir="${remote_root}/data/${run}"
input="${run_dir}/input.yaml"
[[ -f "${source_input}" && -x "${build}/binaries/heatbath" ]] || {
    echo "missing campaign input or binary" >&2
    exit 1
}
[[ "$(git -C "${source_dir}" rev-parse HEAD)" == "${expected_commit}" ]] || {
    echo "KLFT commit changed after build" >&2
    exit 1
}
[[ "$(sha256sum "${source_input}" | awk '{print $1}')" == \
   "${expected_input_sha}" ]] || {
    echo "campaign input hash mismatch" >&2
    exit 1
}
mkdir -p "${run_dir}"
if [[ -e "${input}" ]]; then
    cmp --silent "${source_input}" "${input}" || {
        echo "staged input differs from immutable campaign input" >&2
        exit 1
    }
else
    cp "${source_input}" "${input}"
fi
if compgen -G "${run_dir}/*.out" >/dev/null ||
   compgen -G "${run_dir}/*.klft" >/dev/null ||
   [[ -e "${run_dir}/run_metadata.txt" ]]; then
    echo "refusing to append to existing output in ${run_dir}" >&2
    exit 1
fi

input_l2=$(awk '$1 == "L2:" {print $2}' "${input}")
input_beta=$(awk '$1 == "beta:" {print $2}' "${input}")
input_base=$(awk '$1 == "measurement_interval:" {print $2}' "${input}")
input_suite=$(awk '$1 == "dirichlet_suite_measurement_interval:" {print $2}' "${input}")
[[ "${input_l2}" == "${stored_l2}" && "${input_beta}" == "${beta}" &&
   "${input_base}" == "${base_interval}" &&
   "${input_suite}" == "${suite_interval}" ]] || {
    echo "manifest/input parameter mismatch" >&2
    exit 1
}
grep -q '^  temporal_dirichlet: true$' "${input}"
grep -q '^  measure_dirichlet_plaquette_profiles: true$' "${input}"
grep -q '^  measure_boundary_wilson_loops: false$' "${input}"
grep -q '^  measure_wilson_loop_temporal: false$' "${input}"
grep -q '^  measure_wilson_loop_mu_nu: false$' "${input}"

{
    printf 'campaign beta_nt_fixed_pilot\n'
    printf 'purpose %s\n' "${purpose}"
    printf 'job_id %s_%s\n' "${SLURM_ARRAY_JOB_ID}" "${SLURM_ARRAY_TASK_ID}"
    printf 'node %s\ncuda_visible_devices %s\n' \
        "${SLURMD_NODENAME}" "${CUDA_VISIBLE_DEVICES:-unset}"
    printf 'physical_Nt %s\nstored_L2 %s\nNs %s\nbeta %s\n' \
        "${physical_nt}" "${stored_l2}" "${ns}" "${beta}"
    printf 'lambda_squared_tree %s\nchain_index %s\nseed %s\n' \
        "${lambda_squared}" "${chain_index}" "${seed}"
    printf 'sweeps %s\nmeasurement_interval %s\n' "${sweeps}" "${base_interval}"
    printf 'dirichlet_suite_measurement_interval %s\n' "${suite_interval}"
    printf 'klft_commit %s\nbuild_path %s\n' "${expected_commit}" "${build}"
    printf 'heatbath_sha256 '
    sha256sum "${build}/binaries/heatbath" | awk '{print $1}'
    printf 'input_sha256 %s\n' "${expected_input_sha}"
} > "${run_dir}/run_metadata.txt"

cd "${run_dir}"
/usr/bin/time -v "${build}/binaries/heatbath" -f input.yaml

summary_rows=$(awk '!/^#/ {count++} END {print count+0}' holonomy_summary.out)
correlator_rows=$(awk '!/^#/ {count++} END {print count+0}' holonomy_correlator.out)
profile_rows=$(awk '!/^#/ {count++} END {print count+0}' plaquette_profile.out)
expected_correlator_rows=$((sweeps * (ns / 2 + 1)))
expected_profile_rows=$((sweeps / suite_interval * (2 * physical_nt + 1)))
[[ "${summary_rows}" -eq "${sweeps}" &&
   "${correlator_rows}" -eq "${expected_correlator_rows}" &&
   "${profile_rows}" -eq "${expected_profile_rows}" ]] || {
    echo "production output counts are incorrect" >&2
    exit 1
}
awk '!/^#/ && $2 == 0 && $3 != 1 {bad++} END {exit bad != 0}' \
    holonomy_correlator.out
awk -v last="${sweeps}" -v cadence="${suite_interval}" \
    -v rows="$((2 * physical_nt + 1))" \
    '!/^#/ {seen[$1]++; if ($1 % cadence != 0) bad++}
     END {for (step=cadence; step<=last; step+=cadence)
              if (seen[step] != rows) bad++;
          exit bad != 0}' plaquette_profile.out
for disabled in boundary_wilson.out w_temp.out w_mu_nu.out \
                bulk_horizontal_polyakov.out; do
    [[ ! -e "${disabled}" ]] || {
        echo "disabled observable unexpectedly wrote ${disabled}" >&2
        exit 1
    }
done
