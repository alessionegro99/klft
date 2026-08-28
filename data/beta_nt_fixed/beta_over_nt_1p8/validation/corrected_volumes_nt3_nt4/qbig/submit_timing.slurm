#!/bin/bash -l
#SBATCH --job-name=bntf-cv-timing
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:pascal:1
#SBATCH --mem=6G
#SBATCH --time=00:20:00
#SBATCH --output=/qbigwork2/negro/dirichlet/logs/bntf-cv-timing-%j.out

set -euo pipefail

source /hadron/negro/.env/klft_env.sh
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OMP_PROC_BIND=close
export OMP_PLACES=cores

remote_root=/qbigwork2/negro/dirichlet
campaign="${remote_root}/campaigns/step3_corrected_volumes"
build="${remote_root}/build/klft"
binary="${build}/binaries/heatbath"
source_dir=/hadron/negro/src/klft
expected_commit=$(tr -d '[:space:]' < "${campaign}/klft_commit.txt")
expected_binary_sha=$(tr -d '[:space:]' < "${campaign}/heatbath_sha256.txt")
source_input="${campaign}/timing/input.yaml"
expected_input_sha=$(awk '{print $1}' "${campaign}/timing/sha256.txt")
run_dir="${remote_root}/data/step3_corrected_timing_Nt_4_Ns_256_b_7p2"
input="${run_dir}/input.yaml"
gate="${campaign}/timing_gate.tsv"

[[ -x "${binary}" && -f "${source_input}" && ! -e "${gate}" ]] || {
    echo "missing verified binary/input or existing timing gate" >&2
    exit 1
}
[[ "$(git -C "${source_dir}" rev-parse HEAD)" == "${expected_commit}" ]]
[[ "$(sha256sum "${binary}" | awk '{print $1}')" == \
   "${expected_binary_sha}" ]]
[[ "$(sha256sum "${source_input}" | awk '{print $1}')" == \
   "${expected_input_sha}" ]]
mkdir -p "${run_dir}"
if [[ -e "${input}" ]]; then
    cmp --silent "${source_input}" "${input}" || {
        echo "staged timing input differs" >&2
        exit 1
    }
else
    cp "${source_input}" "${input}"
fi
if compgen -G "${run_dir}/*.out" >/dev/null ||
   compgen -G "${run_dir}/*.klft" >/dev/null ||
   [[ -e "${run_dir}/run_metadata.txt" ]]; then
    echo "refusing to append to existing timing output" >&2
    exit 1
fi

[[ "$(awk '$1 == "L0:" {print $2}' "${input}")" == 256 ]]
[[ "$(awk '$1 == "L1:" {print $2}' "${input}")" == 256 ]]
[[ "$(awk '$1 == "L2:" {print $2}' "${input}")" == 5 ]]
[[ "$(awk '$1 == "nSweep:" {print $2}' "${input}")" == 100 ]]
[[ "$(awk '$1 == "beta:" {print $2}' "${input}")" == 7.2 ]]
grep -q '^  temporal_dirichlet: true$' "${input}"
grep -q '^  measurement_interval: 1$' "${input}"
grep -q '^  dirichlet_suite_measurement_interval: 5$' "${input}"
grep -q '^  measure_dirichlet_plaquette_profiles: true$' "${input}"
grep -q '^  measure_boundary_wilson_loops: false$' "${input}"

{
    printf 'campaign beta_nt_fixed_corrected_volumes_timing\n'
    printf 'job_id %s\nnode %s\n' "${SLURM_JOB_ID}" "${SLURMD_NODENAME}"
    printf 'physical_Nt 4\nstored_L2 5\nNs 256\nbeta 7.2\n'
    printf 'seed 1747406840\nsweeps 100\nmeasurement_interval 1\n'
    printf 'dirichlet_suite_measurement_interval 5\n'
    printf 'klft_commit %s\nbuild_path %s\n' "${expected_commit}" "${build}"
    printf 'heatbath_sha256 %s\ninput_sha256 %s\n' \
        "${expected_binary_sha}" "${expected_input_sha}"
} > "${run_dir}/run_metadata.txt"

nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
cd "${run_dir}"
start_ns=$(date +%s%N)
/usr/bin/time -v "${binary}" -f input.yaml
end_ns=$(date +%s%N)
wall_seconds=$(awk -v start="${start_ns}" -v end="${end_ns}" \
    'BEGIN {printf "%.6f", (end-start)/1.0e9}')

summary_rows=$(awk '!/^#/ {count++} END {print count+0}' holonomy_summary.out)
correlator_rows=$(awk '!/^#/ {count++} END {print count+0}' holonomy_correlator.out)
profile_rows=$(awk '!/^#/ {count++} END {print count+0}' plaquette_profile.out)
[[ "${summary_rows}" -eq 100 && "${correlator_rows}" -eq 12900 &&
   "${profile_rows}" -eq 180 ]] || {
    echo "timing output counts are incorrect" >&2
    exit 1
}
awk '!/^#/ && $2 == 0 && $3 != 1 {bad++} END {exit bad != 0}' \
    holonomy_correlator.out
awk '!/^#/ {seen[$1]++; if ($1 % 5 != 0) bad++}
     END {for (step=5; step<=100; step+=5) if (seen[step] != 9) bad++;
          exit bad != 0}' plaquette_profile.out
for disabled in boundary_wilson.out w_temp.out w_mu_nu.out \
                bulk_horizontal_polyakov.out; do
    [[ ! -e "${disabled}" ]] || {
        echo "disabled observable unexpectedly wrote ${disabled}" >&2
        exit 1
    }
done

max_chain_hours=$(awk -v seconds="${wall_seconds}" \
    'BEGIN {printf "%.8f", seconds*200.0/3600.0}')
total_gpu_hours=$(awk -v maximum="${max_chain_hours}" \
    'BEGIN {printf "%.8f", maximum*8.0}')
status=PASS
if ! awk -v maximum="${max_chain_hours}" -v total="${total_gpu_hours}" \
    'BEGIN {exit !(maximum <= 2.0 && total <= 12.0)}'; then
    status=FAIL
fi
{
    printf 'gate\t%s\n' "${status}"
    printf 'wall_seconds_100_sweeps\t%s\n' "${wall_seconds}"
    printf 'projected_max_chain_hours\t%s\n' "${max_chain_hours}"
    printf 'projected_total_gpu_hours_upper_bound\t%s\n' "${total_gpu_hours}"
    printf 'max_chain_limit_hours\t2\n'
    printf 'total_limit_gpu_hours\t12\n'
} > "${gate}"
[[ "${status}" == PASS ]] || {
    echo "corrected-volume timing exceeds a stop threshold" >&2
    exit 1
}
