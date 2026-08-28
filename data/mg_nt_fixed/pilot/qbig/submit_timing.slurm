#!/bin/bash -l
#SBATCH --job-name=mgnt-timing
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:pascal:1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=/qbigwork2/negro/dirichlet/logs/mgnt-timing-%j.out

set -euo pipefail

source /hadron/negro/.env/klft_env.sh
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OMP_PROC_BIND=close
export OMP_PLACES=cores

remote_root=/qbigwork2/negro/dirichlet
project="${remote_root}/project"
campaign="${remote_root}/campaigns/mg_nt_fixed"
source_dir=/hadron/negro/src/klft
build="${remote_root}/build/klft"
binary="${build}/binaries/heatbath"
expected_commit=$(tr -d '[:space:]' < "${campaign}/klft_commit.txt")
expected_binary_sha=$(tr -d '[:space:]' < "${campaign}/heatbath_sha256.txt")
expected_uv_sha=$(tr -d '[:space:]' < "${campaign}/uv_lock_sha256.txt")
source_input="${campaign}/timing/input.yaml"
run_dir="${remote_root}/data/mg_nt_fixed/timing_Nt_4_Ns_96_b_6p20/seed_206107806"
input="${run_dir}/input.yaml"
gate="${campaign}/timing_gate.tsv"

[[ -x "${binary}" && -f "${source_input}" && ! -e "${gate}" ]] || {
    echo "missing timing prerequisite or existing timing gate" >&2
    exit 1
}
[[ "$(git -C "${source_dir}" rev-parse HEAD)" == "${expected_commit}" ]] || {
    echo "KLFT revision changed; rebuild the sole project P100 build first" >&2
    exit 1
}
[[ "$(sha256sum "${binary}" | awk '{print $1}')" == "${expected_binary_sha}" ]] || {
    echo "P100 executable hash mismatch" >&2
    exit 1
}
[[ "$(sha256sum "${project}/uv.lock" | awk '{print $1}')" == "${expected_uv_sha}" ]] || {
    echo "analysis dependency lock hash mismatch" >&2
    exit 1
}
mkdir -p "${run_dir}"
if [[ -e "${input}" ]]; then
    cmp --silent "${source_input}" "${input}" || {
        echo "existing timing input differs from campaign input" >&2
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

{
    printf 'campaign_type nonperturbative_mg_nt_timing\n'
    printf 'job_id %s\nnode %s\n' "${SLURM_JOB_ID}" "${SLURMD_NODENAME}"
    printf 'physical_Nt 4\nstored_L2 5\nNs 96\nbeta 6.20\nseed 206107806\n'
    printf 'sweeps 100\nmeasurement_interval 1\n'
    printf 'dirichlet_suite_measurement_interval 5\n'
    printf 'klft_commit %s\nbuild_path %s\n' "${expected_commit}" "${build}"
    printf 'heatbath_sha256 %s\nuv_lock_sha256 %s\n' \
        "${expected_binary_sha}" "${expected_uv_sha}"
    printf 'input_sha256 '
    sha256sum "${input}" | awk '{print $1}'
    printf 'cuda_version '
    nvcc --version | tail -n 1
    printf 'compiler '
    c++ --version | head -n 1
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
[[ "${summary_rows}" -eq 100 && "${correlator_rows}" -eq 4900 &&
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

# Ns^2*(Nt+Ns/2) proxies updates plus the full-distance correlator. The sum
# over all 100 new chains is 26.1581196581 largest-chain equivalents.
max_chain_hours=$(awk -v seconds="${wall_seconds}" \
    'BEGIN {printf "%.8f", seconds*80.0/3600.0}')
total_gpu_hours=$(awk -v maximum="${max_chain_hours}" \
    'BEGIN {printf "%.8f", maximum*26.15811965811966}')
status=$(awk -v maximum="${max_chain_hours}" -v total="${total_gpu_hours}" \
    'BEGIN {print (maximum <= 2.0 && total <= 12.0) ? "PASS" : "FAIL"}')
{
    printf 'gate\t%s\n' "${status}"
    printf 'wall_seconds_100_sweeps\t%s\n' "${wall_seconds}"
    printf 'projected_largest_chain_gpu_hours\t%s\n' "${max_chain_hours}"
    printf 'projected_total_gpu_hours\t%s\n' "${total_gpu_hours}"
    printf 'timing_job_id\t%s\n' "${SLURM_JOB_ID}"
} > "${gate}"
[[ "${status}" == PASS ]] || {
    echo "timing exceeds the preregistered resource gate" >&2
    exit 1
}
