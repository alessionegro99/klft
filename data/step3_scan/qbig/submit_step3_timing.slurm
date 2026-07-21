#!/bin/bash -l
#SBATCH --job-name=step3-timing
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:pascal:1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=/qbigwork2/negro/dirichlet/logs/step3-timing-%j.out

set -euo pipefail

source /hadron/negro/.env/klft_env.sh
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OMP_PROC_BIND=close
export OMP_PLACES=cores

remote_root=/qbigwork2/negro/dirichlet
project="${remote_root}/project"
campaign="${remote_root}/campaigns/step3_scan"
source_dir=/hadron/negro/src/klft
expected_commit=$(tr -d '[:space:]' < "${project}/data/step3_scan/klft_commit.txt")
short_commit=${expected_commit:0:7}
build=/hadron/negro/build/klft-p100-3D-su2-step3-${short_commit}
run_dir="${remote_root}/data/step3_timing_Nt_4_Ns_192_b_7p2"
source_input="${campaign}/timing/input.yaml"
input="${run_dir}/input.yaml"
gate="${campaign}/timing_gate.tsv"

[[ -x "${build}/binaries/heatbath" ]] || {
    echo "missing fresh Step-3 heatbath binary" >&2
    exit 1
}
[[ "$(git -C "${source_dir}" rev-parse HEAD)" == "${expected_commit}" ]] || {
    echo "KLFT source commit changed after build" >&2
    exit 1
}
[[ -f "${source_input}" && ! -e "${gate}" ]] || {
    echo "missing timing input or existing timing gate" >&2
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
    printf 'campaign step3_scan_timing\n'
    printf 'job_id %s\nnode %s\n' "${SLURM_JOB_ID}" "${SLURMD_NODENAME}"
    printf 'physical_Nt 4\nstored_L2 5\nNs 192\nbeta 7.2\nseed 2026072104\n'
    printf 'klft_commit %s\nbuild_path %s\n' "${expected_commit}" "${build}"
    printf 'heatbath_sha256 '
    sha256sum "${build}/binaries/heatbath" | awk '{print $1}'
    printf 'input_sha256 '
    sha256sum "${input}" | awk '{print $1}'
} > "${run_dir}/run_metadata.txt"

nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
cd "${run_dir}"
start_ns=$(date +%s%N)
/usr/bin/time -v "${build}/binaries/heatbath" -f input.yaml
end_ns=$(date +%s%N)
wall_seconds=$(awk -v start="${start_ns}" -v end="${end_ns}" \
    'BEGIN {printf "%.6f", (end-start)/1.0e9}')

summary_rows=$(awk '!/^#/ {count++} END {print count+0}' holonomy_summary.out)
correlator_rows=$(awk '!/^#/ {count++} END {print count+0}' holonomy_correlator.out)
profile_rows=$(awk '!/^#/ {count++} END {print count+0}' plaquette_profile.out)
[[ "${summary_rows}" -eq 100 && "${correlator_rows}" -eq 9700 &&
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

# The cost proxy Ns^2*(Nt+Ns/2) includes update volume and the full-R
# correlator. Summed over all 32 chains it is 8.134444 times the timed largest
# grid point. Scaling 100 to 10000 sweeps is deliberately conservative because
# it also scales startup overhead.
max_chain_hours=$(awk -v seconds="${wall_seconds}" \
    'BEGIN {printf "%.8f", seconds*100.0/3600.0}')
total_gpu_hours=$(awk -v maximum="${max_chain_hours}" \
    'BEGIN {printf "%.8f", maximum*8.134444444444445}')
status=PASS
if ! awk -v maximum="${max_chain_hours}" -v total="${total_gpu_hours}" \
    'BEGIN {exit !(maximum <= 2.0 && total <= 12.0)}'; then
    status=FAIL
fi
{
    printf 'gate\t%s\n' "${status}"
    printf 'wall_seconds_100_sweeps\t%s\n' "${wall_seconds}"
    printf 'projected_max_chain_hours\t%s\n' "${max_chain_hours}"
    printf 'projected_total_gpu_hours\t%s\n' "${total_gpu_hours}"
    printf 'max_chain_limit_hours\t2\n'
    printf 'total_limit_gpu_hours\t12\n'
} > "${gate}"
[[ "${status}" == PASS ]] || {
    echo "Step-3 timing projection exceeds a stop threshold" >&2
    exit 1
}
