#!/bin/bash -l
#SBATCH --job-name=bntf-fv-smoke
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:pascal:1
#SBATCH --mem=5G
#SBATCH --time=00:20:00
#SBATCH --output=/qbigwork2/negro/dirichlet/logs/bntf-fv-smoke-%j.out

set -euo pipefail

source /hadron/negro/.env/klft_env.sh
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OMP_PROC_BIND=close
export OMP_PLACES=cores

remote_root=/qbigwork2/negro/dirichlet
campaign="${remote_root}/campaigns/step3_fv160"
build="${remote_root}/build/klft"
binary="${build}/binaries/heatbath"
expected_commit=$(tr -d '[:space:]' < "${campaign}/klft_commit.txt")
expected_binary_sha=$(tr -d '[:space:]' < "${campaign}/heatbath_sha256.txt")
source_input="${campaign}/smoke/input.yaml"
expected_input_sha=$(awk '{print $1}' "${campaign}/smoke/sha256.txt")
run_dir="${remote_root}/data/step3_fv160_smoke_Nt_2_Ns_160_b_3p6"
input="${run_dir}/input.yaml"

[[ -x "${binary}" && -f "${source_input}" ]] || {
    echo "missing verified fixed-beta/Nt binary or smoke input" >&2
    exit 1
}
[[ "$(sha256sum "${binary}" | awk '{print $1}')" == \
   "${expected_binary_sha}" ]] || {
    echo "fixed-beta/Nt binary checksum changed" >&2
    exit 1
}
[[ "$(sha256sum "${source_input}" | awk '{print $1}')" == \
   "${expected_input_sha}" ]] || {
    echo "smoke input checksum changed" >&2
    exit 1
}
mkdir -p "${run_dir}"
if [[ -e "${input}" ]]; then
    cmp --silent "${source_input}" "${input}" || {
        echo "staged smoke input differs" >&2
        exit 1
    }
else
    cp "${source_input}" "${input}"
fi
if compgen -G "${run_dir}/*.out" >/dev/null ||
   compgen -G "${run_dir}/*.klft" >/dev/null ||
   [[ -e "${run_dir}/run_metadata.txt" ]]; then
    echo "refusing to append to existing smoke output" >&2
    exit 1
fi

[[ "$(awk '$1 == "L0:" {print $2}' "${input}")" == 160 ]]
[[ "$(awk '$1 == "L1:" {print $2}' "${input}")" == 160 ]]
[[ "$(awk '$1 == "L2:" {print $2}' "${input}")" == 3 ]]
[[ "$(awk '$1 == "nSweep:" {print $2}' "${input}")" == 100 ]]
[[ "$(awk '$1 == "beta:" {print $2}' "${input}")" == 3.6 ]]
grep -q '^  temporal_dirichlet: true$' "${input}"
grep -q '^  measurement_interval: 1$' "${input}"
grep -q '^  dirichlet_suite_measurement_interval: 5$' "${input}"
grep -q '^  measure_dirichlet_plaquette_profiles: true$' "${input}"
grep -q '^  measure_boundary_wilson_loops: false$' "${input}"

{
    printf 'campaign beta_nt_fixed_finite_volume_nt2_smoke\n'
    printf 'job_id %s\nnode %s\n' "${SLURM_JOB_ID}" "${SLURMD_NODENAME}"
    printf 'physical_Nt 2\nstored_L2 3\nNs 160\nbeta 3.6\n'
    printf 'seed 376035314\nsweeps 100\nmeasurement_interval 1\n'
    printf 'dirichlet_suite_measurement_interval 5\n'
    printf 'klft_commit %s\nbuild_path %s\n' "${expected_commit}" "${build}"
    printf 'heatbath_sha256 %s\ninput_sha256 %s\n' \
        "${expected_binary_sha}" "${expected_input_sha}"
} > "${run_dir}/run_metadata.txt"

nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
cd "${run_dir}"
/usr/bin/time -v "${binary}" -f input.yaml

summary_rows=$(awk '!/^#/ {count++} END {print count+0}' holonomy_summary.out)
correlator_rows=$(awk '!/^#/ {count++} END {print count+0}' holonomy_correlator.out)
profile_rows=$(awk '!/^#/ {count++} END {print count+0}' plaquette_profile.out)
[[ "${summary_rows}" -eq 100 && "${correlator_rows}" -eq 8100 &&
   "${profile_rows}" -eq 100 ]] || {
    echo "smoke output counts are incorrect" >&2
    exit 1
}
awk '!/^#/ && $2 == 0 && $3 != 1 {bad++} END {exit bad != 0}' \
    holonomy_correlator.out
awk '!/^#/ {seen[$1]++; if ($1 % 5 != 0) bad++}
     END {for (step=5; step<=100; step+=5) if (seen[step] != 5) bad++;
          exit bad != 0}' plaquette_profile.out
for disabled in boundary_wilson.out w_temp.out w_mu_nu.out \
                bulk_horizontal_polyakov.out; do
    [[ ! -e "${disabled}" ]] || {
        echo "disabled observable unexpectedly wrote ${disabled}" >&2
        exit 1
    }
done
