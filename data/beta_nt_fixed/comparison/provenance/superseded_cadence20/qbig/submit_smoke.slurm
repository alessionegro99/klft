#!/bin/bash -l
#SBATCH --job-name=bntf-smoke
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:pascal:1
#SBATCH --mem=5G
#SBATCH --time=00:20:00
#SBATCH --output=/qbigwork2/negro/dirichlet/logs/bntf-smoke-%j.out

set -euo pipefail

source /hadron/negro/.env/klft_env.sh
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OMP_PROC_BIND=close
export OMP_PLACES=cores

remote_root=/qbigwork2/negro/dirichlet
base_campaign="${remote_root}/campaigns/beta_nt_fixed/production"
campaign="${base_campaign}${BNTF_CAMPAIGN_SUFFIX:+/${BNTF_CAMPAIGN_SUFFIX}}"
validation_suffix="${BNTF_VALIDATION_SUFFIX:-}"
expected_commit=$(tr -d '[:space:]' < "${campaign}/klft_commit.txt")
build="${remote_root}/build/klft-${expected_commit:0:7}-p100"
binary="${build}/binaries/heatbath"
expected_binary_sha=$(awk '$1 == "heatbath_sha256" {print $2}' \
    "${base_campaign}/build_provenance.txt")
source_input="${campaign}/smoke/input.yaml"
expected_input_sha=$(awk '{print $1}' "${campaign}/smoke/sha256.txt")
run_dir="${remote_root}/data/beta_nt_fixed/validation/smoke-${expected_commit:0:7}${validation_suffix}"
input="${run_dir}/input.yaml"
gate="${campaign}/smoke_gate.tsv"

[[ -x "${binary}" && -f "${source_input}" && ! -e "${gate}" ]]
[[ "$(sha256sum "${binary}" | awk '{print $1}')" == \
   "${expected_binary_sha}" ]]
[[ "$(sha256sum "${source_input}" | awk '{print $1}')" == \
   "${expected_input_sha}" ]]
mkdir -p "${run_dir}"
[[ ! -e "${input}" ]] || {
    echo "refusing to reuse smoke directory" >&2
    exit 1
}
cp "${source_input}" "${input}"

ns=$(awk '$1 == "L0:" {print $2}' "${input}")
[[ "$(awk '$1 == "L1:" {print $2}' "${input}")" == "${ns}" ]]
stored_l2=$(awk '$1 == "L2:" {print $2}' "${input}")
physical_nt=$((stored_l2 - 1))
sweeps=$(awk '$1 == "nSweep:" {print $2}' "${input}")
beta=$(awk '$1 == "beta:" {print $2}' "${input}")
seed=$(awk '$1 == "seed:" {print $2}' "${input}")
base_interval=$(awk '$1 == "measurement_interval:" {print $2}' "${input}")
profile_interval=$(awk '$1 == "dirichlet_suite_measurement_interval:" {print $2}' "${input}")
boundary_interval=$(awk '$1 == "boundary_wilson_measurement_interval:" {print $2}' "${input}")
boundary_max_r=$(awk '$1 == "boundary_wilson_max_r:" {print $2}' "${input}")
[[ "${physical_nt}" -gt 0 && "${sweeps}" -eq "${boundary_interval}" ]]
[[ "$((boundary_interval % profile_interval))" -eq 0 ]]
grep -q '^  temporal_dirichlet: true$' "${input}"
grep -q '^  measure_boundary_wilson_loops: true$' "${input}"
grep -q '^  measure_boundary_wilson_all_heights: true$' "${input}"

{
    printf 'campaign beta_nt_fixed_smoke%s\n' "${validation_suffix}"
    printf 'job_id %s\nnode %s\n' "${SLURM_JOB_ID}" "${SLURMD_NODENAME}"
    printf 'physical_Nt %s\nstored_L2 %s\nNs %s\nbeta %s\n' \
        "${physical_nt}" "${stored_l2}" "${ns}" "${beta}"
    printf 'seed %s\nsweeps %s\nmeasurement_interval %s\n' \
        "${seed}" "${sweeps}" "${base_interval}"
    printf 'profile_interval %s\nboundary_interval %s\n' \
        "${profile_interval}" "${boundary_interval}"
    printf 'boundary_max_R %s\n' "${boundary_max_r}"
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
boundary_rows=$(awk '!/^#/ {count++} END {print count+0}' boundary_wilson.out)
expected_correlator_rows=$((sweeps * (ns / 2 + 1)))
expected_profile_rows=$((sweeps / profile_interval * (2 * physical_nt + 1)))
expected_boundary_rows=$((sweeps / boundary_interval * \
    (boundary_max_r + 1) * physical_nt))
[[ "${summary_rows}" -eq "${sweeps}" ]]
[[ "${correlator_rows}" -eq "${expected_correlator_rows}" ]]
[[ "${profile_rows}" -eq "${expected_profile_rows}" ]]
[[ "${boundary_rows}" -eq "${expected_boundary_rows}" ]]
awk '!/^#/ && $2 == 0 && $3 != 1 {bad++} END {exit bad != 0}' \
    holonomy_correlator.out
awk '!/^#/ && $2 == 0 && $4 != 1 {bad++} END {exit bad != 0}' \
    boundary_wilson.out
awk -v last="${sweeps}" -v height="${physical_nt}" \
    'FNR == NR {if ($1 !~ /^#/ && $1 == last) c[$2]=$3; next}
     $1 !~ /^#/ && $3 == height {
         if (!($2 in c) || ($4-c[$2])^2 > 4e-22) bad++
     }
     END {exit bad != 0}' holonomy_correlator.out boundary_wilson.out
awk -v cadence="${profile_interval}" -v last="${sweeps}" \
    -v rows="$((2 * physical_nt + 1))" \
    '!/^#/ {seen[$1]++; if ($1 % cadence != 0) bad++}
     END {for (step=cadence; step<=last; step+=cadence)
              if (seen[step] != rows) bad++;
          exit bad != 0}' plaquette_profile.out
awk -v cadence="${boundary_interval}" -v last="${sweeps}" \
    -v rows="$(((boundary_max_r + 1) * physical_nt))" \
    '!/^#/ {seen[$1]++; if ($1 % cadence != 0) bad++}
     END {for (step=cadence; step<=last; step+=cadence)
              if (seen[step] != rows) bad++;
          exit bad != 0}' boundary_wilson.out
for disabled in w_temp.out w_mu_nu.out bulk_horizontal_polyakov.out; do
    [[ ! -e "${disabled}" ]]
done
[[ -s configuration_final.klft ]]
printf 'gate\tPASS\n' > "${gate}"
