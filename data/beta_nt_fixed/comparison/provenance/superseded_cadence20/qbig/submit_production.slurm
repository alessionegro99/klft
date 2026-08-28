#!/bin/bash -l
#SBATCH --job-name=bntf-prod
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:pascal:1
#SBATCH --mem=8G
#SBATCH --time=3-23:30:00
#SBATCH --array=0-39
#SBATCH --output=/qbigwork2/negro/dirichlet/logs/bntf-prod-%A_%a.out

set -euo pipefail

source /hadron/negro/.env/klft_env.sh
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OMP_PROC_BIND=close
export OMP_PLACES=cores

remote_root=/qbigwork2/negro/dirichlet
base_campaign="${remote_root}/campaigns/beta_nt_fixed/production"
campaign="${base_campaign}${BNTF_CAMPAIGN_SUFFIX:+/${BNTF_CAMPAIGN_SUFFIX}}"
[[ "$(awk -F '\t' '$1 == "gate" {print $2}' \
    "${campaign}/smoke_gate.tsv")" == PASS ]]
[[ "$(awk -F '\t' '$1 == "gate" {print $2}' \
    "${campaign}/timing_gate.tsv")" == PASS ]]

manifest="${campaign}/manifest.tsv"
line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "${manifest}")
[[ -n "${line}" ]]
IFS=$'\t' read -r array_index run physical_nt stored_l2 ns beta \
    beta_over_nt g3_squared_l expected_xi boundary_max_r chain_index seed \
    sweeps base_interval profile_interval boundary_interval expected_input_sha \
    <<< "${line}"
[[ "${array_index}" -eq "${SLURM_ARRAY_TASK_ID}" ]]

expected_commit=$(tr -d '[:space:]' < "${campaign}/klft_commit.txt")
source_dir="${remote_root}/src/klft-${expected_commit:0:7}"
build="${remote_root}/build/klft-${expected_commit:0:7}-p100"
binary="${build}/binaries/heatbath"
expected_binary_sha=$(awk '$1 == "heatbath_sha256" {print $2}' \
    "${base_campaign}/build_provenance.txt")
source_input="${campaign}/${run}/input.yaml"
run_dir="${remote_root}/data/beta_nt_fixed/production/${run}"
input="${run_dir}/input.yaml"

[[ -x "${binary}" && -f "${source_input}" ]]
[[ "$(git -C "${source_dir}" rev-parse HEAD)" == "${expected_commit}" ]]
[[ -z "$(git -C "${source_dir}" status --porcelain --ignore-submodules=none)" ]]
[[ "$(sha256sum "${binary}" | awk '{print $1}')" == \
   "${expected_binary_sha}" ]]
[[ "$(sha256sum "${source_input}" | awk '{print $1}')" == \
   "${expected_input_sha}" ]]
mkdir -p "${run_dir}"
[[ ! -e "${input}" ]] || {
    echo "refusing to append to existing run ${run_dir}" >&2
    exit 1
}
cp "${source_input}" "${input}"

[[ "$(awk '$1 == "L0:" {print $2}' "${input}")" == "${ns}" ]]
[[ "$(awk '$1 == "L1:" {print $2}' "${input}")" == "${ns}" ]]
[[ "$(awk '$1 == "L2:" {print $2}' "${input}")" == "${stored_l2}" ]]
[[ "$(awk '$1 == "nSweep:" {print $2}' "${input}")" == "${sweeps}" ]]
[[ "$(awk '$1 == "beta:" {print $2}' "${input}")" == "${beta}" ]]
[[ "$(awk '$1 == "seed:" {print $2}' "${input}")" == "${seed}" ]]
grep -q '^  temporal_dirichlet: true$' "${input}"
grep -q '^  measure_boundary_wilson_loops: true$' "${input}"
grep -q '^  measure_boundary_wilson_all_heights: true$' "${input}"

{
    printf 'campaign beta_nt_fixed_production\n'
    printf 'job_id %s_%s\n' "${SLURM_ARRAY_JOB_ID}" "${SLURM_ARRAY_TASK_ID}"
    printf 'node %s\ncuda_visible_devices %s\n' \
        "${SLURMD_NODENAME}" "${CUDA_VISIBLE_DEVICES:-unset}"
    printf 'physical_Nt %s\nstored_L2 %s\nNs %s\nbeta %s\n' \
        "${physical_nt}" "${stored_l2}" "${ns}" "${beta}"
    printf 'beta_over_Nt %s\ng3_squared_L %s\n' \
        "${beta_over_nt}" "${g3_squared_l}"
    printf 'chain_index %s\nseed %s\nsweeps %s\n' \
        "${chain_index}" "${seed}" "${sweeps}"
    printf 'measurement_interval %s\nprofile_interval %s\n' \
        "${base_interval}" "${profile_interval}"
    printf 'boundary_interval %s\nboundary_max_R %s\n' \
        "${boundary_interval}" "${boundary_max_r}"
    printf 'klft_commit %s\nbuild_path %s\n' "${expected_commit}" "${build}"
    printf 'heatbath_sha256 %s\ninput_sha256 %s\n' \
        "${expected_binary_sha}" "${expected_input_sha}"
} > "${run_dir}/run_metadata.txt"

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
awk -v first="${boundary_interval}" -v last="${sweeps}" \
    -v maxr="${boundary_max_r}" -v height="${physical_nt}" \
    'FNR == NR {
         if ($1 !~ /^#/ && ($1 == first || $1 == last) && $2 <= maxr)
             c[$1 SUBSEP $2]=$3;
         next
     }
     $1 !~ /^#/ && ($1 == first || $1 == last) && $3 == height {
         key=$1 SUBSEP $2;
         if (!(key in c) || ($4-c[key])^2 > 4e-22) bad++
     }
     END {exit bad != 0}' holonomy_correlator.out boundary_wilson.out
for disabled in w_temp.out w_mu_nu.out bulk_horizontal_polyakov.out; do
    [[ ! -e "${disabled}" ]]
done
[[ -s configuration_final.klft ]]
printf 'status complete\nvalidated_at_utc %s\n' "$(date -u +%FT%TZ)" \
    > run_complete.txt
