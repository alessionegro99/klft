#!/bin/bash -l
#SBATCH --job-name=orb-2p1-pilot
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:pascal:1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --array=1-3
#SBATCH --output=/qbigwork2/negro/orbifold/logs/orb-2p1-pilot-%A_%a.out

# First measured pilot. Acceptance is checked here; equilibrium/plateaus separately.
set -euo pipefail
: "${SLURM_JOB_ID:?Submit through Slurm}"
case "${SLURMD_NODENAME:-}" in
  lnode13|lnode14|lnode15|lnode16|lnode17) ;;
  *) echo 'Expected a qbig P100 node.' >&2; exit 2 ;;
esac
source /hadron/negro/.env/klft_env.sh
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OMP_PROC_BIND=close
export OMP_PLACES=cores

root=/qbigwork2/negro/orbifold
campaign="${root}/campaigns/orbifold_2p1_demo_20260910"
source_dir="${root}/src/klft-1104e1a"
executable="${root}/build/klft-1104e1a-su3-2p1-p100/binaries/orbifold_hmc"
smoke="${root}/data/orbifold_2p1_demo_20260910/gpu_smoke_1104e1a"
expected_commit=1104e1af3fa39838d1ae6fdb9c2620f7b423b134
case "${SLURM_ARRAY_TASK_ID}" in
  1) label=chain01; expected_input_sha=09c6bb375e819bb00f0ecbf33f3d3a3324ae3c91961aaea898ca4f0810a9377d; expected_restart_sha=1b3ab8a54f105fd2e5e9f416de11760924a1243e6a58f15a86a33bb179e68004 ;;
  2) label=chain02; expected_input_sha=27683b2db05e74ee6431028ccb2fd3c424c89ce215c70ed92ed85aa744ab3e75; expected_restart_sha=5fb89a7d6b799248f65a6c84bd997c6808e745007b50a371f03fc464e744eae8 ;;
  3) label=chain03; expected_input_sha=f1ba52536e5a3bc9d9cc032dcc7736175719b9aa43658db2cda65ddb5d7d0c92; expected_restart_sha=2fa8584996025cf15e3afb61e3761ea0e009912008fa5c49285d721b25494c99 ;;
  *) exit 2 ;;
esac
input="${campaign}/pilot_${label}.yaml"
run_dir="${root}/data/orbifold_2p1_demo_20260910/measured_pilot/${label}"
restart_dir="${root}/data/orbifold_2p1_demo_20260910/equilibration/${label}"
[[ -f "${restart_dir}/COMPLETE" ]]
[[ "$(sha256sum "${restart_dir}/final.cfg" | awk '{print $1}')" == "${expected_restart_sha}" ]]
[[ -f "${smoke}/COMPLETE" ]]
[[ "$(git -C "${source_dir}" rev-parse HEAD)" == "${expected_commit}" ]]
[[ -z "$(git -C "${source_dir}" status --porcelain --ignore-submodules=none)" ]]
expected_executable_sha=$(awk -v exe="${executable}" '$2 == exe {print $1}' "${smoke}/build_provenance.txt")
[[ "${expected_executable_sha}" =~ ^[0-9a-f]{64}$ ]]
[[ "$(sha256sum "${executable}" | awk '{print $1}')" == "${expected_executable_sha}" ]]
[[ "$(sha256sum "${input}" | awk '{print $1}')" == "${expected_input_sha}" ]]
[[ ! -e "${run_dir}" ]]
mkdir -p "${root}/data/orbifold_2p1_demo_20260910/measured_pilot"
mkdir "${run_dir}"
cp "${input}" "${run_dir}/input.yaml"
{
  date -u
  printf 'job_id %s\narray_task_id %s\nnode %s\n' "${SLURM_JOB_ID}" "${SLURM_ARRAY_TASK_ID}" "${SLURMD_NODENAME}"
  printf 'klft_commit %s\nsource_dirty false\n' "${expected_commit}"
  printf 'orbifold_hmc_sha256 %s\ninput_sha256 %s\n' "${expected_executable_sha}" "${expected_input_sha}"
  printf 'purpose measured_pilot_not_yet_accepted_potential\n'
  printf 'restart_sha256 %s\n' "${expected_restart_sha}"
  sha256sum "$0" "${smoke}/build_provenance.txt"
  nvidia-smi --query-gpu=uuid,name,driver_version --format=csv,noheader
} > "${run_dir}/run_metadata.txt"
cd "${run_dir}"
start_seconds=${SECONDS}
/usr/bin/time -v "${executable}" -f input.yaml
wall_seconds=$((SECONDS - start_seconds))
[[ "$(awk '!/^#/ && NF {n++} END {print n+0}' hmc.out)" == 3500 ]]
[[ "$(awk '!/^#/ && NF {n++} END {print n+0}' diagnostic.out)" == 350 ]]
[[ "$(awk '!/^#/ && NF {n++} END {print n+0}' wilson.out)" == 76800 ]]
[[ -s final.cfg ]]
! grep -Eiq '(^|[[:space:]])[-+]?(nan|inf)([[:space:]]|$)' hmc.out diagnostic.out wilson.out
grep -Fq '# dimensions 32 32 32' hmc.out
grep -Fq '# theory SU(3) orbifold, 2+1D, periodic, time_direction=2' hmc.out
{
  printf 'wall_seconds %s\n' "${wall_seconds}"
  awk '!/^#/ && NF && $2 > 0 {a+=$3;n++;d=$4<0?-$4:$4;if(d>m)m=d} END {printf "acceptance %.17g\nmax_abs_delta_h %.17g\n",a/n,m}' hmc.out
  awk '!/^#/ && NF && $2 > 0 {q=int(n++/75);a[q]+=$3;w[q]+=$4;c[q]++} END {for(q=0;q<4;q++)printf "window_%d action=%.17g W11=%.17g\n",q+1,a[q]/c[q],w[q]/c[q]}' diagnostic.out
  awk '!/^#/ && NF && $2 > 0 {q=int(n++/500);a[q]+=$3;c[q]++} END {for(q=0;q<6;q++)printf "acceptance_window500_%d %.17g\n",q+1,a[q]/c[q]}' hmc.out
  sha256sum input.yaml hmc.out diagnostic.out wilson.out final.cfg
} >> run_metadata.txt
# This is only an acceptance gate, not a stationarity claim.
awk '!/^#/ && NF && $2 > 0 {a+=$3;n++} END {exit !(n == 3000 && a/n >= 0.70 && a/n <= 0.90)}' hmc.out
touch COMPLETE


