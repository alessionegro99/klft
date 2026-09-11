#!/bin/bash -l
#SBATCH --job-name=orb-2p1-equil
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:pascal:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --array=1-3
#SBATCH --output=/qbigwork2/negro/orbifold/logs/orb-2p1-equil-%A_%a.out

# Independent-chain warmup at the calibrated integrator; no production data.
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
  1) label=chain01; expected_input_sha=9534196b48f8c15c7bd75dd8caeecbc8a887bcdbd4bd6006c9e51ddb0fec8ce1; expected_restart_sha=fbc36affb5e940427cf16656f4eb561a27857099405a4062754eb0caaf01436e ;;
  2) label=chain02; expected_input_sha=ca81bffde80e7fd2fe9b12d20e1482d9de4f46d6c248e6bc83cfd847b8a1f0bc; expected_restart_sha=0d7a1b20ca9cc0e977015dfc00c79147d327df42c1248b7b90f9072a2df55cc7 ;;
  3) label=chain03; expected_input_sha=c36702fd8d09d705f698a1e52e9056f71892e0a793c9536ea870c8e30f146469; expected_restart_sha=e7322e5fa23937d3180fc9325e96ef594606c1e1fcd66340a855c6e5d4b80c85 ;;
  *) exit 2 ;;
esac
input="${campaign}/equilibrate_${label}.yaml"
run_dir="${root}/data/orbifold_2p1_demo_20260910/equilibration/${label}"
restart_dir="${root}/data/orbifold_2p1_demo_20260910/initial_warmup/${label}"
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
mkdir -p "${root}/data/orbifold_2p1_demo_20260910/equilibration"
mkdir "${run_dir}"
cp "${input}" "${run_dir}/input.yaml"
{
  date -u
  printf 'job_id %s\narray_task_id %s\nnode %s\n' "${SLURM_JOB_ID}" "${SLURM_ARRAY_TASK_ID}" "${SLURMD_NODENAME}"
  printf 'klft_commit %s\nsource_dirty false\n' "${expected_commit}"
  printf 'orbifold_hmc_sha256 %s\ninput_sha256 %s\n' "${expected_executable_sha}" "${expected_input_sha}"
  printf 'purpose continued_warmup_not_equilibrium_or_production\n'
  printf 'restart_sha256 %s\n' "${expected_restart_sha}"
  sha256sum "$0" "${smoke}/build_provenance.txt"
  nvidia-smi --query-gpu=uuid,name,driver_version --format=csv,noheader
} > "${run_dir}/run_metadata.txt"
cd "${run_dir}"
start_seconds=${SECONDS}
/usr/bin/time -v "${executable}" -f input.yaml
wall_seconds=$((SECONDS - start_seconds))
[[ "$(awk '!/^#/ && NF {n++} END {print n+0}' hmc.out)" == 1000 ]]
[[ "$(awk '!/^#/ && NF {n++} END {print n+0}' diagnostic.out)" == 100 ]]
[[ "$(awk '!/^#/ && NF {n++} END {print n+0}' wilson.out)" == 0 ]]
[[ -s final.cfg ]]
! grep -Eiq '(^|[[:space:]])[-+]?(nan|inf)([[:space:]]|$)' hmc.out diagnostic.out
grep -Fq '# dimensions 32 32 32' hmc.out
grep -Fq '# theory SU(3) orbifold, 2+1D, periodic, time_direction=2' hmc.out
{
  printf 'wall_seconds %s\n' "${wall_seconds}"
  awk '!/^#/ && NF {a+=$3;n++;d=$4<0?-$4:$4;if(d>m)m=d} END {printf "acceptance %.17g\nmax_abs_delta_h %.17g\n",a/n,m}' hmc.out
  awk '!/^#/ && NF {q=int(n++/25);a[q]+=$3;w[q]+=$4;c[q]++} END {for(q=0;q<4;q++)printf "window_%d action=%.17g W11=%.17g\n",q+1,a[q]/c[q],w[q]/c[q]}' diagnostic.out
  awk '!/^#/ && NF {q=int(n++/100);a[q]+=$3;c[q]++} END {for(q=0;q<10;q++)printf "acceptance_window100_%d %.17g\n",q+1,a[q]/c[q]}' hmc.out
  sha256sum input.yaml hmc.out diagnostic.out wilson.out final.cfg
} >> run_metadata.txt
# This is only an acceptance gate, not a stationarity claim.
awk '!/^#/ && NF {a+=$3;n++} END {exit !(n == 1000 && a/n >= 0.70 && a/n <= 0.90)}' hmc.out
touch COMPLETE

