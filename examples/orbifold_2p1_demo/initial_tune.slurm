#!/bin/bash -l
#SBATCH --job-name=orb-2p1-init
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:pascal:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --array=1-3
#SBATCH --output=/qbigwork2/negro/orbifold/logs/orb-2p1-init-%A_%a.out

# Initial hot-start stability only. Retune after warmup; no production data.
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
source_dir="${root}/src/klft-79380d5"
executable="${root}/build/klft-79380d5-su3-2p1-p100/binaries/orbifold_hmc"
smoke="${root}/data/orbifold_2p1_demo_20260910/gpu_smoke"
expected_commit=79380d592568a43246bfe6ca163c25609c5b56f1
case "${SLURM_ARRAY_TASK_ID}" in
  1) label=h0005; expected_input_sha=4611c9ed126291c5c65cc2356aab6d879565b181ecc2378f8423e40e4d4c4fe3 ;;
  2) label=h0010; expected_input_sha=9b72c45956affa742155d0f902dbc19a576313016fd33db8d03ffb4249743445 ;;
  3) label=h0020; expected_input_sha=17716da9177241deade6a7d3116af921332a86e15ce87d79bec73b806dea96f9 ;;
  *) exit 2 ;;
esac
input="${campaign}/initial_${label}.yaml"
run_dir="${root}/data/orbifold_2p1_demo_20260910/initial_tune/${label}"
[[ -f "${smoke}/COMPLETE" ]]
[[ "$(git -C "${source_dir}" rev-parse HEAD)" == "${expected_commit}" ]]
[[ -z "$(git -C "${source_dir}" status --porcelain --ignore-submodules=none)" ]]
expected_executable_sha=$(awk -v exe="${executable}" '$2 == exe {print $1}' "${smoke}/build_provenance.txt")
[[ "${expected_executable_sha}" =~ ^[0-9a-f]{64}$ ]]
[[ "$(sha256sum "${executable}" | awk '{print $1}')" == "${expected_executable_sha}" ]]
[[ "$(sha256sum "${input}" | awk '{print $1}')" == "${expected_input_sha}" ]]
[[ ! -e "${run_dir}" ]]
mkdir -p "${root}/data/orbifold_2p1_demo_20260910/initial_tune"
mkdir "${run_dir}"
cp "${input}" "${run_dir}/input.yaml"
{
  date -u
  printf 'job_id %s\narray_task_id %s\nnode %s\n' "${SLURM_JOB_ID}" "${SLURM_ARRAY_TASK_ID}" "${SLURMD_NODENAME}"
  printf 'klft_commit %s\nsource_dirty false\n' "${expected_commit}"
  printf 'orbifold_hmc_sha256 %s\ninput_sha256 %s\n' "${expected_executable_sha}" "${expected_input_sha}"
  printf 'purpose initial_stability_not_equilibrium_or_production\n'
  sha256sum "$0" "${smoke}/build_provenance.txt"
  nvidia-smi --query-gpu=uuid,name,driver_version --format=csv,noheader
} > "${run_dir}/run_metadata.txt"
cd "${run_dir}"
start_seconds=${SECONDS}
/usr/bin/time -v "${executable}" -f input.yaml
wall_seconds=$((SECONDS - start_seconds))
[[ "$(awk '!/^#/ && NF {n++} END {print n+0}' hmc.out)" == 60 ]]
[[ "$(awk '!/^#/ && NF {n++} END {print n+0}' diagnostic.out)" == 60 ]]
[[ "$(awk '!/^#/ && NF {n++} END {print n+0}' wilson.out)" == 0 ]]
[[ -s final.cfg ]]
! grep -Eiq '(^|[[:space:]])[-+]?(nan|inf)([[:space:]]|$)' hmc.out diagnostic.out
grep -Fq '# dimensions 32 32 32' hmc.out
grep -Fq '# theory SU(3) orbifold, 2+1D, periodic, time_direction=2' hmc.out
{
  printf 'wall_seconds %s\n' "${wall_seconds}"
  awk '!/^#/ && NF {a+=$3;n++;d=$4<0?-$4:$4;if(d>m)m=d} END {printf "acceptance %.17g\nmax_abs_delta_h %.17g\n",a/n,m}' hmc.out
  awk '!/^#/ && NF {q=int(n++/20);a[q]+=$3;w[q]+=$4;c[q]++} END {for(q=0;q<3;q++)printf "window_%d action=%.17g W11=%.17g\n",q+1,a[q]/c[q],w[q]/c[q]}' diagnostic.out
  sha256sum input.yaml hmc.out diagnostic.out wilson.out final.cfg
} >> run_metadata.txt
touch COMPLETE
