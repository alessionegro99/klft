#!/bin/bash -l
#SBATCH --job-name=orb-2p1-tune
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:pascal:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --array=1-5
#SBATCH --output=/qbigwork2/negro/orbifold/logs/orb-2p1-tune-%A_%a.out

# Acceptance calibration only: target 70--80%, reject sustained settings above 90%.
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
  1) label=n100; expected_input_sha=56a185f807f9a5133742305855eb501c516c2292d42611bdbbe03d85accda32f ;;
  2) label=n80; expected_input_sha=9bad2e572135c111a3a43df70cdba8e673da1e8ef2e6c16d88ec153812747c33 ;;
  3) label=n67; expected_input_sha=20acee5ac677ab125a26b2baf114e3dda3ce7b3d460937a7febf113f3e6742e8 ;;
  4) label=n57; expected_input_sha=8b8033c2c6b5264444832059cabb5eaf42972be25937bf906488a50951d36fc3 ;;
  5) label=n50; expected_input_sha=300b5bf3da703cc60c488a66aacce87c5569dcd525eeb1e1e6dceea969e35f5a ;;
  *) exit 2 ;;
esac
input="${campaign}/postwarm_${label}.yaml"
run_dir="${root}/data/orbifold_2p1_demo_20260910/postwarm_tune/${label}"
[[ -f "${smoke}/COMPLETE" ]]
[[ "$(git -C "${source_dir}" rev-parse HEAD)" == "${expected_commit}" ]]
[[ -z "$(git -C "${source_dir}" status --porcelain --ignore-submodules=none)" ]]
expected_executable_sha=$(awk -v exe="${executable}" '$2 == exe {print $1}' "${smoke}/build_provenance.txt")
[[ "${expected_executable_sha}" =~ ^[0-9a-f]{64}$ ]]
[[ "$(sha256sum "${executable}" | awk '{print $1}')" == "${expected_executable_sha}" ]]
[[ "$(sha256sum "${input}" | awk '{print $1}')" == "${expected_input_sha}" ]]
restart="${root}/data/orbifold_2p1_demo_20260910/initial_warmup/chain01/final.cfg"
expected_restart_sha=fbc36affb5e940427cf16656f4eb561a27857099405a4062754eb0caaf01436e
[[ -f "${root}/data/orbifold_2p1_demo_20260910/initial_warmup/chain01/COMPLETE" ]]
[[ "$(sha256sum "${restart}" | awk '{print $1}')" == "${expected_restart_sha}" ]]
[[ ! -e "${run_dir}" ]]
mkdir -p "${root}/data/orbifold_2p1_demo_20260910/postwarm_tune"
mkdir "${run_dir}"
cp "${input}" "${run_dir}/input.yaml"
{
  date -u
  printf 'job_id %s\narray_task_id %s\nnode %s\n' "${SLURM_JOB_ID}" "${SLURM_ARRAY_TASK_ID}" "${SLURMD_NODENAME}"
  printf 'klft_commit %s\nsource_dirty false\n' "${expected_commit}"
  printf 'orbifold_hmc_sha256 %s\ninput_sha256 %s\n' "${expected_executable_sha}" "${expected_input_sha}"
  printf 'purpose acceptance_calibration_not_equilibrium_or_production\n'
  printf 'configuration_input_sha256 %s\n' "${expected_restart_sha}"
  sha256sum "$0" "${smoke}/build_provenance.txt"
  nvidia-smi --query-gpu=uuid,name,driver_version --format=csv,noheader
} > "${run_dir}/run_metadata.txt"
cd "${run_dir}"
start_seconds=${SECONDS}
/usr/bin/time -v "${executable}" -f input.yaml
wall_seconds=$((SECONDS - start_seconds))
[[ "$(awk '!/^#/ && NF {n++} END {print n+0}' hmc.out)" == 100 ]]
[[ "$(awk '!/^#/ && NF {n++} END {print n+0}' diagnostic.out)" == 10 ]]
[[ "$(awk '!/^#/ && NF {n++} END {print n+0}' wilson.out)" == 0 ]]
[[ -s final.cfg ]]
! grep -Eiq '(^|[[:space:]])[-+]?(nan|inf)([[:space:]]|$)' hmc.out diagnostic.out
grep -Fq '# dimensions 32 32 32' hmc.out
grep -Fq '# theory SU(3) orbifold, 2+1D, periodic, time_direction=2' hmc.out
{
  printf 'wall_seconds %s\n' "${wall_seconds}"
  awk '!/^#/ && NF {a+=$3;n++;d=$4<0?-$4:$4;if(d>m)m=d} END {printf "acceptance %.17g\nmax_abs_delta_h %.17g\n",a/n,m}' hmc.out
  awk '!/^#/ && NF {q=int(n++/5);a[q]+=$3;w[q]+=$4;c[q]++} END {for(q=0;q<2;q++)printf "window_%d action=%.17g W11=%.17g\n",q+1,a[q]/c[q],w[q]/c[q]}' diagnostic.out
  awk '!/^#/ && NF {q=int(n++/50);a[q]+=$3;c[q]++} END {for(q=0;q<2;q++)printf "acceptance_window50_%d %.17g\n",q+1,a[q]/c[q]}' hmc.out
  sha256sum input.yaml hmc.out diagnostic.out wilson.out final.cfg
} >> run_metadata.txt
touch COMPLETE
