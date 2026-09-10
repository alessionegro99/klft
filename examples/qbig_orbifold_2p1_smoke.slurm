#!/bin/bash -l
#SBATCH --job-name=orb-2p1-build
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:pascal:1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/qbigwork2/negro/orbifold/logs/orb-2p1-build-%j.out

# Submit with one inspected P100 node, e.g. sbatch --nodelist=lnode14 FILE.
# This validates a new executable; it does not launch a production ensemble.
set -euo pipefail
: "${SLURM_JOB_ID:?Submit through Slurm, not on the login node}"
case "${SLURMD_NODENAME:-}" in
  lnode13|lnode14|lnode15|lnode16|lnode17) ;;
  *) echo 'Expected a qbig P100 node (lnode13--17).' >&2; exit 2 ;;
esac

source /hadron/negro/.env/klft_env.sh
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export CUDA_LAUNCH_BLOCKING=1

# These nodes prepend obsolete CUDA 9.2 headers; use the selected toolkit.
clean_cpath=
IFS=: read -r -a cpath_entries <<< "${CPATH:-}"
for entry in "${cpath_entries[@]}"; do
  [[ -z "${entry}" || "${entry}" == /usr/local/cuda/include ]] && continue
  [[ -z "${clean_cpath}" ]] || clean_cpath+=:
  clean_cpath+="${entry}"
done
export CPATH="${clean_cpath}"

root=/qbigwork2/negro/orbifold
expected_commit=772db3d2bd3dbdda0a42686e3f545a3ee326373c
source_dir="${root}/src/klft-772db3d"
build="${root}/build/klft-772db3d-su3-2p1-p100"
run_dir="${root}/data/orbifold_2p1_demo_20260910/gpu_smoke_cuda_fix"
uv=/qbigwork2/negro/dirichlet/bin/uv
export UV_CACHE_DIR="${root}/.uv-cache"

[[ "$(git -C "${source_dir}" rev-parse HEAD)" == "${expected_commit}" ]]
[[ -z "$(git -C "${source_dir}" status --porcelain --ignore-submodules=none)" ]]
[[ ! -e "${build}" && ! -e "${run_dir}" ]]
mkdir -p "${root}/build" "${root}/data/orbifold_2p1_demo_20260910"
mkdir "${build}" "${run_dir}"
cp "${source_dir}/examples/orbifold_2p1_smoke.yaml" "${run_dir}/input.yaml"
export TMPDIR="${run_dir}"
unset CXX
host_cxx=$(command -v c++)
{
  date -u
  printf 'klft_commit %s\nsource_dirty false\n' "${expected_commit}"
  printf 'node %s\njob_id %s\nbuild_path %s\n' "${SLURMD_NODENAME}" "${SLURM_JOB_ID}" "${build}"
  git -C "${source_dir}" submodule status --recursive
  "${host_cxx}" --version
  nvcc --version
  cmake --version
  nvidia-smi --query-gpu=uuid,name,driver_version --format=csv,noheader
  sha256sum "$0" "${run_dir}/input.yaml"
} > "${run_dir}/build_provenance.txt"

cmake -S "${source_dir}" -B "${build}" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER="${host_cxx}" \
  -DCMAKE_CXX_FLAGS='-Wall -Wextra' -DCUDAToolkit_ROOT="${CUDA_ROOT}" \
  -DKLFT_NDIM=3 -DKLFT_NC=3 -DKokkos_ENABLE_CUDA=ON \
  -DKokkos_ENABLE_OPENMP=ON -DKokkos_ARCH_PASCAL60=ON
! grep -E -- '-ffast-math|-Ofast|--use_fast_math' "${build}/CMakeCache.txt"
cmake --build "${build}" -j "${SLURM_CPUS_PER_TASK}"
ctest --test-dir "${build}" --output-on-failure
"${uv}" run --offline --no-project "${source_dir}/tests/orbifold_driver_check.py" \
  "${build}/binaries/orbifold_hmc" 3 3

cd "${run_dir}"
"${build}/binaries/orbifold_hmc" -f input.yaml
[[ "$(awk '!/^#/ && NF {n++} END {print n+0}' hmc.out)" == 10 ]]
[[ "$(awk '!/^#/ && NF {n++} END {print n+0}' wilson.out)" == 20 ]]
[[ "$(awk '!/^#/ && NF {n++} END {print n+0}' diagnostic.out)" == 10 ]]
[[ -s final.cfg ]]
! grep -Eiq '(^|[[:space:]])[-+]?(nan|inf)([[:space:]]|$)' hmc.out wilson.out diagnostic.out
grep -Fq '# theory SU(3) orbifold, 2+1D, periodic, time_direction=2' hmc.out
{
  sha256sum "${build}/binaries/orbifold_hmc" "${build}/tests/orbifold_test" "${build}/CMakeCache.txt"
  grep -E '^(CMAKE_(BUILD_TYPE|CXX_COMPILER|CXX_FLAGS.*)|KLFT_N[CD]|Kokkos_(ENABLE_CUDA|ENABLE_OPENMP|ARCH_PASCAL60))' "${build}/CMakeCache.txt"
  awk '!/^#/ && NF {a+=$3;n++;d=$4<0?-$4:$4;if(d>m)m=d} END {printf "smoke_acceptance %.17g\nsmoke_max_abs_delta_h %.17g\n",a/n,m}' hmc.out
  sha256sum input.yaml hmc.out wilson.out diagnostic.out final.cfg
} >> build_provenance.txt
touch COMPLETE
