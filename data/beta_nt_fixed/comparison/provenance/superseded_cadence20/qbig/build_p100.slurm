#!/bin/bash -l
#SBATCH --job-name=bntf-build
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:pascal:1
#SBATCH --mem=8G
#SBATCH --time=00:45:00
#SBATCH --output=/qbigwork2/negro/dirichlet/logs/bntf-build-%j.out

set -euo pipefail

source /hadron/negro/.env/klft_env.sh
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OMP_PROC_BIND=close
export OMP_PLACES=cores

# P100 nodes add CUDA 9.2 headers to CPATH. Use only CUDA 11.5 here.
clean_cpath=
IFS=: read -r -a cpath_entries <<< "${CPATH:-}"
for entry in "${cpath_entries[@]}"; do
    [[ "${entry}" == /usr/local/cuda/include ]] && continue
    [[ -z "${clean_cpath}" ]] || clean_cpath+=:
    clean_cpath+="${entry}"
done
export CPATH="${clean_cpath}"

remote_root=/qbigwork2/negro/dirichlet
campaign="${remote_root}/campaigns/beta_nt_fixed/production"
expected_commit=$(tr -d '[:space:]' < "${campaign}/klft_commit.txt")
source_dir="${remote_root}/src/klft-${expected_commit:0:7}"
build="${remote_root}/build/klft-${expected_commit:0:7}-p100"
provenance="${campaign}/build_provenance.txt"

[[ -d "${source_dir}/.git" && ! -e "${provenance}" ]] || {
    echo "missing clean source or existing provenance" >&2
    exit 1
}
[[ "$(git -C "${source_dir}" rev-parse HEAD)" == "${expected_commit}" ]]
[[ -z "$(git -C "${source_dir}" status --porcelain --ignore-submodules=none)" ]]
[[ ! -e "${build}" ]] || {
    echo "refusing to reuse build directory ${build}" >&2
    exit 1
}

nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
unset CXX
host_cxx=$(command -v c++)
cmake -S "${source_dir}" -B "${build}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER="${host_cxx}" \
    -DCUDAToolkit_ROOT="${CUDA_ROOT}" \
    -DKLFT_NDIM=3 \
    -DKLFT_NC=2 \
    -DKLFT_ENABLE_STATISTICAL_TESTS=ON \
    -DKokkos_ENABLE_CUDA=ON \
    -DKokkos_ENABLE_OPENMP=ON \
    -DKokkos_ARCH_PASCAL60=ON
cmake --build "${build}" -j "${SLURM_CPUS_PER_TASK}"
ctest --test-dir "${build}" --output-on-failure \
    -R '^temporal_dirichlet_(deterministic|statistical)$'

binary="${build}/binaries/heatbath"
[[ -x "${binary}" ]]
{
    printf 'klft_commit %s\n' "${expected_commit}"
    printf 'source_path %s\n' "${source_dir}"
    printf 'build_path %s\n' "${build}"
    printf 'cmake_options %s\n' \
        'Release;KLFT_NDIM=3;KLFT_NC=2;CUDA=ON;OpenMP=ON;PASCAL60=ON'
    printf 'node %s\njob_id %s\n' "${SLURMD_NODENAME}" "${SLURM_JOB_ID}"
    cmake --version | head -1
    "${host_cxx}" --version | head -1
    nvcc --version | tail -1
    printf 'heatbath_sha256 '
    sha256sum "${binary}" | awk '{print $1}'
} > "${provenance}"
