#!/bin/bash -l
#SBATCH --job-name=step3-build
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:pascal:1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/qbigwork2/negro/dirichlet/logs/step3-build-%j.out

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
build="${remote_root}/build/klft"
provenance="${campaign}/build_provenance.txt"

actual_commit=$(git -C "${source_dir}" rev-parse HEAD)
if [[ "${actual_commit}" != "${expected_commit}" ]]; then
    echo "KLFT commit ${actual_commit} does not match ${expected_commit}" >&2
    exit 1
fi
if [[ -n "$(git -C "${source_dir}" status --porcelain --untracked-files=no)" ]]; then
    echo "KLFT tracked worktree is dirty" >&2
    exit 1
fi
if [[ -e "${provenance}" ]]; then
    echo "refusing to overwrite Step-3 build provenance" >&2
    exit 1
fi

mkdir -p "${campaign}" "$(dirname "${build}")"
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
cmake --build "${build}" --target clean
cmake --build "${build}" -j "${SLURM_CPUS_PER_TASK}"
ctest --test-dir "${build}" --output-on-failure \
    -R '^temporal_dirichlet_(deterministic|statistical)$'

{
    printf 'klft_commit %s\n' "${actual_commit}"
    printf 'build_path %s\n' "${build}"
    printf 'cmake_options %s\n' \
        'Release;KLFT_NDIM=3;KLFT_NC=2;CUDA=ON;OpenMP=ON;PASCAL60=ON'
    printf 'node %s\n' "${SLURMD_NODENAME}"
    printf 'job_id %s\n' "${SLURM_JOB_ID}"
    cmake --version | head -1
    "${host_cxx}" --version | head -1
    nvcc --version | tail -1
    gcc --version | head -1
    printf 'heatbath_sha256 '
    sha256sum "${build}/binaries/heatbath" | awk '{print $1}'
} > "${provenance}"
