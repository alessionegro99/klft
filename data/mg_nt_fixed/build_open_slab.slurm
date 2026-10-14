#!/bin/bash -l
#SBATCH --job-name=dir-open-build
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --nodelist=lcpunode02
#SBATCH --mem=8G
#SBATCH --time=00:45:00
#SBATCH --output=/qbigwork2/negro/dirichlet/logs/dir-open-build-%j.out

set -euo pipefail
: "${SLURM_JOB_ID:?Submit this build through Slurm, not on the login node}"
: "${SLURM_CPUS_PER_TASK:?Missing Slurm CPU allocation}"

remote_root=/qbigwork2/negro/dirichlet
source_dir="${remote_root}/src/klft"
build_dir="${remote_root}/build/klft"
records="${remote_root}/campaigns/mg_nt_fixed"
provenance="${records}/open_slab_build.txt"
expected_commit=c78efbd1ed6c60fc809db54ac9309fa532854982

[[ "$(git -C "${source_dir}" rev-parse HEAD)" == "${expected_commit}" ]]
[[ -z "$(git -C "${source_dir}" status --porcelain --ignore-submodules=none)" ]]
[[ ! -e "${provenance}" ]] || {
    echo "Refusing to overwrite existing build provenance: ${provenance}" >&2
    exit 1
}
# The historical executable/build must be preserved explicitly before replacing
# the sole active P100 build. This script never moves or deletes an old build.
if [[ -f "${build_dir}/CMakeCache.txt" ]] &&
   ! grep -Fxq "CMAKE_HOME_DIRECTORY:INTERNAL=${source_dir}" "${build_dir}/CMakeCache.txt"; then
    echo "Preserve the historical build before configuring the open-slab source" >&2
    exit 1
fi

source /hadron/negro/.env/klft_env.sh
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OMP_PROC_BIND=close
export OMP_PLACES=cores
# Some compute nodes inject CUDA 9.2 headers; use the configured CUDA 11.5 stack.
dirichlet_cpath=
IFS=: read -r -a dirichlet_cpath_entries <<< "${CPATH:-}"
for entry in "${dirichlet_cpath_entries[@]}"; do
    [[ -z "${entry}" || "${entry}" == /usr/local/cuda/include ]] && continue
    [[ -z "${dirichlet_cpath}" ]] || dirichlet_cpath+=:
    dirichlet_cpath+="${entry}"
done
export CPATH="${dirichlet_cpath}"
unset CXX
host_cxx=$(command -v c++)

cmake -S "${source_dir}" -B "${build_dir}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER="${host_cxx}" \
    -DCMAKE_CXX_FLAGS='-Wall -Wextra' \
    -DCUDAToolkit_ROOT="${CUDA_ROOT}" \
    -DKLFT_NDIM=3 -DKLFT_NC=2 \
    -DBUILD_TESTING=ON -DKLFT_ENABLE_STATISTICAL_TESTS=ON \
    -DKokkos_ENABLE_CUDA=ON -DKokkos_ENABLE_CUDA_LAMBDA=ON \
    -DKokkos_ENABLE_OPENMP=ON \
    -DKokkos_ARCH_PASCAL60=ON
cmake --build "${build_dir}" -j "${SLURM_CPUS_PER_TASK}"

# CUDA executables are tested in a separate, manually submitted P100 job.
mkdir -p "${records}"
set -o noclobber
{
    printf 'klft_commit %s\nsource_path %s\nbuild_path %s\n' \
        "${expected_commit}" "${source_dir}" "${build_dir}"
    printf 'job_id %s\nnode %s\n' "${SLURM_JOB_ID}" "${SLURMD_NODENAME}"
    printf 'cmake_options %s\n' \
        'Release;-Wall -Wextra;KLFT_NDIM=3;KLFT_NC=2;CUDA=ON;CUDA_LAMBDA=ON;OpenMP=ON;PASCAL60=ON;statistical_tests=ON'
    cmake --version | sed -n '1p'
    "${host_cxx}" --version | sed -n '1p'
    nvcc --version
    git -C "${source_dir}" submodule status --recursive
    printf 'heatbath_sha256 %s\n' "$(sha256sum "${build_dir}/binaries/heatbath" | cut -d ' ' -f 1)"
    printf 'cmake_cache_sha256 %s\n' "$(sha256sum "${build_dir}/CMakeCache.txt" | cut -d ' ' -f 1)"
    for artifact in binaries/heatbath binaries/gradient_flow_check \
        tests/partitioning_test tests/dirichlet_test \
        tests/dirichlet_statistical_test tests/orbifold_test tests/dirichlet_benchmark; do
        printf 'artifact_sha256 %s\n' "$(sha256sum "${build_dir}/${artifact}")"
    done
    printf 'gpu_validation pending\n'
} > "${provenance}"
