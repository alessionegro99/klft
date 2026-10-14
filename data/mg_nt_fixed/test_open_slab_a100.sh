#!/usr/bin/env bash
# User-authorized direct-node test. This script launches no production jobs.
set -euo pipefail
[[ "$(hostname)" == lnode19 ]] || { echo 'Run only on lnode19' >&2; exit 1; }

remote_root=/qbigwork2/negro/dirichlet
source_dir="${remote_root}/src/klft"
build_dir="${remote_root}/build/klft-a100"
records="${remote_root}/campaigns/mg_nt_fixed"
expected_commit=f47421d895469b72d4af8a9b8479807f2b09a692
gpu=GPU-439f10b2-3f2f-c6d2-12cc-deca128ca0f2

exec 9> "${records}/open_slab_a100.lock"
flock -n 9 || { echo 'Another project A100 test holds the lock' >&2; exit 1; }
[[ ! -e "${records}/open_slab_a100_started.txt" && ! -e "${build_dir}" ]]
[[ "$(git -C "${source_dir}" rev-parse HEAD)" == "${expected_commit}" ]]
[[ -z "$(git -C "${source_dir}" status --porcelain --ignore-submodules=none)" ]]

check_idle() {
    local pids
    pids=$(nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits)
    [[ -z "${pids}" ]] || { echo "A100 already occupied: ${pids}" >&2; exit 1; }
}
check_idle
set -o noclobber
printf 'pid %s\nnode %s\ngpu_uuid %s\nklft_commit %s\n' \
    "$$" "$(hostname)" "${gpu}" "${expected_commit}" > "${records}/open_slab_a100_started.txt"
trap 'result=$?; printf "exit_code %s\n" "${result}" > "${records}/open_slab_a100_exit.txt"' EXIT

source /hadron/negro/setup_qbig3.sh
source /hadron/negro/load_modules_qbig3.sh
export CUDA_VISIBLE_DEVICES="${gpu}"
export OMP_NUM_THREADS=1
export OMP_PROC_BIND=close
export OMP_PLACES=cores
unset CXX
host_cxx=$(command -v c++)
nvidia-smi -i "${gpu}" --query-gpu=index,uuid,name,driver_version,memory.used,utilization.gpu --format=csv

cmake -S "${source_dir}" -B "${build_dir}" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER="${host_cxx}" \
    -DCMAKE_CXX_FLAGS='-Wall -Wextra' -DCUDAToolkit_ROOT="${CUDA_ROOT}" \
    -DKLFT_NDIM=3 -DKLFT_NC=2 \
    -DBUILD_TESTING=ON -DKLFT_ENABLE_STATISTICAL_TESTS=ON \
    -DKokkos_ENABLE_CUDA=ON -DKokkos_ENABLE_CUDA_LAMBDA=ON \
    -DKokkos_ENABLE_OPENMP=ON -DKokkos_ARCH_AMPERE80=ON
cmake --build "${build_dir}" -j 4
binary_hash=$(sha256sum "${build_dir}/binaries/heatbath" | cut -d ' ' -f 1)
{
    printf 'klft_commit %s\nsource_path %s\nbuild_path %s\n' \
        "${expected_commit}" "${source_dir}" "${build_dir}"
    printf 'node lnode19\ngpu_uuid %s\nslab_boundary_flag 3\n' "${gpu}"
    printf 'cmake_options Release;-Wall -Wextra;KLFT_NDIM=3;KLFT_NC=2;CUDA=ON;CUDA_LAMBDA=ON;OpenMP=ON;AMPERE80=ON;statistical_tests=ON\n'
    cmake --version | sed -n '1p'
    "${host_cxx}" --version | sed -n '1p'
    nvcc --version
    git -C "${source_dir}" submodule status --recursive
    printf 'heatbath_sha256 %s\n' "${binary_hash}"
    sha256sum "${build_dir}/CMakeCache.txt" "${records}/test_open_slab_a100.sh" \
        "${records}/mg_nt_fixed_a100_test.py" "${records}/mg_nt_fixed_nt1_run.py"
    for artifact in binaries/heatbath binaries/gradient_flow_check tests/partitioning_test \
        tests/dirichlet_test tests/dirichlet_statistical_test tests/orbifold_test; do
        printf 'artifact_sha256 %s\n' "$(sha256sum "${build_dir}/${artifact}")"
    done
} > "${records}/open_slab_a100_build.txt"

check_idle
ctest --test-dir "${build_dir}" --output-on-failure --no-tests=error
check_idle
compute-sanitizer --tool memcheck --error-exitcode 99 \
    "${build_dir}/tests/dirichlet_test" "${source_dir}/partitionings/linear_m3.yaml"
{
    printf 'klft_commit %s\nheatbath_sha256 %s\n' "${expected_commit}" "${binary_hash}"
    printf 'node lnode19\ngpu_uuid %s\nctest passed\ncuda_memcheck passed\n' "${gpu}"
} > "${records}/open_slab_a100_validation.txt"

cd "${remote_root}/project"
"${remote_root}/bin/uv" run --frozen python "${records}/mg_nt_fixed_a100_test.py"
