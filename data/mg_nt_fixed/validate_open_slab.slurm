#!/bin/bash -l
#SBATCH --job-name=dir-open-check
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:pascal:1
#SBATCH --mem=8G
#SBATCH --time=00:45:00
#SBATCH --output=/qbigwork2/negro/dirichlet/logs/dir-open-check-%j.out

set -euo pipefail
: "${SLURM_JOB_ID:?Submit GPU validation through Slurm}"
: "${SLURM_CPUS_PER_TASK:?Missing Slurm CPU allocation}"
case "${SLURMD_NODENAME}" in
    lnode13|lnode14|lnode15|lnode16|lnode17) ;;
    *) echo "Validation requires one of the project's P100 nodes" >&2; exit 1 ;;
esac

remote_root=/qbigwork2/negro/dirichlet
source_dir="${remote_root}/src/klft"
build_dir="${remote_root}/build/klft"
records="${remote_root}/campaigns/mg_nt_fixed"
provenance="${records}/open_slab_build.txt"
validation="${records}/open_slab_validation.txt"
expected_commit=c78efbd1ed6c60fc809db54ac9309fa532854982

[[ "$(git -C "${source_dir}" rev-parse HEAD)" == "${expected_commit}" ]]
[[ -z "$(git -C "${source_dir}" status --porcelain --ignore-submodules=none)" ]]
[[ -f "${provenance}" && ! -e "${validation}" ]]
grep -Fxq "klft_commit ${expected_commit}" "${provenance}"
binary_hash=$(sha256sum "${build_dir}/binaries/heatbath" | cut -d ' ' -f 1)
grep -Fxq "heatbath_sha256 ${binary_hash}" "${provenance}"
cache_hash=$(sha256sum "${build_dir}/CMakeCache.txt" | cut -d ' ' -f 1)
grep -Fxq "cmake_cache_sha256 ${cache_hash}" "${provenance}"
awk '$1 == "artifact_sha256" {print $2 "  " $3}' "${provenance}" | sha256sum --check -

source /hadron/negro/.env/klft_env.sh
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OMP_PROC_BIND=close
export OMP_PLACES=cores
nvidia-smi --query-gpu=index,uuid,name,memory.used,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv
ctest --test-dir "${build_dir}" --output-on-failure

# Diagnostic scaling only: the benchmark uses two overrelaxations. Production
# inputs must separately time 100 sweeps with FIVE overrelaxations, holonomy
# every sweep and profiles every five sweeps before applying the cost gates.
# These candidate sizes do not establish an adequate physical volume.
for ns in 128 512 2048; do
    "${build_dir}/tests/dirichlet_benchmark" "${ns}" 1 3
done

set -o noclobber
{
    printf 'klft_commit %s\nheatbath_sha256 %s\n' "${expected_commit}" "${binary_hash}"
    printf 'job_id %s\nnode %s\n' "${SLURM_JOB_ID}" "${SLURMD_NODENAME}"
    printf 'ctest passed\nbenchmark passed\nproduction_timing pending\n'
} > "${validation}"
# No dependent jobs or production submissions are made here.
