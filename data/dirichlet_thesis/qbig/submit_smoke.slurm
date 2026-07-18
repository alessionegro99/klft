#!/bin/bash -l
#SBATCH --job-name=dir-smoke
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --nodelist=lnode13
#SBATCH --gres=gpu:1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/qbigwork2/negro/dirichlet/logs/smoke-%j.out

set -euo pipefail

source /hadron/negro/.env/klft_env.sh
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OMP_PROC_BIND=close
export OMP_PLACES=cores

build=/hadron/negro/build/klft-p100-3D-su2
source_dir=/hadron/negro/src/klft
run_dir=/qbigwork2/negro/dirichlet/data/smoke_Nt_1_Ns_8_b_12p15266

shopt -s nullglob
outputs=("${run_dir}"/*.out "${run_dir}"/*.klft)
if (( ${#outputs[@]} != 0 )); then
    echo "refusing to append to existing smoke output in ${run_dir}" >&2
    exit 1
fi

nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
ctest --test-dir "${build}" --output-on-failure \
    -R '^temporal_dirichlet_deterministic$'
"${build}/tests/dirichlet_statistical_test"
cd "${run_dir}"
"${build}/binaries/heatbath" -f input.yaml

git -C "${source_dir}" rev-parse HEAD
sha256sum "${build}/binaries/heatbath"
