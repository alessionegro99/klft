#!/bin/bash -l
#SBATCH --job-name=dir-timing
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --nodelist=lnode13
#SBATCH --gres=gpu:1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/qbigwork2/negro/dirichlet/logs/timing-%j.out

set -euo pipefail

source /hadron/negro/.env/klft_env.sh
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OMP_PROC_BIND=close
export OMP_PLACES=cores

build=/hadron/negro/build/klft-p100-3D-su2
run_dir=/qbigwork2/negro/dirichlet/data/timing_Nt_2_Ns_64_b_12p15266

shopt -s nullglob
outputs=("${run_dir}"/*.out "${run_dir}"/*.klft)
if (( ${#outputs[@]} != 0 )); then
    echo "refusing to append to existing timing output in ${run_dir}" >&2
    exit 1
fi

cd "${run_dir}"
/usr/bin/time -v "${build}/binaries/heatbath" -f input.yaml

measurements=$(awk '!/^#/ {count++} END {print count+0}' holonomy_summary.out)
if [[ "${measurements}" -ne 100 ]]; then
    echo "expected 100 measurements, found ${measurements}" >&2
    exit 1
fi
awk '!/^#/ && $2 == 0 && $3 != 1 {bad++} END {exit bad != 0}' \
    holonomy_correlator.out
git -C /hadron/negro/src/klft rev-parse HEAD
sha256sum "${build}/binaries/heatbath"
