#!/bin/bash -l
#SBATCH --job-name=mgnt-nt1-y
#SBATCH --partition=batch
#SBATCH --exclude=lnode16
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:pascal:1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/qbigwork2/negro/dirichlet/logs/mgnt-nt1-y-%A_%a.out
set -euo pipefail
: "${SLURM_JOB_ID:?Submit through Slurm}"
case "${1:-}" in
    smoke|timing|production) ;;
    *) echo 'Specify smoke, timing or production; submit each stage manually' >&2; exit 1 ;;
esac
source /hadron/negro/.env/klft_env.sh
dirichlet_driver=/usr/lib/x86_64-linux-gnu/libcuda.so.1
[[ -r "$dirichlet_driver" ]]
export LD_PRELOAD="${dirichlet_driver}${LD_PRELOAD:+:${LD_PRELOAD}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export OMP_PROC_BIND=close OMP_PLACES=cores
nvidia-smi --query-gpu=index,uuid,name,memory.used,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv
cd /qbigwork2/negro/dirichlet/project
/qbigwork2/negro/dirichlet/bin/uv run --frozen python \
    /qbigwork2/negro/dirichlet/campaigns/mg_nt_fixed/mg_nt_fixed_nt1_run.py \
    "$1" --campaign nt1_scaling --index "${SLURM_ARRAY_TASK_ID:-0}"
