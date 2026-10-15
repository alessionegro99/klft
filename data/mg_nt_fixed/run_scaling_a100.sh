#!/usr/bin/env bash
# User-approved three-GPU launch; GPU 6 remains free.
set -euo pipefail
[[ "$(hostname)" == lnode19 && -z "${SLURM_JOB_ID:-}" ]]
case "${1:-}" in
    preflight|production) ;;
    *) echo 'Specify preflight or production' >&2; exit 1 ;;
esac
case "${2:-}" in
    0) gpu=GPU-439f10b2-3f2f-c6d2-12cc-deca128ca0f2; cpu=32 ;;
    1) gpu=GPU-94eef473-cba0-0850-d8e5-dc5dafa2493a; cpu=33 ;;
    2) gpu=GPU-97c59efd-ea9e-94d9-7f18-1959e77798e4; cpu=34 ;;
    *) echo 'Specify worker 0, 1 or 2' >&2; exit 1 ;;
esac
remote_root=/qbigwork2/negro/dirichlet
exec 9> "${remote_root}/campaigns/mg_nt_fixed/scaling_a100_worker_${2}.lock"
flock -n 9 || { echo 'Worker already running' >&2; exit 1; }
source /hadron/negro/setup_qbig3.sh
source /hadron/negro/load_modules_qbig3.sh
export CUDA_VISIBLE_DEVICES="${gpu}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export OMP_PROC_BIND=close OMP_PLACES=cores
cd "${remote_root}/project"
exec timeout --signal=TERM --kill-after=30s 2h taskset -c "${cpu}" \
    "${remote_root}/bin/uv" run --frozen python \
    "${remote_root}/campaigns/mg_nt_fixed/mg_nt_fixed_scaling_run.py" "$1" --worker "$2"
