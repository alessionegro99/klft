#!/usr/bin/env bash
# One-shot automatic dispatch, explicitly authorized by the user.
set -euo pipefail
[[ "$(hostname)" == lnode19 && -z "${SLURM_JOB_ID:-}" ]]
source /hadron/negro/setup_qbig3.sh
source /hadron/negro/load_modules_qbig3.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /qbigwork2/negro/dirichlet/project
# 48-hour resource wait plus bounded two-hour workers and shutdown margin.
exec timeout --signal=TERM --kill-after=30s 52h \
    /qbigwork2/negro/dirichlet/bin/uv run --frozen python \
    /qbigwork2/negro/dirichlet/campaigns/mg_nt_fixed/mg_nt_fixed_scaling_schedule.py
