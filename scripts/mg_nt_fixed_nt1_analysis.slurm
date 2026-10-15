#!/bin/bash
#SBATCH --job-name=mgnt-nt1-analysis
#SBATCH --partition=batch
#SBATCH --exclude=lnode[13-17]
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/qbigwork2/negro/dirichlet/logs/mgnt-nt1-analysis-%A_%a.out
set -euo pipefail
project=/qbigwork2/negro/dirichlet
snapshot=$project/reviews/mg_nt_fixed_nt1
export UV_CACHE_DIR=$project/.uv-cache MPLCONFIGDIR=$project/.matplotlib
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$project/project"
driver=mg_nt_fixed_adaptive.py
if [[ "${1:-}" == "--first-stage-only" ]]; then
  driver=mg_nt_fixed_parallel.py
fi
"$project/bin/uv" run --frozen python "$snapshot/scripts/$driver" \
  --manifest "$snapshot/data/mg_nt_fixed/nt1_analysis_manifest.json" \
  --index "${SLURM_ARRAY_TASK_ID:?explicit Nt1 manifest index required}" \
  --output-root "$project/analysis/mg_nt_fixed" "$@"
