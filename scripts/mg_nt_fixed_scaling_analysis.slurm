#!/bin/bash
#SBATCH --job-name=mgnt-scaling-analysis
#SBATCH --partition=batch
#SBATCH --exclude=lnode[13-17]
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/qbigwork2/negro/dirichlet/logs/mgnt-scaling-analysis-%A_%a.out
set -euo pipefail
project=/qbigwork2/negro/dirichlet
snapshot=$project/reviews/mg_nt_fixed_scaling
manifest=$snapshot/data/mg_nt_fixed/scaling_analysis_manifest.json
if [[ "${1:-}" == "--nt1" ]]; then
  manifest=$snapshot/data/mg_nt_fixed/nt1_scaling_analysis_manifest.json
  shift
fi
export UV_CACHE_DIR=$project/.uv-cache MPLCONFIGDIR=$project/.matplotlib
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$project/project"
"$project/bin/uv" run --frozen python "$snapshot/scripts/mg_nt_fixed_parallel.py" \
  --manifest "$manifest" \
  --index "${SLURM_ARRAY_TASK_ID:?explicit matched-y manifest index required}" \
  --output-root "$project/analysis/mg_nt_fixed" "$@"
