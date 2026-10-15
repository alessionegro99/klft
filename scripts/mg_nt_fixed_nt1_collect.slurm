#!/bin/bash
#SBATCH --job-name=mgnt-nt1-collect
#SBATCH --partition=batch
#SBATCH --exclude=lnode[13-17]
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/qbigwork2/negro/dirichlet/logs/mgnt-nt1-collect-%j.out
set -euo pipefail
project=/qbigwork2/negro/dirichlet
snapshot=$project/reviews/mg_nt_fixed_nt1
export UV_CACHE_DIR=$project/.uv-cache MPLCONFIGDIR=$project/.matplotlib
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$project/project"
for script in mg_nt_fixed_campaign mg_nt_fixed_verify mg_nt_fixed_logbook; do
  "$project/bin/uv" run --frozen python "$snapshot/scripts/$script.py"
done
"$project/bin/uv" run --frozen python "$snapshot/scripts/mg_nt_fixed_logbook.py" --check
