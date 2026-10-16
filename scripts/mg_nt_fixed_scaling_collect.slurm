#!/bin/bash
#SBATCH --job-name=mgnt-scaling-collect
#SBATCH --partition=batch
#SBATCH --exclude=lnode[13-17]
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/qbigwork2/negro/dirichlet/logs/mgnt-scaling-collect-%j.out
set -euo pipefail
project=/qbigwork2/negro/dirichlet
snapshot=$project/reviews/mg_nt_fixed_scaling
export UV_CACHE_DIR=$project/.uv-cache MPLCONFIGDIR=$project/.matplotlib
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$project/project"
if [[ "${1:-}" == "--tests-only" ]]; then
  PYTHONPATH="$snapshot/scripts" "$project/bin/uv" run --frozen python -m unittest \
    test_mg_nt_fixed_scaling_analysis test_mg_nt_fixed_nt1_analysis \
    test_mg_nt_fixed_inventory test_mg_nt_fixed_campaign \
    test_mg_nt_fixed_statistics test_mg_nt_fixed_parallel \
    test_mg_nt_fixed_scaling test_mg_nt_fixed_scaling_schedule
  exit
fi
case "${1:-}" in
  --collect-only) steps=(mg_nt_fixed_campaign mg_nt_fixed_verify);;
  --plots-only) steps=(mg_nt_fixed_scaling_plots);;
  --logbook-only) steps=(mg_nt_fixed_logbook);;
  "") steps=(mg_nt_fixed_campaign mg_nt_fixed_verify mg_nt_fixed_logbook);;
  *) exit 2;;
esac
for script in "${steps[@]}"; do
  "$project/bin/uv" run --frozen python "$snapshot/scripts/$script.py"
done
if [[ "${1:-}" != "--collect-only" && "${1:-}" != "--plots-only" ]]; then
  "$project/bin/uv" run --frozen python "$snapshot/scripts/mg_nt_fixed_logbook.py" --check
fi
