#!/bin/bash -l
#SBATCH --job-name=bntf-cv-plots
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=00:30:00
#SBATCH --output=/qbigwork2/negro/dirichlet/logs/bntf-cv-plots-%j.out

set -euo pipefail
remote_root=/qbigwork2/negro/dirichlet
output="${remote_root}/analysis/step3_corrected_volumes"
plots="${output}/plots"
[[ -f "${output}/corrected_volume_summary.json" ]] || {
    echo "missing corrected-volume summary" >&2
    exit 1
}
if compgen -G "${plots}/*.pdf" >/dev/null; then
    echo "refusing to overwrite existing corrected-volume plots" >&2
    exit 1
fi
export UV_CACHE_DIR="${remote_root}/.uv-cache"
cd "${remote_root}/project"
exec "${remote_root}/bin/uv" run --frozen python \
    scripts/beta_nt_fixed/plot_corrected_volumes.py \
    --pilot-results "${remote_root}/analysis/step3_scan" \
    --corrected-results "${output}" \
    --output "${plots}"
