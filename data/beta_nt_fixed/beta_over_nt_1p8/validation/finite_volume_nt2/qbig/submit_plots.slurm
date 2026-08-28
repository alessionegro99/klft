#!/bin/bash -l
#SBATCH --job-name=bntf-fv-plots
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=00:30:00
#SBATCH --output=/qbigwork2/negro/dirichlet/logs/bntf-fv-plots-%j.out

set -euo pipefail
remote_root=/qbigwork2/negro/dirichlet
output="${remote_root}/analysis/step3_fv160"
plots="${output}/plots"
[[ -f "${output}/finite_volume_summary.json" ]] || {
    echo "missing finite-volume summary" >&2
    exit 1
}
if compgen -G "${plots}/*.pdf" >/dev/null; then
    echo "refusing to overwrite existing finite-volume plots" >&2
    exit 1
fi
export UV_CACHE_DIR="${remote_root}/.uv-cache"
cd "${remote_root}/project"
exec "${remote_root}/bin/uv" run --frozen python \
    scripts/beta_nt_fixed/plot_finite_volume_nt2.py \
    --pilot-results "${remote_root}/analysis/step3_scan" \
    --extension-results "${output}" \
    --output "${plots}"
