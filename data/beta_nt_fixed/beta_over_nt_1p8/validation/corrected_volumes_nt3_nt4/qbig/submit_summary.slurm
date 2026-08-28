#!/bin/bash -l
#SBATCH --job-name=bntf-cv-summary
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/qbigwork2/negro/dirichlet/logs/bntf-cv-summary-%j.out

set -euo pipefail
remote_root=/qbigwork2/negro/dirichlet
output="${remote_root}/analysis/step3_corrected_volumes"
for file in corrected_volume_summary.json corrected_volume_points.csv \
            corrected_volume_points.npz correlator_comparison_Nt3.npz \
            correlator_comparison_Nt4.npz; do
    [[ ! -e "${output}/${file}" ]] || {
        echo "refusing to overwrite ${output}/${file}" >&2
        exit 1
    }
done
export UV_CACHE_DIR="${remote_root}/.uv-cache"
cd "${remote_root}/project"
exec "${remote_root}/bin/uv" run --frozen python \
    scripts/beta_nt_fixed/corrected_volumes_summary.py \
    --pilot-results "${remote_root}/analysis/step3_scan" \
    --finite-volume-results "${remote_root}/analysis/step3_fv160" \
    --corrected-results "${output}" \
    --output "${output}"
