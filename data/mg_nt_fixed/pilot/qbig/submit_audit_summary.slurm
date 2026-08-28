#!/bin/bash -l
#SBATCH --job-name=mgnt-audit-s
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=/qbigwork2/negro/dirichlet/logs/mgnt-audit-s-%j.out

set -euo pipefail
remote_root=/qbigwork2/negro/dirichlet
project="${remote_root}/project"
campaign="${remote_root}/campaigns/mg_nt_fixed"
output="${remote_root}/analysis/mg_nt_fixed/audit"
export UV_CACHE_DIR="${remote_root}/.uv-cache"
[[ ! -e "${campaign}/audit_gate.tsv" ]] || {
    echo "refusing to overwrite existing audit gate" >&2
    exit 1
}
cd "${project}"
"${remote_root}/bin/uv" run --frozen python scripts/mg_nt_fixed_audit.py \
    --prior-points "${campaign}/prior_points.tsv" \
    --project-root "${project}" \
    --audit-results "${output}" \
    --output "${output}"
cp "${output}/audit_gate.tsv" "${campaign}/audit_gate.tsv"
