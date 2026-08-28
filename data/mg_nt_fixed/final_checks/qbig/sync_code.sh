#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -gt 1 || ("$#" -eq 1 && "$1" != "--apply") ]]; then
    echo "usage: $0 [--apply]" >&2
    exit 2
fi
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
files=(
    scripts/beta_nt_fixed/analysis.py
    scripts/mg_nt_fixed_tuning.py
    scripts/mg_nt_fixed_final_checks.py
)
options=(-avR --partial)
if [[ "$#" -eq 0 ]]; then options+=(--dry-run); fi
cd "${root}"
rsync "${options[@]}" -- "${files[@]}" negro@qbig:/qbigwork2/negro/dirichlet/project/
