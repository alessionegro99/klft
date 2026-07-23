#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -gt 1 || ("$#" -eq 1 && "${1:-}" != "--dry-run") ]]; then
    echo "usage: $0 --dry-run | $0" >&2
    exit 2
fi

local_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)/analysis/step3_fv160/"
remote_dir="negro@qbig:/qbigwork2/negro/dirichlet/analysis/step3_fv160/"
options=(
    -avzh
    --partial
    --progress
    --timeout=30
    -e "ssh -o BatchMode=yes -o ConnectTimeout=10"
    --exclude='*.log'
)
if [[ "${1:-}" == "--dry-run" ]]; then
    options+=(--dry-run)
fi
mkdir -p "${local_dir}"
rsync "${options[@]}" -- "${remote_dir}" "${local_dir}"
