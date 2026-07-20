#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -gt 1 || ("$#" -eq 1 && "${1:-}" != "--dry-run") ]]; then
    echo "usage: $0 --dry-run | $0" >&2
    exit 2
fi

local_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)/analysis/pcm_beta_pilot/"
remote_dir="negro@qbig:/qbigwork2/negro/dirichlet/analysis/pcm_beta_pilot/"
options=(-avzh --partial --progress --exclude=analysis.log)
if [[ "${1:-}" == "--dry-run" ]]; then
    options+=(--dry-run)
fi
mkdir -p "${local_dir}"
rsync "${options[@]}" -- "${remote_dir}" "${local_dir}"
