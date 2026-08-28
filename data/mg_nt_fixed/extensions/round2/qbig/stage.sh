#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -gt 1 || ("$#" -eq 1 && "$1" != "--apply") ]]; then
    echo "usage: $0 [--apply]" >&2
    exit 2
fi
source_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/
remote=negro@qbig:/qbigwork2/negro/dirichlet/campaigns/mg_nt_fixed/extensions/round2/
options=(-av --partial --exclude='__pycache__' --exclude='queued_jobs.tsv')
if [[ "$#" -eq 0 ]]; then
    options+=(--dry-run)
else
    ssh negro@qbig '[[ ! -e /qbigwork2/negro/dirichlet/campaigns/mg_nt_fixed/extensions/round2/manifest.tsv ]]' || {
        echo "refusing to overwrite staged round-2 manifest" >&2
        exit 1
    }
    ssh negro@qbig 'mkdir -p /qbigwork2/negro/dirichlet/campaigns/mg_nt_fixed/extensions'
fi
rsync "${options[@]}" -- "${source_dir}" "${remote}"
