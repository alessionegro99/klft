#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -gt 1 || ( "$#" -eq 1 && "$1" != "--apply" ) ]]; then
    echo "usage: $0 [--apply]" >&2
    exit 2
fi

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)
remote=negro@qbig:/qbigwork2/negro/dirichlet/analysis/beta_nt_fixed/
destination="${root}/analysis/beta_nt_fixed/"
options=(
    -av
    --partial
    --timeout=30
    -e "ssh -o HostName=131.220.226.30 -o BatchMode=yes"
    --exclude analysis.log
    --exclude diagnostics.log
    --exclude summary.log
)
if [[ "$#" -eq 0 ]]; then
    options+=(--dry-run)
fi
mkdir -p "${destination}"
rsync "${options[@]}" "${remote}" "${destination}"
