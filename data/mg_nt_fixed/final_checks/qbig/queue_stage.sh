#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 1 || ! "$1" =~ ^(smoke|timing|production|diagnostics|analysis|bootstrap10k|tuning)$ ]]; then
    echo "usage: $0 smoke|timing|production|diagnostics|analysis|bootstrap10k|tuning" >&2
    exit 2
fi
campaign=/qbigwork2/negro/dirichlet/campaigns/mg_nt_fixed/final_checks
record="${campaign}/queued_jobs.tsv"
job=$(/opt/slurm/bin/sbatch --parsable "${campaign}/qbig/submit_$1.slurm")
job=${job%%;*}
if [[ ! -e "${record}" ]]; then printf 'stage\tjob_id\tscheduler_dependency\n' > "${record}"; fi
printf '%s\t%s\tnone\n' "$1" "${job}" >> "${record}"
printf '%s\n' "${job}"
