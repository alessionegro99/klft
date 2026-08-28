#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 1 ]]; then
    echo "usage: $0 timing|production|diagnostics|analysis|summary|plots" >&2
    exit 2
fi
case "$1" in
    timing) script=submit_timing.slurm ;;
    production) script=submit_production.slurm ;;
    diagnostics) script=submit_diagnostics.slurm ;;
    analysis) script=submit_analysis.slurm ;;
    summary) script=submit_summary.slurm ;;
    plots) script=submit_plots.slurm ;;
    *)
        echo "usage: $0 timing|production|diagnostics|analysis|summary|plots" >&2
        exit 2
        ;;
esac

remote_root=/qbigwork2/negro/dirichlet
scripts="${remote_root}/project/data/beta_nt_fixed/beta_over_nt_1p8/corrected_volumes_nt3_nt4/qbig"
campaign="${remote_root}/campaigns/step3_corrected_volumes"
record="${campaign}/queued_jobs.tsv"
mkdir -p "${remote_root}/logs"
job=$(/opt/slurm/bin/sbatch --parsable "${scripts}/${script}")
job=${job%%;*}
if [[ ! -e "${record}" ]]; then
    printf 'stage\tjob_id\tscheduler_dependency\n' > "${record}"
fi
printf '%s\t%s\tnone\n' "$1" "${job}" >> "${record}"
printf '%s\n' "${job}"
