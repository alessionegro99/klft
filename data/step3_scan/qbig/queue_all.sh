#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "usage: $0 build|timing|production|diagnostics|analysis|aggregate|plots" >&2
    exit 2
}

[[ "$#" -eq 1 ]] || usage
stage="$1"
case "${stage}" in
    build) script=build_step3_p100.slurm ;;
    timing) script=submit_step3_timing.slurm ;;
    production) script=submit_step3_production.slurm ;;
    diagnostics) script=submit_step3_diagnostics.slurm ;;
    analysis) script=submit_step3_analysis.slurm ;;
    aggregate) script=submit_step3_aggregate.slurm ;;
    plots) script=submit_step3_plots.slurm ;;
    *) usage ;;
esac

project=/qbigwork2/negro/dirichlet/project
scripts="${project}/data/step3_scan/qbig"
campaign=/qbigwork2/negro/dirichlet/campaigns/step3_scan
record="${campaign}/queued_jobs.tsv"
mkdir -p /qbigwork2/negro/dirichlet/logs

job=$(sbatch --parsable "${scripts}/${script}")
job=${job%%;*}
if [[ ! -e "${record}" ]]; then
    printf 'stage\tjob_id\tdependency\n' > "${record}"
fi
printf '%s\t%s\tnone\n' "${stage}" "${job}" >> "${record}"
printf '%s\n' "${job}"
