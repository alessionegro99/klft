#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 1 ]]; then
    echo "usage: $0 timing|audit-diagnostics|audit-analysis|audit-summary|production|diagnostics|analysis|tuning|plots|confirmation-production|confirmation-diagnostics|confirmation-analysis|confirmation-gate" >&2
    exit 2
fi
case "$1" in
    timing) script=submit_timing.slurm ;;
    audit-diagnostics) script=submit_audit_diagnostics.slurm ;;
    audit-analysis) script=submit_audit_analysis.slurm ;;
    audit-summary) script=submit_audit_summary.slurm ;;
    production) script=submit_production.slurm ;;
    diagnostics) script=submit_diagnostics.slurm ;;
    analysis) script=submit_analysis.slurm ;;
    tuning) script=submit_tuning.slurm ;;
    plots) script=submit_plots.slurm ;;
    confirmation-production) script=submit_confirmation_production.slurm ;;
    confirmation-diagnostics) script=submit_confirmation_diagnostics.slurm ;;
    confirmation-analysis) script=submit_confirmation_analysis.slurm ;;
    confirmation-gate) script=submit_confirmation_gate.slurm ;;
    *)
        echo "usage: $0 timing|audit-diagnostics|audit-analysis|audit-summary|production|diagnostics|analysis|tuning|plots|confirmation-production|confirmation-diagnostics|confirmation-analysis|confirmation-gate" >&2
        exit 2
        ;;
esac

remote_root=/qbigwork2/negro/dirichlet
scripts="${remote_root}/project/data/mg_nt_fixed/pilot/qbig"
campaign="${remote_root}/campaigns/mg_nt_fixed"
record="${campaign}/queued_jobs.tsv"
mkdir -p "${remote_root}/logs"
job=$(/opt/slurm/bin/sbatch --parsable "${scripts}/${script}")
job=${job%%;*}
if [[ ! -e "${record}" ]]; then
    printf 'stage\tjob_id\tscheduler_dependency\n' > "${record}"
fi
printf '%s\t%s\tnone\n' "$1" "${job}" >> "${record}"
printf '%s\n' "${job}"
