#!/usr/bin/env bash
set -euo pipefail

project=/qbigwork2/negro/dirichlet/project
scripts="${project}/data/step3_scan/qbig"
campaign=/qbigwork2/negro/dirichlet/campaigns/step3_scan
record="${campaign}/queued_jobs.tsv"
mkdir -p /qbigwork2/negro/dirichlet/logs
[[ ! -e "${record}" ]] || {
    echo "refusing to overwrite ${record}" >&2
    exit 1
}

if [[ "$#" -eq 0 ]]; then
    build_job=$(sbatch --parsable "${scripts}/build_step3_p100.slurm")
    build_job=${build_job%%;*}
elif [[ "$#" -eq 2 && "$1" == --completed-build && "$2" =~ ^[0-9]+$ ]]; then
    build_job="$2"
    build_state=$(sacct -X -n -j "${build_job}" --format=State |
        awk 'NF {print $1; exit}')
    [[ "${build_state}" == COMPLETED ]] || {
        echo "build job ${build_job} is ${build_state:-unknown}, not COMPLETED" >&2
        exit 1
    }
else
    echo "usage: $0 [--completed-build JOB_ID]" >&2
    exit 2
fi
timing_job=$(sbatch --parsable --dependency="afterok:${build_job}" \
    "${scripts}/submit_step3_timing.slurm")
timing_job=${timing_job%%;*}
production_job=$(sbatch --parsable --dependency="afterok:${timing_job}" \
    "${scripts}/submit_step3_production.slurm")
production_job=${production_job%%;*}
diagnostics_job=$(sbatch --parsable --dependency="afterok:${production_job}" \
    "${scripts}/submit_step3_diagnostics.slurm")
diagnostics_job=${diagnostics_job%%;*}
analysis_job=$(sbatch --parsable --dependency="afterok:${diagnostics_job}" \
    "${scripts}/submit_step3_analysis.slurm")
analysis_job=${analysis_job%%;*}
aggregate_job=$(sbatch --parsable --dependency="afterok:${analysis_job}" \
    "${scripts}/submit_step3_aggregate.slurm")
aggregate_job=${aggregate_job%%;*}
plots_job=$(sbatch --parsable --dependency="afterok:${aggregate_job}" \
    "${scripts}/submit_step3_plots.slurm")
plots_job=${plots_job%%;*}

{
    printf 'stage\tjob_id\tdependency\n'
    printf 'build\t%s\tnone\n' "${build_job}"
    printf 'timing\t%s\tafterok:%s\n' "${timing_job}" "${build_job}"
    printf 'production\t%s\tafterok:%s\n' "${production_job}" "${timing_job}"
    printf 'diagnostics\t%s\tafterok:%s\n' "${diagnostics_job}" "${production_job}"
    printf 'analysis\t%s\tafterok:%s\n' "${analysis_job}" "${diagnostics_job}"
    printf 'aggregate\t%s\tafterok:%s\n' "${aggregate_job}" "${analysis_job}"
    printf 'plots\t%s\tafterok:%s\n' "${plots_job}" "${aggregate_job}"
} | tee "${record}"
