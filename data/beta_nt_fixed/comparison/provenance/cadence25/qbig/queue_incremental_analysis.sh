#!/usr/bin/env bash
set -euo pipefail

remote_root=/qbigwork2/negro/dirichlet
project="${remote_root}/project"
campaign="${remote_root}/campaigns/beta_nt_fixed/production/cadence25"
record="${campaign}/incremental_analysis_jobs.tsv"
sbatch=/opt/slurm/bin/sbatch
production_array=282274

[[ ! -e "${record}" ]] || {
    echo "refusing to overwrite ${record}" >&2
    exit 1
}

printf 'ensemble_index\tproduction_tasks\tdiagnostics_job\tanalysis_job\n' \
    > "${record}"

cd "${project}"
for ensemble_index in 0 1 2 3 4 5 6 7 8 9; do
    if [[ "${ensemble_index}" -eq 2 ]]; then
        printf '2\t282274_8:282274_9:282274_10:282274_11\t282354\t282355\n' \
            >> "${record}"
        continue
    fi
    first_task=$((4 * ensemble_index))
    dependencies=""
    production_tasks=""
    for offset in 0 1 2 3; do
        task="${production_array}_$((first_task + offset))"
        dependencies="${dependencies:+${dependencies}:}${task}"
        production_tasks="${production_tasks:+${production_tasks}:}${task}"
    done
    diagnostics_job=$(
        "${sbatch}" --parsable --array="${ensemble_index}" \
            --dependency="afterok:${dependencies}" \
            data/beta_nt_fixed/production/cadence25/qbig/submit_diagnostics.slurm
    )
    diagnostics_job="${diagnostics_job%%;*}"
    analysis_job=$(
        "${sbatch}" --parsable --array="${ensemble_index}" \
            --dependency="afterok:${diagnostics_job}_*" \
            data/beta_nt_fixed/production/cadence25/qbig/submit_analysis.slurm
    )
    analysis_job="${analysis_job%%;*}"
    printf '%s\t%s\t%s\t%s\n' \
        "${ensemble_index}" "${production_tasks}" \
        "${diagnostics_job}" "${analysis_job}" >> "${record}"
done

sed -n '1,12p' "${record}"
