#!/usr/bin/env bash
set -euo pipefail

root=/qbigwork2/negro/dirichlet
manifest="${root}/campaigns/mg_nt_fixed/final_checks/manifest.tsv"
validated=0
while IFS=$'\t' read -r _ _ _ run _ _ physical_nt _ ns _ _ seed sweeps _ \
    suite_interval expected_input_sha _ _ _; do
    directory="${root}/data/${run}"
    for file in input.yaml run_metadata.txt holonomy_summary.out holonomy_correlator.out \
        plaquette_profile.out configuration_final.klft run_complete.txt; do [[ -f "${directory}/${file}" ]]; done
    [[ "$(sha256sum "${directory}/input.yaml" | awk '{print $1}')" == "${expected_input_sha}" ]]
    [[ "$(awk '$1 == "seed" {print $2}' "${directory}/run_metadata.txt")" == "${seed}" ]]
    [[ "$(awk '!/^#/ {n++} END {print n+0}' "${directory}/holonomy_summary.out")" -eq "${sweeps}" ]]
    [[ "$(awk '!/^#/ {n++} END {print n+0}' "${directory}/holonomy_correlator.out")" -eq $((sweeps * (ns / 2 + 1))) ]]
    [[ "$(awk '!/^#/ {n++} END {print n+0}' "${directory}/plaquette_profile.out")" -eq $((sweeps / suite_interval * (2 * physical_nt + 1))) ]]
    validated=$((validated + 1))
done < <(tail -n +2 "${manifest}")
[[ "${validated}" -eq 12 ]]
printf 'validated_chains\t%s\n' "${validated}"
