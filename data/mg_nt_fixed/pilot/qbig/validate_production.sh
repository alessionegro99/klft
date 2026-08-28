#!/usr/bin/env bash
set -euo pipefail

remote_root=/qbigwork2/negro/dirichlet
campaign="${remote_root}/campaigns/mg_nt_fixed"
manifest="${campaign}/manifest.tsv"
validated=0

while IFS=$'\t' read -r array_index campaign_type target_c run input_relative \
    purpose physical_nt stored_l2 ns beta chain_index seed sweeps base_interval \
    suite_interval expected_input_sha _binary_sha _commit _uv_sha; do
    run_dir="${remote_root}/data/${run}"
    input="${run_dir}/input.yaml"
    metadata="${run_dir}/run_metadata.txt"
    for file in "${input}" "${metadata}" "${run_dir}/holonomy_summary.out" \
                "${run_dir}/holonomy_correlator.out" \
                "${run_dir}/plaquette_profile.out" \
                "${run_dir}/configuration_final.klft"; do
        [[ -f "${file}" ]] || {
            echo "task ${array_index}: missing ${file}" >&2
            exit 1
        }
    done
    [[ "$(sha256sum "${input}" | awk '{print $1}')" == \
       "${expected_input_sha}" ]] || {
        echo "task ${array_index}: input hash mismatch" >&2
        exit 1
    }
    [[ "$(awk '$1 == "campaign_type" {print $2}' "${metadata}")" == \
       "${campaign_type}" &&
       "$(awk '$1 == "target_c" {print $2}' "${metadata}")" == "${target_c}" &&
       "$(awk '$1 == "physical_Nt" {print $2}' "${metadata}")" == \
       "${physical_nt}" &&
       "$(awk '$1 == "stored_L2" {print $2}' "${metadata}")" == \
       "${stored_l2}" &&
       "$(awk '$1 == "Ns" {print $2}' "${metadata}")" == "${ns}" &&
       "$(awk '$1 == "beta" {print $2}' "${metadata}")" == "${beta}" &&
       "$(awk '$1 == "chain_index" {print $2}' "${metadata}")" == \
       "${chain_index}" &&
       "$(awk '$1 == "seed" {print $2}' "${metadata}")" == "${seed}" &&
       "$(awk '$1 == "sweeps" {print $2}' "${metadata}")" == "${sweeps}" &&
       "$(awk '$1 == "measurement_interval" {print $2}' "${metadata}")" == \
       "${base_interval}" &&
       "$(awk '$1 == "dirichlet_suite_measurement_interval" {print $2}' \
           "${metadata}")" == "${suite_interval}" ]] || {
        echo "task ${array_index}: metadata mismatch" >&2
        exit 1
    }

    summary_rows=$(awk '!/^#/ {n++} END {print n+0}' \
        "${run_dir}/holonomy_summary.out")
    correlator_rows=$(awk '!/^#/ {n++} END {print n+0}' \
        "${run_dir}/holonomy_correlator.out")
    profile_rows=$(awk '!/^#/ {n++} END {print n+0}' \
        "${run_dir}/plaquette_profile.out")
    [[ "${summary_rows}" -eq "${sweeps}" &&
       "${correlator_rows}" -eq $((sweeps * (ns / 2 + 1))) &&
       "${profile_rows}" -eq \
           $((sweeps / suite_interval * (2 * physical_nt + 1))) ]] || {
        echo "task ${array_index}: output count mismatch" >&2
        exit 1
    }
    awk -v last="${sweeps}" \
        '!/^#/ {seen[$1]++} END {for (step=1; step<=last; step++)
            if (seen[step] != 1) bad++; exit bad != 0}' \
        "${run_dir}/holonomy_summary.out" || {
        echo "task ${array_index}: summary grid mismatch" >&2
        exit 1
    }
    awk -v last="${sweeps}" -v maximum_r="$((ns / 2))" \
        '!/^#/ {seen[$1 SUBSEP $2]++; if ($2 == 0 && $3 != 1) bad++}
         END {for (step=1; step<=last; step++)
                  for (r=0; r<=maximum_r; r++)
                      if (seen[step SUBSEP r] != 1) bad++;
              exit bad != 0}' "${run_dir}/holonomy_correlator.out" || {
        echo "task ${array_index}: correlator grid or C_G(0) mismatch" >&2
        exit 1
    }
    awk -v last="${sweeps}" -v cadence="${suite_interval}" \
        -v rows="$((2 * physical_nt + 1))" \
        '!/^#/ {seen[$1]++; if ($1 % cadence != 0) bad++}
         END {for (step=cadence; step<=last; step+=cadence)
                  if (seen[step] != rows) bad++;
              exit bad != 0}' "${run_dir}/plaquette_profile.out" || {
        echo "task ${array_index}: profile grid mismatch" >&2
        exit 1
    }
    for disabled in boundary_wilson.out w_temp.out w_mu_nu.out \
                    bulk_horizontal_polyakov.out; do
        [[ ! -e "${run_dir}/${disabled}" ]] || {
            echo "task ${array_index}: unexpected ${disabled}" >&2
            exit 1
        }
    done
    validated=$((validated + 1))
done < <(tail -n +2 "${manifest}")

[[ "${validated}" -eq 100 ]] || {
    echo "validated ${validated} manifest rows, expected 100" >&2
    exit 1
}
printf 'validated_chains\t%s\n' "${validated}"
