#!/bin/bash -l
#SBATCH --job-name=dir-thesis
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:pascal:1
#SBATCH --mem=4G
#SBATCH --time=08:00:00
#SBATCH --array=0-11
#SBATCH --output=/qbigwork2/negro/dirichlet/logs/thesis-%A_%a.out

set -euo pipefail

source /hadron/negro/.env/klft_env.sh
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OMP_PROC_BIND=close
export OMP_PLACES=cores

build=/hadron/negro/build/klft-p100-3D-su2
source_dir=/hadron/negro/src/klft
data_root=/qbigwork2/negro/dirichlet/data
manifest=/qbigwork2/negro/dirichlet/campaigns/thesis_initial/manifest.tsv
expected_commit=2871b8abc9c28917c2049243d3483d6dada4276a

line_number=$((SLURM_ARRAY_TASK_ID + 2))
line=$(sed -n "${line_number}p" "${manifest}")
if [[ -z "${line}" ]]; then
    echo "no manifest row for array index ${SLURM_ARRAY_TASK_ID}" >&2
    exit 1
fi
IFS=$'\t' read -r run _input physical_nt nx ny seed <<< "${line}"
run_dir="${data_root}/${run}"
input="${run_dir}/input.yaml"

if [[ ! -f "${input}" ]]; then
    echo "missing input ${input}" >&2
    exit 1
fi
if compgen -G "${run_dir}/*.out" >/dev/null || \
   compgen -G "${run_dir}/*.klft" >/dev/null || \
   [[ -e "${run_dir}/run_metadata.txt" ]]; then
    echo "refusing to append to existing output in ${run_dir}" >&2
    exit 1
fi

actual_commit=$(git -C "${source_dir}" rev-parse HEAD)
if [[ "${actual_commit}" != "${expected_commit}" ]]; then
    echo "KLFT commit ${actual_commit} does not match ${expected_commit}" >&2
    exit 1
fi

{
    printf 'job_id %s\n' "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
    printf 'node %s\n' "${SLURMD_NODENAME}"
    printf 'cuda_visible_devices %s\n' "${CUDA_VISIBLE_DEVICES:-unset}"
    printf 'physical_Nt %s\nNs %s\nbeta 12.15266\nseed %s\n' \
        "${physical_nt}" "${nx}" "${seed}"
    printf 'klft_commit %s\n' "${actual_commit}"
    printf 'heatbath_sha256 '
    sha256sum "${build}/binaries/heatbath" | awk '{print $1}'
    printf 'input_sha256 '
    sha256sum "${input}" | awk '{print $1}'
} > "${run_dir}/run_metadata.txt"

cd "${run_dir}"
/usr/bin/time -v "${build}/binaries/heatbath" -f input.yaml

expected_measurements=$(awk '$1 == "nSweep:" {print $2}' input.yaml)
measurements=$(awk '!/^#/ {count++} END {print count+0}' holonomy_summary.out)
if [[ "${measurements}" -ne "${expected_measurements}" ]]; then
    echo "expected ${expected_measurements} measurements, found ${measurements}" >&2
    exit 1
fi
awk '!/^#/ && $2 == 0 && $3 != 1 {bad++} END {exit bad != 0}' \
    holonomy_correlator.out
awk -v height="${physical_nt}" -v rmax="$((nx / 2))" \
    'FNR == NR {if ($0 !~ /^#/) cg[$1 SUBSEP $2]=$3; next}
     $0 !~ /^#/ && $3 == height && $2 <= rmax {
         delta=$4-cg[$1 SUBSEP $2]; if (delta < 0) delta=-delta;
         if (delta > 5e-11) bad++
     }
     END {exit bad != 0}' \
    holonomy_correlator.out boundary_wilson.out
