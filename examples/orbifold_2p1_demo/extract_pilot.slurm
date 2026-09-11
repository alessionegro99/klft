#!/bin/bash -l
#SBATCH --job-name=orb-2p1-potential
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/qbigwork2/negro/orbifold/logs/orb-2p1-potential-%j.out

set -euo pipefail
: "${SLURM_JOB_ID:?Submit through Slurm}"
root=/qbigwork2/negro/orbifold
base="${root}/analysis/orbifold_2p1_demo_20260910/measured_pilot"
output="${base}/potential"
campaign="${root}/campaigns/orbifold_2p1_demo_20260910"
source_dir="${root}/src/klft-1104e1a"
environment="${root}/campaigns/large_mass_validation_20260831/analysis"
uv=/qbigwork2/negro/dirichlet/bin/uv
export UV_CACHE_DIR=/qbigwork2/negro/dirichlet/.uv-cache
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${source_dir}/analysis"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
[[ -f "${base}/COMPLETE" && ! -e "${output}" ]]
[[ "$(git -C "${source_dir}" rev-parse HEAD)" == 1104e1af3fa39838d1ae6fdb9c2620f7b423b134 ]]
[[ -z "$(git -C "${source_dir}" status --porcelain --ignore-submodules=none)" ]]
[[ "$(sha256sum "${campaign}/orbifold_reblock.py" | awk '{print $1}')" == 493a9c0095b4cff28d5e9eca47273aad1bc6f6332b0ea9b0e551394b5e1506f7 ]]
[[ "$(sha256sum "${environment}/uv.lock" | awk '{print $1}')" == c8673a6d8ad34f824e539738790953cc4e577045cee75d2dcfb579ccc45be3be ]]
# Verify every immutable statistics archive and the original analysis sources.
awk '$1 ~ /^[0-9a-f]+$/ && ($2 ~ /wilson_stats/ || $2 ~ /src\/klft/ || $2 ~ /uv.lock$/) {print}' "${base}/analysis_metadata.txt" | sha256sum -c -
mkdir "${output}"
run_python() {
  "${uv}" run --offline --frozen --project "${environment}" python "$@"
}
fit_windows() {
  local archive=$1 table=$2 lo=$3 hi=$4
  local args=()
  for r in 1 2 3 4 5 6 7; do
    args+=(--plateau "${r}:${lo}-${hi}")
  done
  run_python "${source_dir}/analysis/orbifold_static_potential.py" \
    "${archive}" --output "${table}" --a-s 0.2 --a-t 0.2 "${args[@]}"
}
run_python "${campaign}/orbifold_reblock.py" --self-test
for factor in 2 4; do
  run_python "${campaign}/orbifold_reblock.py" "${base}/discard0/wilson_stats.npz" \
    --output "${output}/blocks$((8 * factor)).npz" --factor "${factor}" \
    --samples 10000 --seed 26091107
done
# Common late-time window; R>=8 is unresolved under these conservative checks.
fit_windows "${output}/blocks32.npz" "${output}/static_potential.tsv" 7 9
fit_windows "${output}/blocks16.npz" "${output}/blocks16_potential.tsv" 7 9
fit_windows "${output}/blocks32.npz" "${output}/window6_9.tsv" 6 9
fit_windows "${output}/blocks32.npz" "${output}/window7_10.tsv" 7 10
fit_windows "${output}/blocks32.npz" "${output}/window8_10.tsv" 8 10
fit_windows "${base}/discard75/wilson_stats.npz" "${output}/discard75_potential.tsv" 7 9
fit_windows "${base}/discard150/wilson_stats.npz" "${output}/discard150_potential.tsv" 7 9
{
  date -u
  printf 'job_id %s\nnode %s\n' "${SLURM_JOB_ID}" "${SLURMD_NODENAME}"
  printf 'theory periodic_SU3_2p1D_full_unfixed_orbifold_HMC_32cubed\n'
  printf 'action a_s=0.2,a_t=0.2,g=1,m=40,u1_mass=40,gamma=0\n'
  printf 'simulation_commit 1104e1af3fa39838d1ae6fdb9c2620f7b423b134\n'
  printf 'main_estimator complete_covariance_GLS_T7_to_T9_R1_to_R7\n'
  printf 'main_blocks 32_vectors_320_trajectories_9_blocks_per_chain_3_chains\n'
  printf 'uncertainty vector_hierarchical_bootstrap_10000_seed26091107\n'
  printf 'limitations modest_statistics_no_continuum_volume_or_mass_extrapolation\n'
  sha256sum "$0" "${campaign}/orbifold_reblock.py" \
    "${source_dir}/analysis/orbifold_static_potential.py" \
    "${source_dir}/analysis/orbifold_hmc_stats.py" "${environment}/uv.lock" \
    "${base}/analysis_metadata.txt"
  for file in "${output}"/*.npz "${output}"/*.tsv "${output}"/*.summary.txt; do
    sha256sum "${file}"
  done
} > "${output}/potential_metadata.txt"
touch "${output}/COMPLETE"

