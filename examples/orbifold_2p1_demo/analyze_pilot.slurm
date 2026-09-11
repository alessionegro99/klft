#!/bin/bash -l
#SBATCH --job-name=orb-2p1-analysis
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/qbigwork2/negro/orbifold/logs/orb-2p1-analysis-%j.out

# Submit afterok on the measured-pilot array. This analyzes, but does not
# automatically accept, potential data. Raw histories remain on qbig.
set -euo pipefail
: "${SLURM_JOB_ID:?Submit through Slurm}"
root=/qbigwork2/negro/orbifold
data_root="${root}/data/orbifold_2p1_demo_20260910/measured_pilot"
output="${root}/analysis/orbifold_2p1_demo_20260910/measured_pilot"
source_dir="${root}/src/klft-1104e1a"
environment="${root}/campaigns/large_mass_validation_20260831/analysis"
uv=/qbigwork2/negro/dirichlet/bin/uv
export UV_CACHE_DIR=/qbigwork2/negro/dirichlet/.uv-cache
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
expected_commit=1104e1af3fa39838d1ae6fdb9c2620f7b423b134
expected_stats_sha=5d54f09ee4be687fabe0249e964bee87b2028142a692727e36196cf9fd7322e6
expected_potential_sha=8e4f74f653cc2a24379eec9e824c99d4fd0b0c7dc66d407534eb4ef38f49696e
expected_tau_sha=99f154e742eeffff8f1b9b44cc53899cae2eff7713bb829f241e04fdde850823
expected_lock_sha=c8673a6d8ad34f824e539738790953cc4e577045cee75d2dcfb579ccc45be3be

[[ "$(git -C "${source_dir}" rev-parse HEAD)" == "${expected_commit}" ]]
[[ -z "$(git -C "${source_dir}" status --porcelain --ignore-submodules=none)" ]]
[[ "$(sha256sum "${source_dir}/analysis/orbifold_hmc_stats.py" | awk '{print $1}')" == "${expected_stats_sha}" ]]
[[ "$(sha256sum "${source_dir}/analysis/orbifold_static_potential.py" | awk '{print $1}')" == "${expected_potential_sha}" ]]
[[ "$(sha256sum "${source_dir}/analysis/partition_benchmark.py" | awk '{print $1}')" == "${expected_tau_sha}" ]]
[[ "$(sha256sum "${environment}/uv.lock" | awk '{print $1}')" == "${expected_lock_sha}" ]]
[[ ! -e "${output}" ]]
for chain_number in 01 02 03; do
  chain="${data_root}/chain${chain_number}"
  [[ -f "${chain}/COMPLETE" ]]
  grep -Fx "klft_commit ${expected_commit}" "${chain}/run_metadata.txt"
  [[ "$(awk '!/^#/ && NF {n++} END {print n+0}' "${chain}/hmc.out")" == 3500 ]]
  [[ "$(awk '!/^#/ && NF {n++} END {print n+0}' "${chain}/diagnostic.out")" == 350 ]]
  [[ "$(awk '!/^#/ && NF {n++} END {print n+0}' "${chain}/wilson.out")" == 76800 ]]
  (cd "${chain}" && tail -5 run_metadata.txt | sha256sum -c -)
done
mkdir -p "${output}"
"${uv}" run --offline --frozen --project "${environment}" \
  python "${source_dir}/analysis/orbifold_hmc_stats.py" --self-test
"${uv}" run --offline --frozen --project "${environment}" \
  python "${source_dir}/analysis/orbifold_static_potential.py" --self-test
# Compare no additional cut with quarter/half cuts. Never concatenate chains
# before estimating autocorrelation or resampling complete loop vectors.
for discard in 0 75 150; do
  table="${output}/discard${discard}/wilson_stats.tsv"
  "${uv}" run --offline --frozen --project "${environment}" \
    python "${source_dir}/analysis/orbifold_hmc_stats.py" \
    "${data_root}" --output "${table}" --discard-vectors "${discard}" \
    --bootstrap-samples 10000 --seed 20260911
  [[ "$(grep -vc '^#' "${table}")" == 256 ]]
done
{
  date -u
  printf 'job_id %s\nnode %s\n' "${SLURM_JOB_ID}" "${SLURMD_NODENAME}"
  printf 'theory periodic_SU3_2p1D_full_unfixed_orbifold\n'
  printf 'action a_s=0.2,a_t=0.2,g=1,m=40,u1_mass=40,gamma=0,ma=8\n'
  printf 'lattice 32,32,32\nspace 0,1\ntime 2\n'
  printf 'loops normalized_ReTr_over_Nc_polar_spatial_explicit_temporal_unsmeared\n'
  printf 'simulation_and_analysis_commit %s\n' "${expected_commit}"
  printf 'statistics chainwise_Wolff_tau;common_vector_block;hierarchical_vector_bootstrap\n'
  printf 'bootstrap_samples 10000\nbootstrap_seed 20260911\n'
  "${uv}" --version
  "${uv}" run --offline --frozen --project "${environment}" \
    python -c 'import sys,numpy; print(sys.version); print("numpy",numpy.__version__)'
  sha256sum "$0" "${source_dir}/analysis/orbifold_hmc_stats.py" \
    "${source_dir}/analysis/orbifold_static_potential.py" \
    "${source_dir}/analysis/partition_benchmark.py" "${environment}/uv.lock"
  for chain_number in 01 02 03; do
    sha256sum "${data_root}/chain${chain_number}/run_metadata.txt"
  done
  for discard in 0 75 150; do
    sha256sum "${output}/discard${discard}/wilson_stats.tsv" \
      "${output}/discard${discard}/wilson_stats.npz" \
      "${output}/discard${discard}/wilson_stats.summary.txt"
  done
} > "${output}/analysis_metadata.txt"
touch "${output}/COMPLETE"

