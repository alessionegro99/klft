#!/bin/bash -l
#SBATCH --job-name=bntf-fv-analysis
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=08:00:00
#SBATCH --output=/qbigwork2/negro/dirichlet/logs/bntf-fv-analysis-%j.out

set -euo pipefail
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
exec /qbigwork2/negro/dirichlet/project/data/beta_nt_fixed/beta_over_nt_1p8/finite_volume_nt2/qbig/analyze.sh \
    analyse
