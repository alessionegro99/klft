#!/bin/bash -l
#SBATCH --job-name=mgnt-final-smoke
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:pascal:1
#SBATCH --mem=5G
#SBATCH --time=00:10:00
#SBATCH --output=/qbigwork2/negro/dirichlet/logs/mgnt-final-smoke-%j.out

set -euo pipefail
source /hadron/negro/.env/klft_env.sh
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
exec bash /qbigwork2/negro/dirichlet/campaigns/mg_nt_fixed/final_checks/qbig/run_validation.sh smoke
