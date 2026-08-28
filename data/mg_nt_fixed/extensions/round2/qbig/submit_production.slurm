#!/bin/bash -l
#SBATCH --job-name=mgnt-ext2-prod
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:pascal:1
#SBATCH --mem=5G
#SBATCH --time=02:00:00
#SBATCH --array=0-3
#SBATCH --output=/qbigwork2/negro/dirichlet/logs/mgnt-ext2-prod-%A_%a.out

set -euo pipefail
source /hadron/negro/.env/klft_env.sh
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OMP_PROC_BIND=close
export OMP_PLACES=cores
exec bash /qbigwork2/negro/dirichlet/campaigns/mg_nt_fixed/extensions/round2/qbig/run_production.sh
