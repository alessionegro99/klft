#!/bin/bash -l
#SBATCH --job-name=mgnt-final-tuning
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/qbigwork2/negro/dirichlet/logs/mgnt-final-tuning-%j.out

set -euo pipefail
exec bash /qbigwork2/negro/dirichlet/campaigns/mg_nt_fixed/final_checks/qbig/run_final_tuning.sh
