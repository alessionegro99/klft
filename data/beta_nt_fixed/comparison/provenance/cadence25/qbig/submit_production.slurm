#!/bin/bash -l
#SBATCH --job-name=bntf-prod25
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:pascal:1
#SBATCH --mem=8G
#SBATCH --time=3-23:30:00
#SBATCH --array=0-39
#SBATCH --output=/qbigwork2/negro/dirichlet/logs/bntf-prod25-%A_%a.out

set -euo pipefail
export BNTF_CAMPAIGN_SUFFIX=cadence25
exec bash /qbigwork2/negro/dirichlet/campaigns/beta_nt_fixed/production/qbig/submit_production.slurm
