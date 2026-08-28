#!/bin/bash -l
#SBATCH --job-name=bntf-smoke25
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:pascal:1
#SBATCH --mem=5G
#SBATCH --time=00:20:00
#SBATCH --output=/qbigwork2/negro/dirichlet/logs/bntf-smoke25-%j.out

set -euo pipefail
export BNTF_CAMPAIGN_SUFFIX=cadence25
export BNTF_VALIDATION_SUFFIX=-c25
exec bash /qbigwork2/negro/dirichlet/campaigns/beta_nt_fixed/production/qbig/submit_smoke.slurm
