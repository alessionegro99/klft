#!/bin/bash -l
#SBATCH --job-name=pcm-beta-plots
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=/qbigwork2/negro/dirichlet/logs/pcm-beta-plots-%j.out

set -euo pipefail

remote_root=/qbigwork2/negro/dirichlet
cd "${remote_root}/project"
exec "${remote_root}/bin/uv" run --frozen python scripts/plot_pcm_beta_pilot.py \
    --results "${remote_root}/analysis/pcm_beta_pilot" \
    --output "${remote_root}/analysis/pcm_beta_pilot/plots"
