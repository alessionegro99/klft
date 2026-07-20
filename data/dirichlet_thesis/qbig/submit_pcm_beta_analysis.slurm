#!/bin/bash -l
#SBATCH --job-name=pcm-beta-analysis
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --array=0-2
#SBATCH --output=/qbigwork2/negro/dirichlet/logs/pcm-beta-analysis-%A_%a.out

set -euo pipefail

mode=${MODE:-diagnose}
beta_tags=(1p9 2p0 2p1)
beta_tag=${beta_tags[SLURM_ARRAY_TASK_ID]}

/qbigwork2/negro/dirichlet/project/data/dirichlet_thesis/qbig/analyze_pcm_beta_pilot.sh \
    "${mode}" "${beta_tag}"
