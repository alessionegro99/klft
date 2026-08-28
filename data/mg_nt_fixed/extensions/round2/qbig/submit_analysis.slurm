#!/bin/bash -l
#SBATCH --job-name=mgnt-ext2-analysis
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=/qbigwork2/negro/dirichlet/logs/mgnt-ext2-analysis-%j.out

set -euo pipefail
exec bash /qbigwork2/negro/dirichlet/campaigns/mg_nt_fixed/extensions/round2/qbig/analyze_extension.sh analyse
