#!/bin/bash -l
#SBATCH --job-name=mgnt-conf-a
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=/qbigwork2/negro/dirichlet/logs/mgnt-conf-a-%j.out

set -euo pipefail
exec /qbigwork2/negro/dirichlet/project/data/mg_nt_fixed/pilot/qbig/analyze_confirmation.sh analyse
