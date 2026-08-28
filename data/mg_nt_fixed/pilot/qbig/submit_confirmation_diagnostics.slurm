#!/bin/bash -l
#SBATCH --job-name=mgnt-conf-d
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --output=/qbigwork2/negro/dirichlet/logs/mgnt-conf-d-%j.out

set -euo pipefail
exec /qbigwork2/negro/dirichlet/project/data/mg_nt_fixed/pilot/qbig/analyze_confirmation.sh diagnose
