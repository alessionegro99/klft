#!/bin/bash -l
#SBATCH --job-name=dir-plots
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=/qbigwork2/negro/dirichlet/logs/plots-%j.out

set -euo pipefail

remote_root=/qbigwork2/negro/dirichlet
cd "${remote_root}/project"
exec "${remote_root}/bin/uv" run --frozen python scripts/plot_thesis_results.py \
    --results "${remote_root}/analysis" \
    --output "${remote_root}/analysis/thesis_initial_plots" \
    --benchmarks data/dirichlet_thesis/thesis_benchmarks.json
