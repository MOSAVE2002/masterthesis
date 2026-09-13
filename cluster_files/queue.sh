#!/bin/bash
#SBATCH --job-name=fjsp_queue
#SBATCH --account=nhul21847
#SBATCH --partition=mpp.share
#SBATCH --mail-user=moritz.sarstedt@stud.uni-hannover.de
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=cluster_files/out/queue.out
#SBATCH --error=cluster_files/out/queue.err

set -e

# =============================================================================
# QUEUE SCRIPT - Submits multiple jobs to the SLURM scheduler
# =============================================================================
# This script prepares the benchmark instances and submits one job per case.
# Modify the parameters below to adjust resource allocation for your jobs.
# Navigate into the project directory and run this script via sbatch queue.sh
# =============================================================================

# Default values (can be overridden via environment variables)
export walltime=48:00:00   # maximum time for three benchmark cases
export memory=16G          # memory per job
export partition=mpp.share # cluster partition
export job_array=0-39%8    # 40 jobs with 3 cases each, at most 8 at the same time

# Get the directory of this script (use SLURM_SUBMIT_DIR when running in SLURM)
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}/cluster_files"
PROJECT_PATH=$SLURM_SUBMIT_DIR

echo "Initializing the Virtual Environment and installing dependencies (this may take a moment)..."

module load GCCcore/.14.2.0  # Required dependency for Python and uv
module load Python/3.13.1    # Python version required by this project
module load uv/0.9.22        # Python package manager for virtual environments

uv sync

# Use the cluster license while preparing the calibrated due dates
export GRB_LICENSE_FILE=/sw/apps/software/arch/Compiler/GCCcore/14.3.0/Gurobi/13.0.3/gurobi.lic

mkdir -p "$PROJECT_PATH/logs/Gurobi" "$PROJECT_PATH/logs/out" "$PROJECT_PATH/logs/error"

echo "Preparing the benchmark instances..."
uv run main.py --prepare-solve-instances

echo "Submitting the FJSP benchmark jobs..."

sbatch --export=ALL \
       --job-name="fjsp_benchmark" \
       --array="$job_array" \
       --time="$walltime" \
       --mem="$memory" \
       --cpus-per-task=4 \
       --partition="$partition" \
       --mail-user=moritz.sarstedt@stud.uni-hannover.de \
       --mail-type=BEGIN,END,FAIL \
       --output="$PROJECT_PATH/logs/out/fjsp_%A_%a.out" \
       --error="$PROJECT_PATH/logs/error/fjsp_%A_%a.err" \
       "$SCRIPT_DIR/job.sh"

echo "Job submitted."
