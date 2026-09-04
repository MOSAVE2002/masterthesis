#!/bin/bash
#SBATCH --job-name=fjsp_queue
#SBATCH --account=nhul21847
#SBATCH --partition=mpp.share
#SBATCH --mail-user=moritz.sarstedt@stud.uni-hannover.de
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --ntasks=1
#SBATCH --mem=1G
#SBATCH --time=00:15:00
#SBATCH --output=cluster_files/out/queue.out
#SBATCH --error=cluster_files/out/queue.err

# =============================================================================
# QUEUE SCRIPT - Submits multiple jobs to the SLURM scheduler
# =============================================================================
# This script submits one job that runs the complete FJSP pipeline in main.py.
# Modify the parameters below to adjust resource allocation for your jobs.
# Navigate into the project directory and run this script via sbatch queue.sh
# =============================================================================

# Default values (can be overridden via environment variables)
export walltime=00:30:00   # maximum time for the test job
export memory=16G          # memory per job
export partition=gpu       # cluster partition

# Get the directory of this script (use SLURM_SUBMIT_DIR when running in SLURM)
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}/cluster_files"
PROJECT_PATH=$BIGWORK/FJSP_Simulation

echo "Initializing the Virtual Environment and installing dependencies (this may take a moment)..."

module load GCCcore/.14.2.0  # Required dependency for Python and uv
module load Python/3.13.1    # Python version required by this project
module load uv/0.9.22        # Python package manager for virtual environments

uv sync

mkdir -p "$PROJECT_PATH/logs/Gurobi" "$PROJECT_PATH/logs/out" "$PROJECT_PATH/logs/error"

echo "Submitting the FJSP test job..."

sbatch --export=ALL \
       --job-name="fjsp_test" \
       --time="$walltime" \
       --mem="$memory" \
       --cpus-per-task=4 \
       --partition="$partition" \
       --gres=gpu:a100:1 \
       --mail-user=moritz.sarstedt@stud.uni-hannover.de \
       --mail-type=BEGIN,END,FAIL \
       --output="$PROJECT_PATH/logs/out/fjsp_%j.out" \
       --error="$PROJECT_PATH/logs/error/fjsp_%j.err" \
       "$SCRIPT_DIR/job.sh"

echo "Job submitted."
