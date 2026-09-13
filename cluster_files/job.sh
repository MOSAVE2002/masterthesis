#!/bin/bash -l
#SBATCH --ntasks=1

set -e

# =============================================================================
# JOB SCRIPT - Runs on the compute node
# =============================================================================
# This script is executed on each compute node. It:
# 1. Sets up the environment (modules, working directory)
# 2. Runs the FJSP pipeline through main.py
# =============================================================================

# Show which computer the job ran on (useful for debugging)
echo "Job ran on: $(hostname)"
echo "Job started at: $(date)"

# queue.sh submits the job from the project directory
project_path=$SLURM_SUBMIT_DIR

# Change to the project directory
cd "${project_path}"

# Echo current directory for verification
echo "Current directory: $(pwd)"

# Load Python and uv modules
# Note: Use "module spider <package>" to check dependencies
module load GCCcore/.14.2.0  # Required dependency for Python and uv
module load Python/3.13.1    # Python version required by this project
module load uv/0.9.22        # Python package manager for virtual environments

# Use the cluster license with the gurobipy version installed by uv sync
export GRB_LICENSE_FILE=/sw/apps/software/arch/Compiler/GCCcore/14.3.0/Gurobi/13.0.3/gurobi.lic

# =============================================================================
# Run the FJSP pipeline
# =============================================================================
# main.py reads the workflow settings from config.json.
# =============================================================================
echo ""
first_case=$((SLURM_ARRAY_TASK_ID * 3))
last_case=$((first_case + 2))

for solve_plan_index in $(seq "$first_case" "$last_case"); do
    echo "Running benchmark case $solve_plan_index"
    echo ""
    uv run main.py --workflow solve --solve-plan-index "$solve_plan_index"
done

echo ""
echo "Job finished at: $(date)"
