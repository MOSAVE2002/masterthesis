#!/bin/bash -l
#SBATCH --ntasks=1

# =============================================================================
# JOB SCRIPT - Runs on the compute node
# =============================================================================
# This script is executed on each compute node. It:
# 1. Sets up the environment (modules, working directory)
# 2. Checks that the requested GPU is available
# 3. Runs the FJSP pipeline through main.py
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
echo "Checking whether PyTorch can use the requested GPU..."
uv run python -c 'import torch; assert torch.cuda.is_available(), "CUDA GPU is not available"; print(torch.cuda.get_device_name(0))'
echo ""
echo "Running benchmark case $SLURM_ARRAY_TASK_ID"
echo ""
uv run main.py --workflow solve --solve-plan-index "$SLURM_ARRAY_TASK_ID"

echo ""
echo "Job finished at: $(date)"
