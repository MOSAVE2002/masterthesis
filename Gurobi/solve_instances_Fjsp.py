"""

Usage:
    uv run Gurobi/solve_instances_Fjsp.py instance_name=i3_k3_1 TimeLimit=30 Threads=1

Arguments:
    instance_name: Name of the FJSP instance file (without .fjsp extension)
    TimeLimit: Maximum solver time in seconds (Gurobi parameter)
    Threads: Number of CPU threads to use (Gurobi parameter)
    Any other Gurobi parameter can be passed as key=value

Example:
    $ uv run Gurobi/solve_instances_Fjsp.py instance_name=i3_k3_1 TimeLimit=60 MIPGap=0.01
"""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))


import pickle
import gurobipy as gp
from gurobipy import GRB
from generator.instance_generator import FJSPData
from Gurobi.build_fjsp import build_fjsp, write_solution_file


def solveModel(**kwargs):
    """
    Main function to solve a FJSP instance.

    This function:
    1. Loads a FJSP instance from a pickle file
    2. Creates a Gurobi optimization model
    3. Applies any Gurobi parameters passed as kwargs
    4. Builds and solves the FJSP model
    5. Writes the solution to a file

    Args:
        **kwargs: Keyword arguments including:
            - instance_name (str): Required. Name of the instance file (without extension)
            - Any valid Gurobi parameter (e.g., TimeLimit, Threads, MIPGap)

    Returns:
        None

    Raises:
        FileNotFoundError: If the instance file does not exist
        gurobipy.GurobiError: If there are issues with the Gurobi license or model
    """
    known_kwargs = {"instance_name"}

    # Validate required arguments
    if "instance_name" not in kwargs:
        print("Error: Please provide an instance name using the 'instance_name' keyword argument.")
        print("Usage: uv run Gurobi/solve_instances_Fjsp.py instance_name=<name> [TimeLimit=<seconds>] [Threads=<n>]")
        return

    instance_name = kwargs.get("instance_name")

    # Load the FJSP instance from a pickle file
    path = ROOT_DIR / "data" / "fjsp_instances" / f"{instance_name}.fjsp"

    if not path.exists():
        print(f"Error: Instance file not found at '{path}'")
        print("Available instances:")
        instances_dir = ROOT_DIR / "data" / "fjsp_instances"
        if instances_dir.exists():
            for f in instances_dir.iterdir():
                if f.suffix == ".fjsp":
                    print(f"  - {f.stem}")
        return

    with open(path, "rb") as f:
        fjsp_instance: FJSPData = pickle.load(f)

    print(f"Loaded instance: {fjsp_instance.instance_name}")
    print(f"  Number of Jobs: {fjsp_instance.num_jobs}")
    print(f"  Number of Machines: {fjsp_instance.num_machines}")

    # Create the Gurobi model
    model = gp.Model("FJSP Model")

    # Apply Gurobi parameters from kwargs
    # Es werden alle bekannten Parameter an Gurobi weitergegeben
    #TODO will ich das wirklich so machen? muss ich wahrscheinlich
    for key, value in kwargs.items():
        if hasattr(model.Params, key):
            known_kwargs.add(key)
            setattr(model.Params, key, value)
            print(f"  Set Gurobi parameter {key} = {value}")

    # Warn about unknown arguments
    for key in kwargs.keys():
        if key not in known_kwargs:
            print(f"Warning: Unknown keyword argument '{key}' provided. It will be ignored.")

    #

    # Build the FJSP model (add variables, objective, and constraints)
    model, variables = build_fjsp(model, fjsp_instance)

    # Optimize the model
    print("\nStarting optimization...")
    model.optimize()

    solution_dir = ROOT_DIR / "data" / "fjsp_solutions"
    solution_dir.mkdir(parents=True, exist_ok=True)
    solution_path = solution_dir / f"solution_{instance_name}.txt"

    # Check solution status and write output
    if model.Status == GRB.OPTIMAL:
        print(f"\nOptimal solution found! Objective value: {model.ObjVal:.2f}")
        write_solution_file(model, variables, fjsp_instance, solution_path)
        print(f"Solution written to: {solution_path}")
    elif model.Status == GRB.TIME_LIMIT and model.SolCount > 0:
        print(f"\nTime limit reached. Best solution found: {model.ObjVal:.2f}")
        write_solution_file(model, variables, fjsp_instance, solution_path)
        print(f"Solution written to: {solution_path}")
    elif model.Status == GRB.INFEASIBLE:
        print("\nError: Model is infeasible. No solution exists.")
    elif model.Status == GRB.UNBOUNDED:
        print("\nError: Model is unbounded.")
    else:
        print(f"\nOptimization ended with status {model.Status}. No solution written.")

if __name__ == '__main__':
    pass
    #TODO Kwarg code von Timo habe ich hier gelöscht, vielleicht wieder in solve_ins einbauen?