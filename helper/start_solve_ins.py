import sys
import pickle
import json
import csv
import hashlib
import random
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

import gurobipy as gp
from gurobipy import GRB

from generator.instance_generator import FJSPData
from Gurobi.build_fjsp import build_fjsp, write_solution_file


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _sample_seed(instance_name: str, sample_idx: int, base_seed) -> int:
    seed_source = f"{instance_name}:{sample_idx}:{base_seed}" # zeichenkette
    digest = hashlib.sha256(seed_source.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") # Aus den ersten 8 Bytes des hashs wird eine Ganzzahl erzeugt


def _generate_random_fixed_y(instance, instance_name: str, sample_idx: int, base_seed):
    rng = random.Random(_sample_seed(instance_name, sample_idx, base_seed))
    return rng, {
        operation: rng.choice(instance.eligible_machines[operation])
        for operation in instance.real_operations
    }


def _apply_fixed_y_values(model, variables, instance, fixed_y_assignment, rng, fix_ratio = None):
    

    if fix_ratio is None:
        fix_ratio = rng.uniform(0.5, 0.8)
   

    Y = variables["Y"]
    ops = list(fixed_y_assignment.keys())
    k = round(len(ops) * fix_ratio)
    to_fix = set(rng.sample(ops, k))

    for operation, chosen_machine in fixed_y_assignment.items():

        if chosen_machine not in instance.eligible_machines[operation]:
            raise ValueError(
                f"Machine {chosen_machine} is not eligible for operation {operation}."
            )

        if operation not in to_fix:
            continue

        for machine in instance.eligible_machines[operation]:
            fixed_value = 1.0 if machine == chosen_machine else 0.0
            Y[operation, machine].lb = fixed_value
            Y[operation, machine].ub = fixed_value

    model.update()


def _write_fixed_y_assignment(instance_name: str, sample_idx: int, base_seed, fixed_y_assignment):
    output_dir = ROOT_DIR / "data" / "fixed_y_assignments"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{instance_name}__yfix_{sample_idx:03d}.json"

    payload = {
        "instance_name": instance_name,
        "sample_idx": sample_idx,
        "base_seed": base_seed,
        "assignment": [
            {"operation": operation, "machine": fixed_y_assignment[operation]}
            for operation in sorted(fixed_y_assignment)
        ],
    }

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)

    return output_path


def _append_fixed_y_result_csv(
    instance,
    instance_name: str,
    sample_idx: int,
    fixed_y_assignment,
    model,
):
    output_dir = ROOT_DIR / "data" / "fixed_y_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{instance_name}.csv"

    operations = sorted(instance.real_operations)
    fieldnames = [
        "sample_idx",
        "makespan",
    ] + [f"y_op_{operation}" for operation in operations]

    row = {
        "sample_idx": sample_idx,
        "makespan": int(model.ObjVal) if model.SolCount > 0 else "",
    }

    for operation in operations:
        row[f"y_op_{operation}"] = fixed_y_assignment[operation]

    file_exists = output_path.exists()
    with open(output_path, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    return output_path


def solveModel(**kwargs):
    """Load a pickled FJSP instance, solve it with Gurobi, and write a solution file."""
    known_kwargs = {"instance_name"}

    if "instance_name" not in kwargs:
        print("Error: Please provide an instance name using the 'instance_name' keyword argument.")
        print("Usage: solve_instances_with_solver(instance_name=<name>, solver='gurobi', TimeLimit=<seconds>)")
        return

    instance_name = kwargs.get("instance_name")
    create_fixed_y = _as_bool(kwargs.pop("create_fixed_y", False))
    sample_idx = int(kwargs.pop("sample_idx", 0))
    random_seed = kwargs.pop("random_seed", 0)
    save_assignments = _as_bool(kwargs.pop("save_assignments", False))
    path = ROOT_DIR / "data" / "fjsp_instances" / f"{instance_name}.fjsp"

    if not path.exists():
        print(f"Error: Instance file not found at '{path}'")
        print("Available instances:")
        instances_dir = ROOT_DIR / "data" / "fjsp_instances"
        if instances_dir.exists():
            for file in instances_dir.iterdir():
                if file.suffix == ".fjsp":
                    print(f"  - {file.stem}")
        else:
            print("Dir does not exist")
        return

    with open(path, "rb") as file:
        fjsp_instance: FJSPData = pickle.load(file)

    print(f"Loaded instance: {fjsp_instance.instance_name}")
    print(f"Number of Jobs: {fjsp_instance.num_jobs}")
    print(f"Number of Machines: {fjsp_instance.num_machines}")

    model = gp.Model("FJSP Model")

    for key, value in kwargs.items():
        if hasattr(model.Params, key):
            known_kwargs.add(key)
            setattr(model.Params, key, value)
            print(f"  Set Gurobi parameter {key} = {value}")

    for key in kwargs.keys():
        if key not in known_kwargs:
            print(f"Warning: Unknown keyword argument '{key}' provided. It will be ignored.")

    model, variables = build_fjsp(model, fjsp_instance)

    fixed_y_assignment = None
    if create_fixed_y:
        rng, fixed_y_assignment = _generate_random_fixed_y(
            fjsp_instance,
            instance_name=instance_name,
            sample_idx=sample_idx,
            base_seed=random_seed,
        )
        _apply_fixed_y_values(model, variables, fjsp_instance, fixed_y_assignment, rng)
        print(
            "  Using fixed random Y assignment "
            f"(sample {sample_idx + 1}, base seed {random_seed})"
        )

        if save_assignments:
            assignment_path = _write_fixed_y_assignment(
                instance_name=instance_name,
                sample_idx=sample_idx,
                base_seed=random_seed,
                fixed_y_assignment=fixed_y_assignment,
            )
            print(f"  Saved fixed Y assignment to: {assignment_path}")

    print("\nStarting optimization...")
    model.optimize()

    solution_dir = ROOT_DIR / "data" / "fjsp_solutions"
    solution_dir.mkdir(parents=True, exist_ok=True)
    solution_suffix = ""
    if create_fixed_y:
        solution_suffix = f"__yfix_{sample_idx:03d}"
    solution_path = solution_dir / f"solution_{instance_name}{solution_suffix}.txt"

    if create_fixed_y:
        pass
    else:
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

    if create_fixed_y and fixed_y_assignment is not None:
        csv_path = _append_fixed_y_result_csv(
            instance=fjsp_instance,
            instance_name=instance_name,
            sample_idx=sample_idx,
            fixed_y_assignment=fixed_y_assignment,
            model=model,
        )
        print(f"Fixed-Y result appended to: {csv_path}")
    model.dispose()

def solve_instances_with_solver(**kwargs):
    solver_kwargs = dict(kwargs)

    instance_name = solver_kwargs.pop("instance_name", None)
    num_jobs      = solver_kwargs.pop("num_jobs",     20)
    num_machines  = solver_kwargs.pop("num_machines", 10)
    instance_nb   = solver_kwargs.pop("instance_nb",  1)
    solver        = solver_kwargs.pop("solver", "").lower()
    create_fixed_y = _as_bool(solver_kwargs.pop("create_fixed_y", False))
    amount_of_samples_per_instance = int(
        solver_kwargs.pop("amount_of_samples_per_instance", 1)
    )
    random_seed = solver_kwargs.pop("random_seed", 0)
    save_assignments = _as_bool(
        solver_kwargs.pop("save_assignments", False)
    )

    if instance_name is None:
        instance_name = f"i{num_jobs}_k{num_machines}_{instance_nb}"

    if solver == "gurobi":
        if create_fixed_y:
            for sample_idx in range(amount_of_samples_per_instance):
                solveModel(
                    instance_name=instance_name,
                    create_fixed_y=True,
                    sample_idx=sample_idx,
                    random_seed=random_seed,
                    save_assignments=save_assignments,
                    **solver_kwargs,
                )
        else:
            solveModel(instance_name=instance_name, **solver_kwargs)

    
    else:
        print(f"Error: Unknown solver '{solver}'. Use 'gurobi'")
