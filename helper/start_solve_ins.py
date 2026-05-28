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

from pyscipopt import Model as SCIPModel

from generator.instance_generator import FJSPData
from Gurobi.build_fjsp import build_fjsp as build_fjsp_gurobi
from Gurobi.build_fjsp import write_solution_file as write_solution_file_gurobi
from SCIP.build_fjsp import build_fjsp as build_fjsp_scip
from SCIP.build_fjsp import write_solution_file as write_solution_file_scip


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


def _apply_fixed_y_values(model, variables, instance, fixed_y_assignment, rng, fix_ratio=None):
    

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
            variable = Y[operation, machine]
            if hasattr(variable, "lb") and hasattr(variable, "ub"):
                variable.lb = fixed_value
                variable.ub = fixed_value
            else:
                model.chgVarLb(variable, fixed_value)
                model.chgVarUb(variable, fixed_value)

    if hasattr(model, "update"):
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
    objective_value,
    has_solution,
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
        "makespan": int(objective_value) if has_solution else "",
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


def _set_scip_param(model, key, value):
    candidates = [key]
    if "_" in key:
        candidates.append(key.replace("_", "/"))

    for candidate in candidates:
        try:
            model.setParam(candidate, value)
            return candidate
        except Exception:
            continue

    return None


def _solve_gurobi_model(fjsp_instance, instance_name: str, create_fixed_y: bool, sample_idx: int, random_seed, save_assignments: bool, solver_kwargs):
    if gp is None:
        raise ImportError("gurobipy is not installed, but solver='gurobi' was requested.")

    model = gp.Model("FJSP Model")

    for key, value in solver_kwargs.items():
        if hasattr(model.Params, key):
            setattr(model.Params, key, value)
            print(f"  Set Gurobi parameter {key} = {value}")
        else:
            print(f"Warning: Unknown keyword argument '{key}' provided. It will be ignored.")

    model, variables = build_fjsp_gurobi(model, fjsp_instance)

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
    solution_suffix = f"__yfix_{sample_idx:03d}" if create_fixed_y else ""
    solution_path = solution_dir / f"solution_{instance_name}_gurobi{solution_suffix}.txt"

    if not create_fixed_y:
        if model.Status == GRB.OPTIMAL:
            print(f"\nOptimal solution found! Objective value: {model.ObjVal:.2f}")
            write_solution_file_gurobi(model, variables, fjsp_instance, solution_path)
            print(f"Solution written to: {solution_path}")
        elif model.Status == GRB.TIME_LIMIT and model.SolCount > 0:
            print(f"\nTime limit reached. Best solution found: {model.ObjVal:.2f}")
            write_solution_file_gurobi(model, variables, fjsp_instance, solution_path)
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
            objective_value=model.ObjVal if model.SolCount > 0 else None,
            has_solution=model.SolCount > 0,
        )
        print(f"Fixed-Y result appended to: {csv_path}")
    model.dispose()


def _solve_scip_model(fjsp_instance, instance_name: str, create_fixed_y: bool, sample_idx: int, random_seed, save_assignments: bool, solver_kwargs):
    model = SCIPModel("FJSP Model")

    for key, value in solver_kwargs.items():
        actual_key = _set_scip_param(model, key, value)
        if actual_key is None:
            print(f"Warning: Unknown keyword argument '{key}' provided. It will be ignored.")
        else:
            print(f"  Set SCIP parameter {actual_key} = {value}")

    model, variables = build_fjsp_scip(model, fjsp_instance)

    if create_fixed_y:
        print("Warning: create_fixed_y is not supported for solver 'scip'. Ignoring it.")

    print("\nStarting optimization...")
    model.optimize()

    status = model.getStatus()
    has_solution = model.getNSols() > 0

    solution_dir = ROOT_DIR / "data" / "fjsp_solutions"
    solution_dir.mkdir(parents=True, exist_ok=True)
    solution_path = solution_dir / f"solution_{instance_name}_scip.txt"

    if status == "optimal":
        print(f"\nOptimal solution found! Objective value: {model.getObjVal():.2f}")
        write_solution_file_scip(model, variables, fjsp_instance, solution_path)
        print(f"Solution written to: {solution_path}")
    elif status in {"timelimit", "gaplimit", "sollimit", "bestsollimit"} and has_solution:
        print(f"\nLimit reached. Best solution found: {model.getObjVal():.2f}")
        write_solution_file_scip(model, variables, fjsp_instance, solution_path)
        print(f"Solution written to: {solution_path}")
    elif status == "infeasible":
        print("\nError: Model is infeasible. No solution exists.")
    elif status == "unbounded":
        print("\nError: Model is unbounded.")
    else:
        print(f"\nOptimization ended with status {status}. No solution written.")
    model.freeProb()


def solveModel(**kwargs):
    """Load a pickled FJSP instance, solve it with Gurobi or SCIP, and write a solution file."""
    if "instance_name" not in kwargs:
        print("Error: Please provide an instance name using the 'instance_name' keyword argument.")
        print("Usage: solve_instances_with_solver(instance_name=<name>, solver='gurobi', TimeLimit=<seconds>)")
        return

    instance_name = kwargs.pop("instance_name")
    solver = kwargs.pop("solver", "gurobi").lower()
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

    if solver == "gurobi":
        _solve_gurobi_model(
            fjsp_instance,
            instance_name=instance_name,
            create_fixed_y=create_fixed_y,
            sample_idx=sample_idx,
            random_seed=random_seed,
            save_assignments=save_assignments,
            solver_kwargs=kwargs,
        )
    elif solver == "scip":
        _solve_scip_model(
            fjsp_instance,
            instance_name=instance_name,
            create_fixed_y=create_fixed_y,
            sample_idx=sample_idx,
            random_seed=random_seed,
            save_assignments=save_assignments,
            solver_kwargs=kwargs,
        )
    else:
        print(f"Error: Unknown solver '{solver}'. Use 'gurobi' or 'scip'")

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

    if solver in {"gurobi", "scip"}:
        if create_fixed_y:
            for sample_idx in range(amount_of_samples_per_instance):
                solveModel(
                    instance_name=instance_name,
                    solver=solver,
                    create_fixed_y=True,
                    sample_idx=sample_idx,
                    random_seed=random_seed,
                    save_assignments=save_assignments,
                    **solver_kwargs,
                )
        else:
            solveModel(instance_name=instance_name, solver=solver, **solver_kwargs)

    
    else:
        print(f"Error: Unknown solver '{solver}'. Use 'gurobi' or 'scip'")
