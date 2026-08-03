from pathlib import Path
import pickle
import sys
from pyscipopt import Model, exp, quicksum

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from helper.surrogate_constraint import (
    CONSTRAINT_WEIBULL,
    effective_constraint_budget,
    validate_constraint_type,
)


def _machine_parameter(instance, attr_name, machine, default):
    value = getattr(instance, attr_name, default)
    if isinstance(value, dict):
        return float(value.get(machine, default))
    if isinstance(value, (list, tuple)):
        if machine < len(value):
            return float(value[machine])
        return float(default)
    return float(value)


def _first_machine_parameter(instance, attr_names, machine, default):
    for attr_name in attr_names:
        if hasattr(instance, attr_name):
            return _machine_parameter(instance, attr_name, machine, default)
    return float(default)


def _failure_cost(instance, operation, machine, default=1.0):
    for attr_name in ("failure_costs", "failure_cost", "c_fail"):
        if not hasattr(instance, attr_name):
            continue
        value = getattr(instance, attr_name)
        if isinstance(value, dict):
            if (operation, machine) in value:
                return float(value[operation, machine])
            if operation in value and isinstance(value[operation], dict):
                if machine in value[operation]:
                    return float(value[operation][machine])
            if machine in value:
                return float(value[machine])
            continue
        return float(value)
    return float(default)


def _directed_a(A_plus, A_minus, source, target, machine):
    if source < target:
        return A_plus[source, target, machine]
    return A_minus[target, source, machine]


def _format_indexed_values(values, prefix):
    if not values:
        return ""

    ordered_items = sorted(values.items())
    unique_values = {round(float(value), 10) for _idx, value in ordered_items}
    if len(unique_values) == 1:
        return f"{float(ordered_items[0][1]):.4f}"

    return ", ".join(
        f"{prefix}{idx}={float(value):.4f}"
        for idx, value in ordered_items
    )


def _format_failure_cost_values(failure_costs):
    if not failure_costs:
        return ""

    ordered_items = sorted(failure_costs.items())
    unique_values = {round(float(value), 10) for _idx, value in ordered_items}
    if len(unique_values) == 1:
        return f"{float(ordered_items[0][1]):.4f}"

    return ", ".join(
        f"({operation},{machine})={float(value):.4f}"
        for (operation, machine), value in ordered_items
    )


def build_fjsp(
    fjsp: Model,
    instance,
    add_machine_load_lb=True,
    constraint_type=CONSTRAINT_WEIBULL,
    weibull_budget=10.0,
    enforce_constraint=True,
    safety_margin=0.0,
    safety_margin_per_operation=0.0,
    scale_weibull_budget=False,
    weibull_budget_per_operation=0.4,
):
    """
    Build the nonlinear SCIP MINLP for the reliability-aware FJSP.

    Formulas:
        S_i       = C_i - sum_k Y_i,k p_i,k
        A_i,j,k   = Y_i,k Y_j,k X_i,j,k
        A_j,i,k   = Y_i,k Y_j,k (1 - X_i,j,k)
        R_i,k     = R0_k Y_i,k + sum_j p_j,k A_j,i,k
        pi_i,k    = Y_i,k * (1 - exp(-(((R_i,k + p_i,k)/eta_k)^beta_k
                                      - (R_i,k/eta_k)^beta_k)))

    Required nonlinear instance attributes are created by FJSPData:
    - mu_fail, machine_initial_age, weibull_eta, weibull_beta, failure_cost

    The A product equations are linearized exactly for binary variables. The
    Weibull probabilities remain exact nonlinear SCIP constraints.
    """
    model = fjsp
    constraint_type = validate_constraint_type(constraint_type)
    constraint_budget = effective_constraint_budget(
        constraint_type,
        len(instance.real_operations),
        weibull_budget=weibull_budget,
        scale_weibull_budget=scale_weibull_budget,
        weibull_budget_per_operation=weibull_budget_per_operation,
    )
    if constraint_budget < 0.0:
        raise ValueError("The surrogate constraint budget must be nonnegative.")

    machines = list(range(instance.num_machines))
    if not hasattr(instance, "mu_fail"):
        instance._set_default_nonlinear_parameters()

    mu_fail = float(getattr(instance, "mu_fail", 1.0))
    initial_age = {
        machine: _first_machine_parameter(
            instance,
            ("machine_initial_age", "R0", "r0"),
            machine,
            0.0,
        )
        for machine in machines
    }
    weibull_eta = {
        machine: _first_machine_parameter(
            instance,
            ("weibull_eta", "eta"),
            machine,
            100.0,
        )
        for machine in machines
    }
    weibull_beta = {
        machine: _first_machine_parameter(
            instance,
            ("weibull_beta", "beta"),
            machine,
            2.0,
        )
        for machine in machines
    }
    for machine in machines:
        if initial_age[machine] < 0.0:
            raise ValueError("machine_initial_age must be nonnegative.")
        if weibull_eta[machine] <= 0.0:
            raise ValueError("weibull_eta must be positive.")
        if weibull_beta[machine] <= 0.0:
            raise ValueError("weibull_beta must be positive.")

    H = sum(
        max(
            instance.processing_times[operation, machine]
            for machine in instance.eligible_machines[operation]
        )
        for operation in instance.real_operations
    )

    C = {
        operation: model.addVar(vtype="C", lb=0.0, name=f"C[{operation}]")
        for operation in instance.real_operations
    }
    S = {
        operation: model.addVar(vtype="C", lb=0.0, name=f"S[{operation}]")
        for operation in instance.real_operations
    }
    C_max = model.addVar(vtype="C", lb=0.0, name="C_max")

    Y_index = [
        (operation, machine)
        for operation in instance.real_operations
        for machine in instance.eligible_machines[operation]
    ]
    Y = {
        (operation, machine): model.addVar(
            vtype="B",
            name=f"Y[{operation},{machine}]",
        )
        for operation, machine in Y_index
    }

    X_index = []
    for idx_i, operation_i in enumerate(instance.real_operations):
        for operation_j in instance.real_operations[idx_i + 1:]:
            common_machines = set(instance.eligible_machines[operation_i]) & set(
                instance.eligible_machines[operation_j]
            )
            for machine in common_machines:
                X_index.append((operation_i, operation_j, machine))

    X = {
        (operation_i, operation_j, machine): model.addVar(
            vtype="B",
            name=f"X[{operation_i},{operation_j},{machine}]",
        )
        for operation_i, operation_j, machine in X_index
    }

    A_plus = {
        (operation_i, operation_j, machine): model.addVar(
            vtype="B",
            name=f"A_plus[{operation_i},{operation_j},{machine}]",
        )
        for operation_i, operation_j, machine in X_index
    }
    A_minus = {
        (operation_i, operation_j, machine): model.addVar(
            vtype="B",
            name=f"A_minus[{operation_i},{operation_j},{machine}]",
        )
        for operation_i, operation_j, machine in X_index
    }

    R_bounds = {}
    for operation, machine in Y_index:
        eligible_predecessors = [
            other_operation
            for other_operation in instance.real_operations
            if other_operation != operation
            and machine in instance.eligible_machines[other_operation]
        ]
        R_bounds[operation, machine] = initial_age[machine] + sum(
            float(instance.processing_times[other_operation, machine])
            for other_operation in eligible_predecessors
        )

    R = {
        (operation, machine): model.addVar(
            vtype="C",
            lb=0.0,
            ub=R_bounds[operation, machine],
            name=f"R[{operation},{machine}]",
        )
        for operation, machine in Y_index
    }

    pi_fail = {
        (operation, machine): model.addVar(
            vtype="C",
            lb=0.0,
            ub=1.0,
            name=f"pi_fail[{operation},{machine}]",
        )
        for operation, machine in Y_index
    }

    failure_costs = {
        (operation, machine): _failure_cost(instance, operation, machine)
        for operation, machine in Y_index
    }
    total_failure_cost = quicksum(
        failure_costs[operation, machine] * pi_fail[operation, machine]
        for operation, machine in Y_index
    )
    if enforce_constraint:
        model.addCons(
            total_failure_cost <= constraint_budget,
            name="weibull_budget",
        )
    model.setObjective(C_max, "minimize")

    for operation in instance.real_operations:
        selected_processing_time = quicksum(
            Y[operation, machine]
            * float(instance.processing_times[operation, machine])
            for machine in instance.eligible_machines[operation]
        )
        model.addCons(
            quicksum(
                Y[operation, machine]
                for machine in instance.eligible_machines[operation]
            )
            == 1,
            name=f"assignment[{operation}]",
        )
        model.addCons(
            S[operation] == C[operation] - selected_processing_time,
            name=f"start_time_def[{operation}]",
        )

    for operation in instance.real_operations:
        selected_processing_time = quicksum(
            Y[operation, machine]
            * float(instance.processing_times[operation, machine])
            for machine in instance.eligible_machines[operation]
        )
        for predecessor in instance.predecessors.get(operation, []):
            model.addCons(
                C[operation] >= C[predecessor] + selected_processing_time,
                name=f"precedence[{predecessor}_before_{operation}]",
            )

    for operation_i, operation_j, machine in X_index:
        model.addCons(
            C[operation_i]
            >= C[operation_j]
            + float(instance.processing_times[operation_i, machine])
            - H
            * (
                2
                + X[operation_i, operation_j, machine]
                - Y[operation_i, machine]
                - Y[operation_j, machine]
            ),
            name=f"nonoverlap_j_before_i[{operation_j}_{operation_i}_{machine}]",
        )

        model.addCons(
            C[operation_j]
            >= C[operation_i]
            + float(instance.processing_times[operation_j, machine])
            - H
            * (
                3
                - X[operation_i, operation_j, machine]
                - Y[operation_i, machine]
                - Y[operation_j, machine]
            ),
            name=f"nonoverlap_i_before_j[{operation_i}_{operation_j}_{machine}]",
        )
        model.addCons(
            A_plus[operation_i, operation_j, machine] <= Y[operation_i, machine],
            name=f"adj_plus_yi_ub[{operation_i}_{operation_j}_{machine}]",
        )
        model.addCons(
            A_plus[operation_i, operation_j, machine] <= Y[operation_j, machine],
            name=f"adj_plus_yj_ub[{operation_i}_{operation_j}_{machine}]",
        )
        model.addCons(
            A_plus[operation_i, operation_j, machine]
            <= X[operation_i, operation_j, machine],
            name=f"adj_plus_x_ub[{operation_i}_{operation_j}_{machine}]",
        )
        model.addCons(
            A_plus[operation_i, operation_j, machine]
            >= (
                Y[operation_i, machine]
                + Y[operation_j, machine]
                + X[operation_i, operation_j, machine]
                - 2
            ),
            name=f"adj_plus_lb[{operation_i}_{operation_j}_{machine}]",
        )

        model.addCons(
            A_minus[operation_i, operation_j, machine] <= Y[operation_i, machine],
            name=f"adj_minus_yi_ub[{operation_i}_{operation_j}_{machine}]",
        )
        model.addCons(
            A_minus[operation_i, operation_j, machine] <= Y[operation_j, machine],
            name=f"adj_minus_yj_ub[{operation_i}_{operation_j}_{machine}]",
        )
        model.addCons(
            A_minus[operation_i, operation_j, machine]
            <= 1 - X[operation_i, operation_j, machine],
            name=f"adj_minus_x_ub[{operation_i}_{operation_j}_{machine}]",
        )
        model.addCons(
            A_minus[operation_i, operation_j, machine]
            >= (
                Y[operation_i, machine]
                + Y[operation_j, machine]
                - X[operation_i, operation_j, machine]
                - 1
            ),
            name=f"adj_minus_lb[{operation_i}_{operation_j}_{machine}]",
        )

    for operation in instance.real_operations:
        selected_processing_time = quicksum(
            Y[operation, machine]
            * float(instance.processing_times[operation, machine])
            for machine in instance.eligible_machines[operation]
        )
        model.addCons(
            C[operation] >= selected_processing_time,
            name=f"completion_lb[{operation}]",
        )

        for machine in instance.eligible_machines[operation]:
            predecessor_runtime = quicksum(
                float(instance.processing_times[other_operation, machine])
                * _directed_a(A_plus, A_minus, other_operation, operation, machine)
                for other_operation in instance.real_operations
                if other_operation != operation
                and machine in instance.eligible_machines[other_operation]
            )
            model.addCons(
                R[operation, machine]
                == initial_age[machine] * Y[operation, machine]
                + predecessor_runtime,
                name=f"machine_age_def[{operation},{machine}]",
            )

            processing_time = float(instance.processing_times[operation, machine])
            eta = weibull_eta[machine]
            beta = weibull_beta[machine]
            age_start = R[operation, machine] / eta
            age_end = (R[operation, machine] + processing_time) / eta
            failure_probability = Y[operation, machine] * (
                1 - exp(-((age_end ** beta) - (age_start ** beta)))
            )
            model.addCons(
                pi_fail[operation, machine] == failure_probability,
                name=f"failure_probability_def[{operation},{machine}]",
            )

    for job in instance.jobs:
        end_operation = instance.job_end_operations[job]
        model.addCons(
            C_max >= C[end_operation],
            name=f"makespan[{job}]",
        )

    if add_machine_load_lb:
        for machine in machines:
            operations_on_machine = [
                operation
                for operation in instance.real_operations
                if machine in instance.eligible_machines[operation]
            ]
            if operations_on_machine:
                model.addCons(
                    C_max
                    >= quicksum(
                        Y[operation, machine]
                        * float(instance.processing_times[operation, machine])
                        for operation in operations_on_machine
                    ),
                    name=f"machine_load_lb[{machine}]",
                )

    variables = {
        "C": C,
        "S": S,
        "C_max": C_max,
        "Y": Y,
        "X": X,
        "A_plus": A_plus,
        "A_minus": A_minus,
        "R": R,
        "pi_fail": pi_fail,
        "mu_fail": mu_fail,
        "H": H,
        "operations": instance.real_operations,
        "real_operations": instance.real_operations,
        "eligible_machines": instance.eligible_machines,
        "processing_times": instance.processing_times,
        "R_bounds": R_bounds,
        "machine_initial_age": initial_age,
        "weibull_eta": weibull_eta,
        "weibull_beta": weibull_beta,
        "failure_costs": failure_costs,
        "total_failure_cost": total_failure_cost,
        "constraint_type": constraint_type,
        "constraint_budget": constraint_budget,
        "constraint_enforced": bool(enforce_constraint),
        "constraint_safety_margin": float(safety_margin),
        "constraint_safety_margin_per_operation": float(safety_margin_per_operation),
        "scale_weibull_budget": bool(scale_weibull_budget),
        "weibull_budget_per_operation": float(weibull_budget_per_operation),
        "machines": machines,
        "jobs": instance.jobs,
        "X_index": X_index,
        "A_index": X_index,
        "Y_index": Y_index,
    }

    return model, variables


def write_solution_file(model: Model, variables, _instance, filename="solution.txt"):
    """Write a SCIP solution for the nonlinear FJSP model."""
    objective = ""
    makespan = ""
    total_failure_cost = ""
    exact_constraint_value = ""
    solution = None

    if model.getNSols() > 0:
        solution = model.getBestSol()
        objective = f"{float(model.getObjVal()):.4f}"
        makespan = f"{float(model.getSolVal(solution, variables['C_max'])):.4f}"
        total_failure_cost_value = sum(
            float(variables["failure_costs"][operation, machine])
            * float(model.getSolVal(solution, variables["pi_fail"][operation, machine]))
            for operation, machine in variables["Y_index"]
        )
        total_failure_cost = f"{total_failure_cost_value:.4f}"
        exact_constraint_value = total_failure_cost

    with open(filename, "w", encoding="utf-8") as file:
        file.write(f"Status: {model.getStatus()}\n")
        file.write(f"Objective: {objective}\n")
        file.write(f"Makespan: {makespan}\n")
        file.write(f"Total failure cost: {total_failure_cost}\n")
        file.write(f"Constraint type: {variables['constraint_type']}\n")
        file.write(f"Constraint value: {exact_constraint_value}\n")
        file.write(f"Constraint budget: {variables['constraint_budget']:.6f}\n")
        file.write(f"Constraint enforced: {variables['constraint_enforced']}\n")
        file.write(f"Big M: {float(variables['H']):.4f}\n")
        file.write(f"Berechnungszeit: {float(model.getSolvingTime()):.4f}\n")
        file.write("\nReliability parameters:\n")
        file.write(f"mu_fail: {float(variables['mu_fail']):.4f}\n")
        file.write(
            "machine_initial_age: "
            f"{_format_indexed_values(variables['machine_initial_age'], 'M')}\n"
        )
        file.write(
            f"weibull_eta: {_format_indexed_values(variables['weibull_eta'], 'M')}\n"
        )
        file.write(
            f"weibull_beta: {_format_indexed_values(variables['weibull_beta'], 'M')}\n"
        )
        file.write(
            f"failure_cost: {_format_failure_cost_values(variables['failure_costs'])}\n"
        )

        if solution is not None:
            file.write("\nOperation values:\n")
            for operation in variables["real_operations"]:
                c_value = float(model.getSolVal(solution, variables["C"][operation]))
                s_value = float(model.getSolVal(solution, variables["S"][operation]))
                file.write(
                    f"op {operation}: C={c_value:.4f}, S={s_value:.4f}\n"
                )

            file.write("\nY values:\n")
            for operation, machine in variables["Y_index"]:
                y_value = int(
                    round(
                        model.getSolVal(
                            solution,
                            variables["Y"][operation, machine],
                        )
                    )
                )
                p_value = float(variables["processing_times"][operation, machine])
                details = f"Y[{operation},{machine}] = {y_value}, p={p_value:.4f}"
                if variables["constraint_type"] == CONSTRAINT_WEIBULL:
                    r_value = float(
                        model.getSolVal(solution, variables["R"][operation, machine])
                    )
                    pi_value = float(model.getSolVal(
                        solution, variables["pi_fail"][operation, machine]
                    ))
                    cost_value = float(variables["failure_costs"][operation, machine])
                    details += (
                        f", R={r_value:.4f}, pi_fail={pi_value:.6f}, "
                        f"c_fail={cost_value:.4f}"
                    )
                file.write(details + "\n")


def build_and_solve(
    instance_name="i5_k5_1",
    instance_path=None,
    solution_path=None,
    time_limit=300.0,
    write_solution=True,
    verbose=True,
    **build_kwargs,
):
    """Convenience runner for quick local checks."""
    if instance_path is None:
        instance_path = ROOT_DIR / "02_data" / "fjsp_instances" / f"{instance_name}.fjsp"
    else:
        instance_path = Path(instance_path)

    if solution_path is None:
        solution_path = (
            ROOT_DIR
            / "data"
            / "fjsp_solutions"
            / "scip_nonlinear"
            / f"solution_{instance_name}_scip_nonlinear.txt"
        )
    else:
        solution_path = Path(solution_path)

    if not instance_path.exists():
        raise FileNotFoundError(f"Instance not found: {instance_path}")

    with open(instance_path, "rb") as file:
        instance = pickle.load(file)

    scip = Model(f"FJSP SCIP Nonlinear {instance_name}")
    if time_limit is not None and time_limit > 0:
        scip.setParam("limits/time", float(time_limit))
        if verbose:
            print(f"[info] SCIP time limit: {time_limit}s")

    model, variables = build_fjsp(scip, instance, **build_kwargs)

    if verbose:
        print(
            f"[info] Loese FJSP-MINLP: instance={instance_name}, "
            f"operations={len(instance.real_operations)}, "
            f"machines={instance.num_machines}"
        )

    model.optimize()

    status = model.getStatus()
    result = {"status": status, "instance": instance_name}
    print(f"\n=== SCIP Status: {status} ===")

    if model.getNSols() > 0:
        solution = model.getBestSol()
        objective = float(model.getObjVal())
        makespan = float(model.getSolVal(solution, variables["C_max"]))
        total_failure_cost = None
        if variables["constraint_type"] == CONSTRAINT_WEIBULL:
            total_failure_cost = sum(
                float(variables["failure_costs"][operation, machine])
                * float(model.getSolVal(solution, variables["pi_fail"][operation, machine]))
                for operation, machine in variables["Y_index"]
            )
        print(f"Objective:      {objective:.4f}")
        print(f"Makespan:       {makespan:.4f}")
        if total_failure_cost is not None:
            print(f"Failure cost:   {total_failure_cost:.4f}")

        if write_solution:
            solution_path.parent.mkdir(parents=True, exist_ok=True)
            write_solution_file(model, variables, instance, solution_path)
            print(f"Solution written to: {solution_path}")

        result.update(
            {
                "objective": objective,
                "makespan": makespan,
                "total_failure_cost": total_failure_cost,
                "solution_path": str(solution_path) if write_solution else None,
            }
        )
    else:
        print("Keine zulaessige Loesung gefunden.")

    return result, model


def _parse_main_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Build and solve the nonlinear SCIP FJSP MINLP."
    )
    parser.add_argument(
        "--instance-name",
        default="i5_k5_1",
        help="Instance name under 02_data/fjsp_instances without .fjsp suffix.",
    )
    parser.add_argument(
        "--instance-path",
        default=None,
        help="Optional explicit path to a .fjsp pickle instance.",
    )
    parser.add_argument(
        "--solution-path",
        default=None,
        help="Optional explicit output path for the solution file.",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=30.0,
        help="SCIP time limit in seconds. Use 0 for no limit.",
    )
    parser.add_argument(
        "--no-write-solution",
        action="store_true",
        help="Do not write a solution text file.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_main_args()

    build_and_solve(
        instance_name=args.instance_name,
        instance_path=args.instance_path,
        solution_path=args.solution_path,
        time_limit=args.time_limit,
        write_solution=not args.no_write_solution,
    )
