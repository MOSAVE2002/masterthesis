import importlib
import math
from pathlib import Path
import pickle
import sys

import gurobipy as gp
from gurobipy import GRB
from helper.gurobi_solution_writer import write_comparable_solution

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

_base_fjsp = importlib.import_module("03_Gurobi.build_fjsp")
STATUS_NAMES = _base_fjsp.STATUS_NAMES
from helper.surrogate_constraint import (
    CONSTRAINT_WEIBULL,
    validate_constraint_type,
)
from helper.sequence_setup import (
    add_reliability_graph_variables,
    normalize_reliability_graph_config,
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


def _repair_duration(instance, operation, machine, default=20.0):
    """Return the expected downtime after a failure on one machine."""
    for attr_name in ("repair_durations", "repair_duration", "tau"):
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
        if isinstance(value, (list, tuple)):
            if machine < len(value):
                return float(value[machine])
            continue
        return float(value)
    # Old instances used failure_cost for the same numeric experiment range.
    return _failure_cost(instance, operation, machine, default)


def _build_relational_fjsp(
    fjsp,
    instance,
    add_machine_load_lb=True,
    constraint_type=CONSTRAINT_WEIBULL,
    enforce_constraint=False,
    weibull_budget_per_operation=0.4,
    reliability_graph_config=None,
):
    """Exact Weibull model with a direct-predecessor transition hazard.

    The legacy budget arguments remain accepted for compatibility, but this
    formulation no longer constrains total expected failure delay.  Expected
    delay remains part of every effective operation duration and is minimized
    through ``C_max``.
    """
    graph_cfg = normalize_reliability_graph_config(reliability_graph_config)
    constraint_type = validate_constraint_type(constraint_type)
    operations = list(instance.real_operations)
    machines = list(range(instance.num_machines))
    if not hasattr(instance, "mu_fail"):
        instance._set_default_nonlinear_parameters()
    initial_age = {
        machine: _first_machine_parameter(
            instance, ("machine_initial_age", "R0", "r0"), machine, 0.0
        )
        for machine in machines
    }
    eta = {
        machine: _first_machine_parameter(
            instance, ("weibull_eta", "eta"), machine, 100.0
        )
        for machine in machines
    }
    beta = {machine: graph_cfg.beta for machine in machines}
    for machine in machines:
        if initial_age[machine] < 0.0:
            raise ValueError("machine_initial_age must be nonnegative.")
        if eta[machine] <= 0.0:
            raise ValueError("weibull_eta must be positive.")
        if beta[machine] <= 0.0:
            raise ValueError("weibull_beta must be positive.")
    repair_durations = {
        (operation, machine): _repair_duration(instance, operation, machine)
        for operation in operations
        for machine in instance.eligible_machines[operation]
    }
    if any(value < 0.0 for value in repair_durations.values()):
        raise ValueError("repair_duration must be nonnegative.")
    extra_upper = {
        operation: max(
            repair_durations[operation, machine]
            for machine in instance.eligible_machines[operation]
        )
        for operation in operations
    }
    model, variables = _base_fjsp.build_fjsp(
        fjsp,
        instance,
        extra_duration_upper_bounds=extra_upper,
        add_time_constraints=False,
    )
    model.Params.NonConvex = 2
    variables.update(
        {
            "machine_initial_age": initial_age,
            "weibull_eta": eta,
            "weibull_beta": beta,
            "repair_durations": repair_durations,
        }
    )
    add_reliability_graph_variables(
        model, variables, instance, graph_cfg
    )
    Y, X = variables["Y"], variables["X"]
    R = variables["R"]
    operation_pi_fail = {}
    pi_fail = {}
    common_beta = graph_cfg.beta
    for operation in operations:
        operation_pi_fail[operation] = model.addVar(
            lb=0.0,
            ub=1.0,
            vtype=GRB.CONTINUOUS,
            name=f"pi_fail_operation[{operation}]",
        )
        eligible = instance.eligible_machines[operation]
        if math.isclose(common_beta, 2.0, rel_tol=0.0, abs_tol=1e-12):
            # Exact expansion of ((R+p)/eta)^2-(R/eta)^2.  The
            # hazard is affine; only the exponential remains nonlinear.
            weibull_hazard = gp.quicksum(
                (
                    2.0 * float(instance.processing_times[operation, machine])
                    * R[operation, machine]
                    + float(instance.processing_times[operation, machine]) ** 2
                    * Y[operation, machine]
                ) / eta[machine] ** 2
                for machine in eligible
            )
        elif math.isclose(common_beta, 1.0, rel_tol=0.0, abs_tol=1e-12):
            weibull_hazard = gp.quicksum(
                float(instance.processing_times[operation, machine])
                / eta[machine] * Y[operation, machine]
                for machine in eligible
            )
        else:
            selected_age = gp.quicksum(
                R[operation, machine] / eta[machine]
                for machine in eligible
            )
            selected_processing = gp.quicksum(
                float(instance.processing_times[operation, machine])
                / eta[machine] * Y[operation, machine]
                for machine in eligible
            )
            weibull_hazard = (
                (selected_age + selected_processing) ** common_beta
                - selected_age ** common_beta
            )
        hazard_increment = (
            weibull_hazard
            + graph_cfg.transition_gamma
            * variables["transition_load"][operation]
        )
        probability = 1 - gp.nlfunc.exp(-hazard_increment)
        model.addGenConstrNL(
            operation_pi_fail[operation],
            probability,
            name=f"relational_failure_probability[{operation}]",
        )
        for machine in eligible:
            assignment = Y[operation, machine]
            selected_probability = model.addVar(
                lb=0.0,
                ub=1.0,
                vtype=GRB.CONTINUOUS,
                name=f"pi_fail[{operation},{machine}]",
            )
            model.addConstr(
                selected_probability <= operation_pi_fail[operation],
                name=f"pi_fail_product_ub_probability[{operation},{machine}]",
            )
            model.addConstr(
                selected_probability <= assignment,
                name=f"pi_fail_product_ub_assignment[{operation},{machine}]",
            )
            model.addConstr(
                selected_probability
                >= operation_pi_fail[operation] - (1.0 - assignment),
                name=f"pi_fail_product_lb[{operation},{machine}]",
            )
            pi_fail[operation, machine] = selected_probability
    probability_scope = "operation"

    Delta, D, S = {}, {}, {}
    for operation in operations:
        max_delay = max(
            repair_durations[operation, machine]
            for machine in instance.eligible_machines[operation]
        )
        Delta[operation] = model.addVar(
            lb=0.0,
            ub=max_delay,
            vtype=GRB.CONTINUOUS,
            name=f"Delta[{operation}]",
        )
        model.addConstr(
            Delta[operation]
            == gp.quicksum(
                repair_durations[operation, machine]
                * pi_fail[operation, machine]
                for machine in instance.eligible_machines[operation]
            ),
            name=f"expected_failure_delay_def[{operation}]",
        )
        selected_processing = gp.quicksum(
            Y[operation, machine]
            * float(instance.processing_times[operation, machine])
            for machine in instance.eligible_machines[operation]
        )
        max_processing = max(
            float(instance.processing_times[operation, machine])
            for machine in instance.eligible_machines[operation]
        )
        D[operation] = model.addVar(
            lb=0.0,
            ub=max_processing + max_delay,
            vtype=GRB.CONTINUOUS,
            name=f"D[{operation}]",
        )
        model.addConstr(
            D[operation]
            == selected_processing + Delta[operation],
            name=f"relational_effective_duration[{operation}]",
        )
        S[operation] = model.addVar(
            lb=0.0,
            ub=float(variables["H"]),
            vtype=GRB.CONTINUOUS,
            name=f"S[{operation}]",
        )
        model.addConstr(
            variables["C"][operation] == S[operation] + D[operation]
        )

    for operation in operations:
        for predecessor in instance.predecessors.get(operation, []):
            model.addConstr(
                variables["C"][operation]
                >= variables["C"][predecessor] + D[operation],
                name=f"relational_precedence[{predecessor},{operation}]",
            )
        model.addConstr(variables["C"][operation] >= D[operation])
    H = float(variables["H"])
    for operation_i, operation_j, machine in variables["X_index"]:
        model.addConstr(
            variables["C"][operation_i]
            >= variables["C"][operation_j] + D[operation_i]
            - H * (
                2 + X[operation_i, operation_j, machine]
                - Y[operation_i, machine] - Y[operation_j, machine]
            )
        )
        model.addConstr(
            variables["C"][operation_j]
            >= variables["C"][operation_i] + D[operation_j]
            - H * (
                3 - X[operation_i, operation_j, machine]
                - Y[operation_i, machine] - Y[operation_j, machine]
            )
        )

    total_failure_delay = gp.quicksum(
        Delta[operation] for operation in operations
    )
    if add_machine_load_lb:
        for machine in machines:
            eligible = [
                operation
                for operation in operations
                if machine in instance.eligible_machines[operation]
            ]
            if eligible:
                model.addConstr(
                    variables["C_max"]
                    >= gp.quicksum(
                        Y[operation, machine]
                        * float(instance.processing_times[operation, machine])
                        + repair_durations[operation, machine]
                        * pi_fail[operation, machine]
                        for operation in eligible
                    ),
                    name=f"relational_machine_load_lb[{machine}]",
                )
    variables.update(
        {
            "pi_fail": pi_fail,
            "operation_pi_fail": operation_pi_fail,
            "weibull_probability_scope": probability_scope,
            "Delta": Delta,
            "D": D,
            "S": S,
            "mu_fail": float(getattr(instance, "mu_fail", 1.0)),
            "total_failure_delay": total_failure_delay,
            "budget_constraint": None,
            "constraint_type": constraint_type,
            "constraint_budget": None,
            "constraint_enforced": False,
            "weibull_budget_per_operation": float(
                weibull_budget_per_operation
            ),
            "formulation": "nonlinear_transition_hazard_weibull_unconstrained",
        }
    )
    model.setObjective(variables["C_max"], GRB.MINIMIZE)
    model.update()
    return model, variables


def build_fjsp(
    fjsp,
    instance,
    add_machine_load_lb=True,
    constraint_type=CONSTRAINT_WEIBULL,
    enforce_constraint=False,
    weibull_budget_per_operation=0.4,
    reliability_graph_config=None,
):
    """Build the exact Weibull expected-delay makespan formulation."""
    return _build_relational_fjsp(
        fjsp,
        instance,
        add_machine_load_lb=add_machine_load_lb,
        constraint_type=constraint_type,
        enforce_constraint=enforce_constraint,
        weibull_budget_per_operation=weibull_budget_per_operation,
        reliability_graph_config=reliability_graph_config,
    )


def build_linear_schedule_for_analytical_labeling(
    fjsp,
    instance,
    add_machine_load_lb=True,
    constraint_type=CONSTRAINT_WEIBULL,
    weibull_budget_per_operation=0.4,
    reliability_graph_config=None,
    **_unused,
):
    """Build the budget-free scheduling core for analytical GNN labels.

    ``weibull_budget_per_operation`` is accepted only for compatibility with
    older callers and has no effect on this model or on sample selection.
    """
    graph_cfg = normalize_reliability_graph_config(reliability_graph_config)
    constraint_type = validate_constraint_type(constraint_type)
    operations = list(instance.real_operations)
    machines = list(range(instance.num_machines))
    if not hasattr(instance, "mu_fail"):
        instance._set_default_nonlinear_parameters()
    initial_age = {
        machine: _first_machine_parameter(
            instance, ("machine_initial_age", "R0", "r0"), machine, 0.0
        )
        for machine in machines
    }
    eta = {
        machine: _first_machine_parameter(
            instance, ("weibull_eta", "eta"), machine, 100.0
        )
        for machine in machines
    }
    beta = {machine: graph_cfg.beta for machine in machines}
    repair_durations = {
        (operation, machine): _repair_duration(instance, operation, machine)
        for operation in operations
        for machine in instance.eligible_machines[operation]
    }
    for machine in machines:
        if initial_age[machine] < 0.0:
            raise ValueError("machine_initial_age must be nonnegative.")
        if eta[machine] <= 0.0:
            raise ValueError("weibull_eta must be positive.")
        if beta[machine] <= 0.0:
            raise ValueError("weibull_beta must be positive.")
    if any(value < 0.0 for value in repair_durations.values()):
        raise ValueError("repair_duration must be nonnegative.")

    model, variables = _base_fjsp.build_fjsp(
        fjsp,
        instance,
        extra_duration_upper_bounds={operation: 0.0 for operation in operations},
        add_time_constraints=False,
    )
    variables.update(
        {
            "machine_initial_age": initial_age,
            "weibull_eta": eta,
            "weibull_beta": beta,
            "repair_durations": repair_durations,
        }
    )
    add_reliability_graph_variables(model, variables, instance, graph_cfg)
    Y, X = variables["Y"], variables["X"]

    duration, start = {}, {}
    for operation in operations:
        selected_processing = gp.quicksum(
            Y[operation, machine]
            * float(instance.processing_times[operation, machine])
            for machine in instance.eligible_machines[operation]
        )
        max_processing = max(
            float(instance.processing_times[operation, machine])
            for machine in instance.eligible_machines[operation]
        )
        duration[operation] = model.addVar(
            lb=0.0,
            ub=max_processing,
            vtype=GRB.CONTINUOUS,
            name=f"linear_label_duration[{operation}]",
        )
        model.addConstr(
            duration[operation] == selected_processing
        )
        start[operation] = model.addVar(
            lb=0.0,
            ub=float(variables["H"]),
            vtype=GRB.CONTINUOUS,
            name=f"linear_label_start[{operation}]",
        )
        model.addConstr(
            variables["C"][operation]
            == start[operation] + duration[operation]
        )

    for operation in operations:
        for predecessor in instance.predecessors.get(operation, []):
            model.addConstr(
                variables["C"][operation]
                >= variables["C"][predecessor] + duration[operation],
                name=f"linear_label_precedence[{predecessor},{operation}]",
            )
        model.addConstr(variables["C"][operation] >= duration[operation])

    big_m = float(variables["H"])
    for operation_i, operation_j, machine in variables["X_index"]:
        model.addConstr(
            variables["C"][operation_i]
            >= variables["C"][operation_j] + duration[operation_i]
            - big_m * (
                2 + X[operation_i, operation_j, machine]
                - Y[operation_i, machine] - Y[operation_j, machine]
            )
        )
        model.addConstr(
            variables["C"][operation_j]
            >= variables["C"][operation_i] + duration[operation_j]
            - big_m * (
                3 - X[operation_i, operation_j, machine]
                - Y[operation_i, machine] - Y[operation_j, machine]
            )
        )

    if add_machine_load_lb:
        for machine in machines:
            eligible = [
                operation
                for operation in operations
                if machine in instance.eligible_machines[operation]
            ]
            if eligible:
                model.addConstr(
                    variables["C_max"]
                    >= gp.quicksum(
                        Y[operation, machine]
                        * float(instance.processing_times[operation, machine])
                        for operation in eligible
                    ),
                    name=f"linear_label_machine_load_lb[{machine}]",
                )

    variables.update(
        {
            "D": duration,
            "S": start,
            "constraint_type": constraint_type,
            "constraint_budget": None,
            "constraint_enforced": False,
            "formulation": "linear_schedule_analytical_weibull_labels",
        }
    )
    model.setObjective(variables["C_max"], GRB.MINIMIZE)
    model.update()
    return model, variables


def analytical_weibull_labels(variables, solution_number=None):
    """Evaluate exact Weibull probabilities for one fixed solved schedule."""
    def solution_value(variable):
        return float(
            variable.X if solution_number is None else variable.Xn
        )

    probabilities = {}
    delays = {}
    for operation in variables["real_operations"]:
        selected = [
            machine
            for machine in variables["eligible_machines"][operation]
            if solution_value(
                variables["Y"][operation, machine]
            ) > 0.5
        ]
        if len(selected) != 1:
            raise ValueError(
                f"Expected one selected machine for operation {operation}."
            )
        machine = selected[0]
        age = solution_value(variables["R"][operation, machine])
        processing = float(
            variables["processing_times"][operation, machine]
        )
        eta = float(variables["weibull_eta"][machine])
        beta = float(variables["weibull_beta"][machine])
        hazard = (
            ((age + processing) / eta) ** beta
            - (age / eta) ** beta
            + float(variables["reliability_graph_config"]["transition_gamma"])
            * solution_value(variables["transition_load"][operation])
        )
        probability = min(1.0, max(0.0, 1.0 - math.exp(-hazard)))
        delay = (
            float(variables["repair_durations"][operation, machine])
            * probability
        )
        probabilities[operation] = probability
        delays[operation] = delay
    return {
        "probabilities": probabilities,
        "delays": delays,
        "total_failure_delay": sum(delays.values()),
    }


def write_solution_file(model, variables, instance, filename="solution.txt"):
    """Write the common exact/GNN comparison format."""
    return write_comparable_solution(
        model,
        variables,
        filename,
        instance=instance,
    )


def _operation_labels(instance):
    labels = {}
    for job, operations in sorted(instance.jobs.items()):
        for local_idx, operation in enumerate(operations, start=1):
            labels[operation] = f"J{job}O{local_idx}"

    return labels


def _operation_math_labels(instance):
    labels = {}
    for job, operations in sorted(instance.jobs.items()):
        for local_idx, operation in enumerate(operations, start=1):
            labels[operation] = rf"$O_{{{job}{local_idx}}}$"

    return labels


def _operation_jobs(instance):
    operation_jobs = {}
    for job, operations in instance.jobs.items():
        for operation in operations:
            operation_jobs[operation] = job

    return operation_jobs


def _candidate_operation_edges_from_instance(instance):
    edges = []
    operations = list(instance.real_operations)
    for idx_i, operation_i in enumerate(operations):
        machines_i = set(instance.eligible_machines[operation_i])
        for operation_j in operations[idx_i + 1:]:
            common_machines = machines_i & set(instance.eligible_machines[operation_j])
            for machine in sorted(common_machines):
                if (
                    (operation_i, machine) in instance.processing_times
                    and (operation_j, machine) in instance.processing_times
                ):
                    edges.append((operation_i, operation_j, machine))

    return edges


def _group_candidate_operation_edges(candidate_edges):
    grouped_edges = {}
    for operation_i, operation_j, machine in candidate_edges:
        grouped_edges.setdefault((operation_i, operation_j), []).append(machine)

    return [
        (operation_i, operation_j, tuple(machines))
        for (operation_i, operation_j), machines in sorted(grouped_edges.items())
    ]


def _selected_machine(variables, operation):
    eligible_machines = variables["eligible_machines"][operation]
    return max(
        eligible_machines,
        key=lambda machine: float(variables["Y"][operation, machine].X),
    )


def _candidate_operation_edges_from_assignment(variables, instance):
    if "Y" not in variables:
        return None

    try:
        selected_machines = {
            operation: _selected_machine(variables, operation)
            for operation in instance.real_operations
        }
    except Exception:
        return None

    edges = []
    operations = list(instance.real_operations)
    for idx_i, operation_i in enumerate(operations):
        for operation_j in operations[idx_i + 1:]:
            machine = selected_machines[operation_i]
            if machine != selected_machines[operation_j]:
                continue

            if (
                (operation_i, machine) in instance.processing_times
                and (operation_j, machine) in instance.processing_times
            ):
                edges.append((operation_i, operation_j, machine))

    return edges


def _a_machine_order_edges(variables):
    edges = []
    for operation_i, operation_j, machine in variables["A_index"]:
        a_plus = int(round(variables["A_plus"][operation_i, operation_j, machine].X))
        a_minus = int(round(variables["A_minus"][operation_i, operation_j, machine].X))

        if a_plus:
            edges.append((operation_i, operation_j, machine))
        if a_minus:
            edges.append((operation_j, operation_i, machine))

    return edges


def _machine_operation_order(machine_operations, machine_edges, variables):
    predecessor_count = {
        operation: 0
        for operation in machine_operations
    }
    operation_set = set(machine_operations)
    for source, target, _machine in machine_edges:
        if source in operation_set and target in operation_set:
            predecessor_count[target] += 1

    return sorted(
        machine_operations,
        key=lambda operation: (
            predecessor_count[operation],
            float(variables["C"][operation].X),
            operation,
        ),
    )


def _prepare_plotting(filename):
    import os

    cache_root = Path(os.environ.get("TMPDIR", "/tmp")) / "fjsp_plot_cache"
    mpl_cache = cache_root / "matplotlib"
    xdg_cache = cache_root / "xdg"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    xdg_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache))

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    return filename, plt, Line2D


def plot_solution_schedule(
    variables,
    instance,
    filename,
    title=None,
):
    """Plot an optimal FJSP solution as a machine-based Gantt chart."""
    filename, plt, _Line2D = _prepare_plotting(filename)
    from matplotlib.patches import Patch

    operations = list(variables["real_operations"])
    machines = list(variables["machines"])
    operation_labels = _operation_math_labels(instance)
    operation_jobs = _operation_jobs(instance)
    jobs = sorted(instance.jobs)

    selected_machines = {
        operation: _selected_machine(variables, operation)
        for operation in operations
    }
    schedule = []
    for operation in operations:
        machine = selected_machines[operation]
        start = float(variables["S"][operation].X)
        duration = float(variables["processing_times"][operation, machine])
        schedule.append(
            {
                "operation": operation,
                "job": operation_jobs[operation],
                "machine": machine,
                "start": start,
                "duration": duration,
            }
        )

    makespan = float(variables["C_max"].X)
    job_palette = [
        "#4E79A7",
        "#F28E2B",
        "#59A14F",
        "#E15759",
        "#B07AA1",
        "#76B7B2",
        "#EDC948",
        "#FF9DA7",
        "#9C755F",
        "#BAB0AC",
    ]
    job_colors = {
        job: job_palette[idx % len(job_palette)]
        for idx, job in enumerate(jobs)
    }

    figure_width = max(10.0, min(18.0, 8.0 + 0.12 * makespan))
    figure_height = max(4.2, min(12.0, 1.0 + 0.8 * len(machines)))
    fig, ax = plt.subplots(figsize=(figure_width, figure_height))

    bar_height = 0.62
    for item in sorted(
        schedule,
        key=lambda entry: (entry["machine"], entry["start"], entry["operation"]),
    ):
        ax.barh(
            item["machine"],
            item["duration"],
            left=item["start"],
            height=bar_height,
            color=job_colors[item["job"]],
            edgecolor="black",
            linewidth=1.2,
            zorder=3,
        )
        ax.text(
            item["start"] + item["duration"] / 2.0,
            item["machine"],
            operation_labels[item["operation"]],
            ha="center",
            va="center",
            fontsize=11,
            color="black",
            clip_on=True,
            zorder=4,
        )

    ax.axvline(
        makespan,
        color="red",
        linestyle=(0, (7, 5)),
        linewidth=2.0,
        zorder=2,
    )
    ax.annotate(
        rf"$C_{{\max}}={makespan:g}$",
        xy=(makespan, machines[-1] if machines else 0),
        xytext=(-8, 16),
        textcoords="offset points",
        ha="right",
        va="bottom",
        color="red",
        fontsize=13,
    )

    ax.set_xlabel("Zeit")
    ax.set_ylabel("Ressourcen")
    ax.set_yticks(machines)
    ax.set_yticklabels([f"Maschine {machine + 1}" for machine in machines])
    ax.set_xlim(0.0, max(makespan * 1.04, 1.0))
    if machines:
        ax.set_ylim(min(machines) - 0.6, max(machines) + 0.6)
    ax.grid(axis="x", color="#d0d0d0", linewidth=0.8, alpha=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(title or "Produktionsprogrammplanung")

    legend_handles = [
        Patch(
            facecolor=job_colors[job],
            edgecolor="black",
            label=f"Job {job}",
        )
        for job in jobs
    ]
    legend_handles.append(
        _Line2D(
            [],
            [],
            linestyle="none",
            marker="",
            label=r"$O_{ij}$ = Operation $j$ von Job $i$",
        )
    )
    if legend_handles:
        ax.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.16),
            ncol=min(6, len(legend_handles)),
            frameon=False,
        )

    fig.tight_layout()
    fig.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return filename


def _disjunctive_machine_arrow_rad(source_pos, target_pos):
    if abs(source_pos[1] - target_pos[1]) < 1e-9:
        return 0.22

    return 0.28 if source_pos[1] <= target_pos[1] else -0.28


def _a_machine_order_orientation_by_pair(variables):
    try:
        a_edges = _a_machine_order_edges(variables)
    except Exception:
        return {}

    return {
        (frozenset((source, target)), machine): (source, target)
        for source, target, machine in a_edges
    }


def _plot_disjunctive_solution_graph(
    model,
    variables,
    instance,
    filename,
    title,
    plt,
    Line2D,
):
    jobs = sorted(instance.jobs)
    labels = _operation_math_labels(instance)
    max_job_length = max(len(instance.jobs[job]) for job in jobs)
    middle_y = (len(jobs) + 1) / 2

    positions = {}
    for row_idx, job in enumerate(jobs):
        y_pos = len(jobs) - row_idx
        for col_idx, operation in enumerate(instance.jobs[job], start=1):
            positions[operation] = (col_idx, y_pos)

    start_pos = (0, middle_y)
    end_pos = (max_job_length + 1, middle_y)

    all_machines = sorted(variables.get("machines", range(instance.num_machines)))
    selected_machines = {
        operation: _selected_machine(variables, operation)
        for operation in variables["real_operations"]
    }
    operations_by_machine = {machine: [] for machine in all_machines}
    for operation, machine in selected_machines.items():
        operations_by_machine.setdefault(machine, []).append(operation)

    a_edges = _a_machine_order_edges(variables)
    ordered_operations_by_machine = {}
    for machine in all_machines:
        machine_edges = [edge for edge in a_edges if edge[2] == machine]
        ordered_operations_by_machine[machine] = _machine_operation_order(
            operations_by_machine.get(machine, []),
            machine_edges,
            variables,
        )

    palette = ["#d7191c", "#2c7bb6", "#f2c500", "#1a9641", "#984ea3", "#ff7f00"]
    machine_colors = {
        machine: palette[idx % len(palette)]
        for idx, machine in enumerate(all_machines)
    }

    fig_width = max(8.5, min(18.0, 1.75 * (max_job_length + 2)))
    fig_height = max(4.8, min(14.0, 1.2 * len(jobs) + 2.2))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    def draw_arrow(source_pos, target_pos, color, width, alpha=1.0, rad=0.0, zorder=2):
        ax.annotate(
            "",
            xy=target_pos,
            xytext=source_pos,
            arrowprops={
                "arrowstyle": "-|>",
                "color": color,
                "lw": width,
                "alpha": alpha,
                "shrinkA": 23,
                "shrinkB": 23,
                "mutation_scale": 14,
                "connectionstyle": f"arc3,rad={rad}",
            },
            zorder=1,
        )

    def draw_node(x_pos, y_pos, label, edgecolor, size=1700):
        ax.scatter(
            [x_pos],
            [y_pos],
            s=size,
            marker="o",
            facecolor="white",
            edgecolor=edgecolor,
            linewidth=1.8,
            zorder=3,
        )
        ax.text(
            x_pos,
            y_pos,
            label,
            ha="center",
            va="center",
            fontsize=12,
            zorder=4,
        )

    for job in jobs:
        operations = instance.jobs[job]
        if not operations:
            continue

        draw_arrow(start_pos, positions[operations[0]], color="black", width=1.7)
        for source, target in zip(operations, operations[1:]):
            draw_arrow(
                positions[source],
                positions[target],
                color="black",
                width=1.7,
            )
        draw_arrow(positions[operations[-1]], end_pos, color="black", width=1.7)

    for machine, operations in ordered_operations_by_machine.items():
        color = machine_colors[machine]
        for source, target in zip(operations, operations[1:]):
            source_pos = positions[source]
            target_pos = positions[target]
            rad = _disjunctive_machine_arrow_rad(source_pos, target_pos)
            draw_arrow(
                source_pos,
                target_pos,
                color=color,
                width=1.8,
                alpha=0.95,
                rad=rad,
            )

    draw_node(*start_pos, "Start", edgecolor="black", size=1800)
    draw_node(*end_pos, "End", edgecolor="black", size=1800)
    for operation, (x_pos, y_pos) in positions.items():
        machine = selected_machines[operation]
        draw_node(
            x_pos,
            y_pos,
            labels.get(operation, f"O{operation}"),
            edgecolor=machine_colors[machine],
            size=1700,
        )

    makespan = float(variables["C_max"].X)
    objective = float(model.ObjVal)
    title = title or "Gurobi nonlinear disjunctive solution graph"
    ax.set_title(
        f"{title}\nObjective {objective:.2f} | C_max {makespan:.2f}",
        fontsize=12,
    )
    ax.set_xlim(-0.55, max_job_length + 1.55)
    ax.set_ylim(0.35, len(jobs) + 0.65)
    ax.axis("off")

    legend_handles = [
        Line2D([0], [0], color="black", lw=1.7, label="Job order"),
    ]
    legend_handles.extend(
        Line2D(
            [0],
            [0],
            color=machine_colors[machine],
            lw=1.8,
            label=f"M{machine}",
        )
        for machine in all_machines
    )
    ax.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
    )

    fig.tight_layout()
    fig.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return filename


def plot_candidate_graph(
    variables,
    instance,
    filename,
    title=None,
):
    filename, plt, Line2D = _prepare_plotting(filename)

    jobs = sorted(instance.jobs)
    labels = _operation_math_labels(instance)
    max_job_length = max(len(instance.jobs[job]) for job in jobs)
    middle_y = (len(jobs) + 1) / 2

    positions = {}
    for row_idx, job in enumerate(jobs):
        y_pos = len(jobs) - row_idx
        for col_idx, operation in enumerate(instance.jobs[job], start=1):
            positions[operation] = (col_idx, y_pos)

    start_pos = (0, middle_y)
    end_pos = (max_job_length + 1, middle_y)

    candidate_edges = _candidate_operation_edges_from_assignment(variables, instance)
    if candidate_edges is None:
        candidate_edges = _candidate_operation_edges_from_instance(instance)
        edge_subtitle = (
            f"{len(candidate_edges)} possible machine-specific operation edges"
        )
    else:
        edge_subtitle = (
            f"{len(candidate_edges)} possible machine-order edges "
            "after Y assignment"
        )
    candidate_edge_groups = _group_candidate_operation_edges(candidate_edges)
    candidate_edge_orientations = _a_machine_order_orientation_by_pair(variables)
    all_machines = sorted(
        variables.get(
            "machines",
            {
                machine
                for eligible_machines in instance.eligible_machines.values()
                for machine in eligible_machines
            },
        )
    )
    palette = ["#d7191c", "#2c7bb6", "#f2c500", "#1a9641", "#984ea3", "#ff7f00"]
    machine_colors = {
        machine: palette[idx % len(palette)]
        for idx, machine in enumerate(all_machines)
    }

    fig_width = max(8.5, min(18.0, 1.75 * (max_job_length + 2)))
    fig_height = max(4.8, min(14.0, 1.2 * len(jobs) + 2.2))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    def draw_arrow(source_pos, target_pos, color, width, alpha=1.0, rad=0.0, zorder=2):
        ax.annotate(
            "",
            xy=target_pos,
            xytext=source_pos,
            arrowprops={
                "arrowstyle": "-|>",
                "color": color,
                "lw": width,
                "alpha": alpha,
                "shrinkA": 23,
                "shrinkB": 23,
                "mutation_scale": 14,
                "connectionstyle": f"arc3,rad={rad}",
            },
            zorder=zorder,
        )

    def draw_job_arrow(source_pos, target_pos):
        draw_arrow(source_pos, target_pos, color="white", width=5.0, zorder=2)
        draw_arrow(source_pos, target_pos, color="black", width=1.7, zorder=3)

    def draw_candidate_edge(source_pos, target_pos, color, rad):
        ax.annotate(
            "",
            xy=target_pos,
            xytext=source_pos,
            arrowprops={
                "arrowstyle": "-|>",
                "color": color,
                "lw": 1.25,
                "alpha": 0.55,
                "linestyle": "--",
                "shrinkA": 24,
                "shrinkB": 24,
                "mutation_scale": 11,
                "connectionstyle": f"arc3,rad={rad}",
            },
            zorder=1,
        )

    def draw_node(x_pos, y_pos, label, size=1700):
        ax.scatter(
            [x_pos],
            [y_pos],
            s=size,
            marker="o",
            facecolor="white",
            edgecolor="black",
            linewidth=1.6,
            zorder=3,
        )
        ax.text(
            x_pos,
            y_pos,
            label,
            ha="center",
            va="center",
            fontsize=12,
            zorder=4,
        )

    for source, target, machines in candidate_edge_groups:
        if source not in positions or target not in positions:
            continue

        center = (len(machines) - 1) / 2
        for idx, machine in enumerate(machines):
            edge_source, edge_target = candidate_edge_orientations.get(
                (frozenset((source, target)), machine),
                (source, target),
            )
            if edge_source not in positions or edge_target not in positions:
                continue

            source_pos = positions[edge_source]
            target_pos = positions[edge_target]
            rad = _disjunctive_machine_arrow_rad(source_pos, target_pos)
            if len(machines) > 1:
                rad += (idx - center) * 0.16

            draw_candidate_edge(
                source_pos,
                target_pos,
                color=machine_colors[machine],
                rad=rad,
            )

    for job in jobs:
        operations = instance.jobs[job]
        if not operations:
            continue

        draw_job_arrow(start_pos, positions[operations[0]])
        for source, target in zip(operations, operations[1:]):
            draw_job_arrow(positions[source], positions[target])
        draw_job_arrow(positions[operations[-1]], end_pos)

    draw_node(*start_pos, "Start", size=1800)
    draw_node(*end_pos, "End", size=1800)
    for operation, (x_pos, y_pos) in positions.items():
        draw_node(
            x_pos,
            y_pos,
            labels.get(operation, f"O{operation}"),
            size=1700,
        )

    title = title or "Gurobi nonlinear candidate graph"
    ax.set_title(
        f"{title}\n{edge_subtitle}",
        fontsize=12,
    )
    ax.set_xlim(-0.55, max_job_length + 1.55)
    ax.set_ylim(0.35, len(jobs) + 0.65)
    ax.axis("off")

    legend_handles = [
        Line2D([0], [0], color="black", lw=1.7, label="Job order"),
    ]
    legend_handles.extend(
        Line2D(
            [0],
            [0],
            color=machine_colors[machine],
            lw=1.25,
            ls="--",
            label=f"M{machine}",
        )
        for machine in all_machines
    )
    ax.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
    )

    fig.tight_layout()
    fig.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return filename


def plot_solution_graph_from_a(
    model,
    variables,
    instance,
    filename,
    title=None,
    style="machine_operation",
):
    if model.SolCount == 0:
        raise ValueError("Cannot plot a solution graph without a solution.")

    filename, plt, Line2D = _prepare_plotting(filename)

    normalized_style = str(style).strip().lower()
    if normalized_style in {"disjunctive", "disjunctive_solution"}:
        return _plot_disjunctive_solution_graph(
            model,
            variables,
            instance,
            filename,
            title,
            plt,
            Line2D,
        )

    jobs = sorted(instance.jobs)
    labels = _operation_labels(instance)
    operation_jobs = _operation_jobs(instance)

    all_machines = sorted(variables.get("machines", range(instance.num_machines)))
    selected_machines = {
        operation: _selected_machine(variables, operation)
        for operation in variables["real_operations"]
    }
    operations_by_machine = {
        machine: []
        for machine in all_machines
    }
    for operation, machine in selected_machines.items():
        operations_by_machine.setdefault(machine, []).append(operation)

    a_edges = _a_machine_order_edges(variables)
    ordered_operations_by_machine = {}
    for machine in all_machines:
        machine_operations = operations_by_machine.get(machine, [])
        machine_edges = [
            edge
            for edge in a_edges
            if edge[2] == machine
        ]
        ordered_operations_by_machine[machine] = _machine_operation_order(
            machine_operations,
            machine_edges,
            variables,
        )

    positions = {}
    machine_positions = {}
    for row_idx, machine in enumerate(all_machines):
        y_pos = len(all_machines) - row_idx
        machine_positions[machine] = (0, y_pos)
        for col_idx, operation in enumerate(
            ordered_operations_by_machine[machine],
            start=1,
        ):
            positions[operation] = (col_idx, y_pos)

    max_machine_length = max(
        1,
        max(
            [len(operations) for operations in ordered_operations_by_machine.values()]
            or [1]
        ),
    )
    job_color_map = plt.get_cmap("tab20", max(len(jobs), 1))
    job_colors = {
        job: job_color_map(idx)
        for idx, job in enumerate(jobs)
    }
    machine_color_map = plt.get_cmap("Set2", max(len(all_machines), 1))
    machine_colors = {
        machine: machine_color_map(idx)
        for idx, machine in enumerate(all_machines)
    }

    figure_width = max(8.0, min(20.0, 1.8 * max_machine_length + 4.0))
    figure_height = max(4.5, min(18.0, 1.1 * len(all_machines) + 2.0))
    fig, ax = plt.subplots(figsize=(figure_width, figure_height))

    def draw_arrow(source, target, color, width, alpha, rad, linestyle="-"):
        if source not in positions or target not in positions:
            return

        ax.annotate(
            "",
            xy=positions[target],
            xytext=positions[source],
            arrowprops={
                "arrowstyle": "->",
                "color": color,
                "lw": width,
                "alpha": alpha,
                "shrinkA": 18,
                "shrinkB": 18,
                "connectionstyle": f"arc3,rad={rad}",
                "linestyle": linestyle,
            },
            zorder=1,
        )

    for machine, operations in ordered_operations_by_machine.items():
        if operations:
            x_start, y_pos = machine_positions[machine]
            x_end = len(operations)
            ax.plot(
                [x_start + 0.2, x_end + 0.25],
                [y_pos, y_pos],
                color=machine_colors[machine],
                lw=5.0,
                alpha=0.18,
                solid_capstyle="round",
                zorder=0,
            )

        for source, target in zip(operations, operations[1:]):
            ax.annotate(
                "",
                xy=positions[target],
                xytext=positions[source],
                arrowprops={
                    "arrowstyle": "->",
                    "color": machine_colors[machine],
                    "lw": 2.0,
                    "alpha": 0.9,
                    "shrinkA": 18,
                    "shrinkB": 18,
                    "connectionstyle": "arc3,rad=0.0",
                },
                zorder=1,
            )

    for job in jobs:
        operations = instance.jobs[job]
        for source, target in zip(operations, operations[1:]):
            draw_arrow(
                source,
                target,
                color="0.45",
                width=1.1,
                alpha=0.65,
                rad=0.2,
                linestyle="--",
            )

    for machine, (x_pos, y_pos) in machine_positions.items():
        ax.scatter(
            [x_pos],
            [y_pos],
            s=1450,
            marker="s",
            color=machine_colors[machine],
            edgecolor="black",
            linewidth=1.2,
            zorder=3,
        )
        ax.text(
            x_pos,
            y_pos,
            f"M{machine}",
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            zorder=4,
        )

    for operation, (x_pos, y_pos) in positions.items():
        job = operation_jobs[operation]
        ax.scatter(
            [x_pos],
            [y_pos],
            s=1250,
            marker="o",
            color=job_colors[job],
            edgecolor="black",
            linewidth=1.1,
            zorder=3,
        )
        ax.text(
            x_pos,
            y_pos,
            labels.get(operation, f"O{operation}"),
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
            zorder=4,
        )

    makespan = float(variables["C_max"].X)
    objective = float(model.ObjVal)
    title = title or "Gurobi nonlinear A graph"
    ax.set_title(
        f"{title}\nObjective {objective:.2f} | C_max {makespan:.2f}",
        fontsize=12,
    )
    ax.set_xlabel("Machine sequence from A variables")
    ax.set_ylabel("Machine")
    ax.set_xticks(range(0, max_machine_length + 1))
    ax.set_xticklabels(["M"] + [str(idx) for idx in range(1, max_machine_length + 1)])
    ax.set_yticks([machine_positions[machine][1] for machine in all_machines])
    ax.set_yticklabels([f"Machine {machine}" for machine in all_machines])
    ax.set_xlim(-0.6, max_machine_length + 0.6)
    ax.set_ylim(0.5, len(all_machines) + 0.5)
    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor="0.80",
            markeredgecolor="black",
            markersize=11,
            label="Machine node",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="0.80",
            markeredgecolor="black",
            markersize=11,
            label="Operation node",
        ),
        Line2D([0], [0], color="0.45", lw=1.1, ls="--", label="Job order"),
    ]
    legend_handles.extend(
        Line2D(
            [0],
            [0],
            color=machine_colors[machine],
            lw=2.0,
            label=f"A sequence M{machine}",
        )
        for machine in all_machines
    )
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=min(4, len(legend_handles)),
        frameon=False,
    )

    fig.tight_layout()
    fig.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return filename


def build_and_solve(
    instance_name="i5_k5_o5-8_1",
    instance_path=None,
    solution_path=None,
    time_limit=300.0,
    write_solution=True,
    verbose=True,
    **build_kwargs,
):
    if instance_path is None:
        load_generated_instance = importlib.import_module(
            "01_generator.instance_generator"
        ).load_generated_instance
        instance = load_generated_instance(instance_name)
    else:
        instance_path = Path(instance_path)
        if not instance_path.exists():
            raise FileNotFoundError(f"Instance not found: {instance_path}")
        with open(instance_path, "rb") as file:
            instance = pickle.load(file)

    if solution_path is None:
        solution_path = (
            ROOT_DIR
            / "02_data"
            / "fjsp_solutions"
            / "gurobi_nonlinear"
            / f"solution_{instance_name}_gurobi_nonlinear.txt"
        )
    else:
        solution_path = Path(solution_path)

    model = gp.Model(f"FJSP Gurobi Nonlinear {instance_name}")
    if time_limit is not None and time_limit > 0:
        model.Params.TimeLimit = float(time_limit)
        if verbose:
            print(f"[info] Gurobi time limit: {time_limit}s")

    model, variables = build_fjsp(model, instance, **build_kwargs)

    if verbose:
        print(
            f"[info] Loese FJSP-MINLP: instance={instance_name}, "
            f"operations={len(instance.real_operations)}, "
            f"machines={instance.num_machines}"
        )

    model.optimize()

    status = STATUS_NAMES.get(model.Status, str(model.Status))
    result = {"status": status, "instance": instance_name}
    print(f"\n=== Gurobi Status: {status} ===")

    if model.SolCount > 0:
        objective = float(model.ObjVal)
        makespan = float(variables["C_max"].X)
        total_failure_delay = None
        if variables["constraint_type"] == CONSTRAINT_WEIBULL:
            total_failure_delay = float(
                variables["total_failure_delay"].getValue()
            )
        print(f"Objective:      {objective:.4f}")
        print(f"Makespan:       {makespan:.4f}")
        if total_failure_delay is not None:
            print(f"Total expected failure delay: {total_failure_delay:.4f}")

        if write_solution:
            solution_path.parent.mkdir(parents=True, exist_ok=True)
            write_solution_file(model, variables, instance, solution_path)
            print(f"Solution written to: {solution_path}")

        result.update(
            {
                "objective": objective,
                "makespan": makespan,
                "total_failure_delay": total_failure_delay,
                "solution_path": str(solution_path) if write_solution else None,
            }
        )
    else:
        print("Keine zulaessige Loesung gefunden.")

    return result, model


def _parse_main_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Build and solve the nonlinear Gurobi FJSP MINLP."
    )
    parser.add_argument(
        "--instance-name",
        default="i5_k5_o5-8_1",
        help="Instance name under 02_data/fjsp_instances without .pkl suffix.",
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
        help="Gurobi time limit in seconds. Use 0 for no limit.",
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
