"""Shared direct-predecessor reliability graph definitions.

The file name is retained to avoid fragile import-path migrations.  Setup
times and operation types are no longer part of the model.  Training-data
generation, the exact nonlinear model and the embedded GNN all use the same
machine-age recursion and processing-time transition shock defined here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import gurobipy as gp
from gurobipy import GRB


RELIABILITY_GRAPH_SCHEMA = "direct_machine_predecessor_transition_hazard_v1"
RELIABILITY_GNN_GRAPH_SCHEMA = (
    "direct_predecessor_processing_transition_edge_features_v1"
)
RELIABILITY_GNN_GRAPH_SCHEMA_WITH_JOB_EDGES = (
    "direct_machine_and_job_predecessor_edge_features_v1"
)
RELIABILITY_GNN_GRAPH_SCHEMA_JOB_ONLY = (
    "direct_job_predecessor_edge_features_v1"
)
RELIABILITY_GNN_OUTPUT_HEAD = (
    "direct_predecessor_edge_conditioned_per_node_probability_relu_"
    "clipped_to_one_times_selected_repair_duration_then_sum"
)
RELIABILITY_NODE_FEATURE_NAMES = [
    "processing_time_over_eta",
    "machine_age_over_eta",
    "weibull_beta_over_5",
]
RELIABILITY_EDGE_FEATURE_NAMES = [
    "processing_time_jump_over_machine_max",
    "predecessor_processing_time_over_eta",
]
RELIABILITY_JOB_EDGE_FEATURE_NAME = "is_job_precedence"


def reliability_edge_feature_names(
    include_job_precedence_edges=False,
    include_machine_predecessor_edges=True,
):
    names = (
        list(RELIABILITY_EDGE_FEATURE_NAMES)
        if include_machine_predecessor_edges
        else []
    )
    if include_job_precedence_edges:
        names.append(RELIABILITY_JOB_EDGE_FEATURE_NAME)
    return names


@dataclass(frozen=True)
class ReliabilityGraphConfig:
    enabled: bool = True
    beta: float = 2.0
    transition_gamma: float = 0.05


def normalize_reliability_graph_config(
    config: ReliabilityGraphConfig | dict | None = None,
    **overrides,
) -> ReliabilityGraphConfig:
    if isinstance(config, ReliabilityGraphConfig):
        values = asdict(config)
    else:
        values = dict(config or {})
    aliases = {
        "shape": "beta",
        "shape_beta": "beta",
        "gamma": "transition_gamma",
    }
    normalized = {
        aliases.get(key, key): value
        for key, value in {**values, **overrides}.items()
    }
    allowed = set(ReliabilityGraphConfig.__dataclass_fields__)
    unknown = set(normalized) - allowed
    if unknown:
        raise ValueError(
            f"Unknown reliability-graph parameters: {sorted(unknown)}"
        )
    result = ReliabilityGraphConfig(**normalized)
    if not result.enabled:
        raise ValueError("reliability_graph.enabled must be true.")
    if float(result.beta) <= 0.0:
        raise ValueError("reliability_graph.beta must be positive.")
    if float(result.transition_gamma) < 0.0:
        raise ValueError(
            "reliability_graph.transition_gamma must be nonnegative."
        )
    return ReliabilityGraphConfig(
        enabled=True,
        beta=float(result.beta),
        transition_gamma=float(result.transition_gamma),
    )


def reliability_graph_config_dict(config=None) -> dict:
    return asdict(normalize_reliability_graph_config(config))


def reliability_node_feature_names(_config=None) -> list[str]:
    return list(RELIABILITY_NODE_FEATURE_NAMES)


def fixed_machine_multiedges(instance, operations):
    operation_to_index = {
        operation: index for index, operation in enumerate(operations)
    }
    return [
        (operation_to_index[source], operation_to_index[target], machine)
        for source in operations
        for target in operations
        if source != target
        for machine in sorted(
            set(instance.eligible_machines[source])
            & set(instance.eligible_machines[target])
        )
    ]


def machine_processing_max(instance, operations=None) -> dict[int, float]:
    operations = list(operations or instance.real_operations)
    result = {}
    for machine in range(instance.num_machines):
        values = [
            float(instance.processing_times[operation, machine])
            for operation in operations
            if machine in instance.eligible_machines[operation]
        ]
        if not values:
            raise ValueError(f"Machine {machine} has no eligible operation.")
        result[machine] = max(values)
    return result


def transition_edge_values(
    instance,
    source,
    target,
    machine,
    eta,
    processing_max=None,
) -> tuple[float, float]:
    processing_max = processing_max or machine_processing_max(instance)
    source_processing = float(instance.processing_times[source, machine])
    target_processing = float(instance.processing_times[target, machine])
    jump = abs(target_processing - source_processing) / processing_max[machine]
    return jump, source_processing / float(eta[machine])


def _directed_order(A_plus, A_minus, source, target, machine):
    if source < target:
        return A_plus[source, target, machine]
    return A_minus[target, source, machine]


def add_order_activations(model, variables):
    """Create directed pair-order indicators for selected machine pairs."""
    if variables.get("A_plus") is not None:
        return variables["A_plus"], variables["A_minus"]
    Y, X = variables["Y"], variables["X"]
    A_plus = model.addVars(
        variables["X_index"], lb=0.0, ub=1.0,
        vtype=GRB.CONTINUOUS, name="A_plus"
    )
    A_minus = model.addVars(
        variables["X_index"], lb=0.0, ub=1.0,
        vtype=GRB.CONTINUOUS, name="A_minus"
    )
    for operation_i, operation_j, machine in variables["X_index"]:
        yi = Y[operation_i, machine]
        yj = Y[operation_j, machine]
        x = X[operation_i, operation_j, machine]
        ap = A_plus[operation_i, operation_j, machine]
        am = A_minus[operation_i, operation_j, machine]
        model.addConstr(ap <= yi)
        model.addConstr(ap <= yj)
        model.addConstr(ap <= x)
        model.addConstr(ap >= yi + yj + x - 2)
        model.addConstr(am <= yi)
        model.addConstr(am <= yj)
        model.addConstr(am <= 1 - x)
        model.addConstr(am >= yi + yj - x - 1)
    variables.update({
        "A_plus": A_plus,
        "A_minus": A_minus,
        "A_index": list(variables["X_index"]),
    })
    return A_plus, A_minus


def add_reliability_graph_variables(
    model,
    variables,
    instance,
    config=None,
    tighten_age_bounds=False,
):
    """Add direct machine predecessors, transition load and machine age."""
    cfg = normalize_reliability_graph_config(config)
    operations = list(variables["real_operations"])
    machines = list(variables["machines"])
    Y = variables["Y"]
    A_plus, A_minus = add_order_activations(model, variables)

    U_index = [
        (source, target, machine)
        for source in operations
        for target in operations
        if source != target
        for machine in sorted(
            set(instance.eligible_machines[source])
            & set(instance.eligible_machines[target])
        )
    ]
    U = model.addVars(U_index, vtype=GRB.BINARY, name="U_direct")
    first = model.addVars(
        variables["Y_index"], vtype=GRB.BINARY, name="machine_first"
    )
    last = model.addVars(
        variables["Y_index"], vtype=GRB.BINARY, name="machine_last"
    )
    machine_used = model.addVars(
        machines, vtype=GRB.BINARY, name="machine_used"
    )
    incoming = {
        (target, machine): [
            (source, U[source, target, machine])
            for source in operations
            if (source, target, machine) in U
        ]
        for target, machine in variables["Y_index"]
    }
    outgoing = {
        (source, machine): [
            (target, U[source, target, machine])
            for target in operations
            if (source, target, machine) in U
        ]
        for source, machine in variables["Y_index"]
    }
    for source, target, machine in U_index:
        model.addConstr(
            U[source, target, machine]
            <= _directed_order(A_plus, A_minus, source, target, machine),
            name=f"direct_predecessor_order[{source},{target},{machine}]",
        )
    for operation, machine in variables["Y_index"]:
        model.addConstr(
            gp.quicksum(value for _source, value in incoming[operation, machine])
            + first[operation, machine]
            == Y[operation, machine],
            name=f"direct_in_degree[{operation},{machine}]",
        )
        model.addConstr(
            gp.quicksum(value for _target, value in outgoing[operation, machine])
            + last[operation, machine]
            == Y[operation, machine],
            name=f"direct_out_degree[{operation},{machine}]",
        )
        model.addConstr(first[operation, machine] <= machine_used[machine])
        model.addConstr(last[operation, machine] <= machine_used[machine])
        model.addConstr(machine_used[machine] >= Y[operation, machine])
    for machine in machines:
        eligible = [
            operation for operation in operations
            if machine in instance.eligible_machines[operation]
        ]
        model.addConstr(
            machine_used[machine]
            <= gp.quicksum(Y[operation, machine] for operation in eligible)
        )
        model.addConstr(
            gp.quicksum(first[operation, machine] for operation in eligible)
            == machine_used[machine],
            name=f"one_machine_first[{machine}]",
        )
        model.addConstr(
            gp.quicksum(last[operation, machine] for operation in eligible)
            == machine_used[machine],
            name=f"one_machine_last[{machine}]",
        )

    initial_age = variables["machine_initial_age"]
    successors = {operation: set() for operation in operations}
    for target in operations:
        for source in instance.predecessors.get(target, []):
            if source in successors:
                successors[source].add(target)

    descendant_cache = {}

    def descendants(operation, visiting=None):
        if operation in descendant_cache:
            return descendant_cache[operation]
        visiting = set(visiting or ())
        if operation in visiting:
            raise ValueError("Job precedence graph must be acyclic.")
        visiting.add(operation)
        result = set(successors[operation])
        for successor in successors[operation]:
            result.update(descendants(successor, visiting))
        descendant_cache[operation] = result
        return result

    age_upper_by_machine = {
        machine: initial_age[machine]
        + sum(
            float(instance.processing_times[operation, machine])
            for operation in operations
            if machine in instance.eligible_machines[operation]
        )
        for machine in machines
    }
    R, R_bounds = {}, {}
    for operation, machine in variables["Y_index"]:
        # R is the machine age immediately before ``operation`` starts.  If
        # the operation is assigned to this machine, its own processing time
        # cannot already be part of that age.  Removing it gives a safe,
        # operation-specific bound and strengthens every downstream neural
        # pre-activation bound without changing the feasible schedules.
        upper = age_upper_by_machine[machine]
        if tighten_age_bounds:
            upper -= float(instance.processing_times[operation, machine])
            # A technological successor cannot be processed before this
            # operation and therefore cannot contribute to its machine age.
            upper -= sum(
                float(instance.processing_times[successor, machine])
                for successor in descendants(operation)
                if machine in instance.eligible_machines[successor]
            )
        R_bounds[operation, machine] = upper
        R[operation, machine] = model.addVar(
            lb=0.0, ub=upper, vtype=GRB.CONTINUOUS,
            name=f"R[{operation},{machine}]",
        )
        model.addConstr(R[operation, machine] <= upper * Y[operation, machine])
        model.addConstr(
            R[operation, machine] - initial_age[machine]
            <= upper * (1.0 - first[operation, machine])
        )
        model.addConstr(
            R[operation, machine] - initial_age[machine]
            >= -upper * (1.0 - first[operation, machine])
        )
    max_processing = max(
        float(instance.processing_times[operation, machine])
        for operation, machine in variables["Y_index"]
    )
    for source, target, machine in U_index:
        expression = (
            R[source, machine]
            + float(instance.processing_times[source, machine])
        )
        big_m = age_upper_by_machine[machine] + max_processing
        model.addConstr(
            R[target, machine] - expression
            <= big_m * (1.0 - U[source, target, machine])
        )
        model.addConstr(
            R[target, machine] - expression
            >= -big_m * (1.0 - U[source, target, machine])
        )

    processing_max = machine_processing_max(instance, operations)
    transition_edge_load = {
        edge: transition_edge_values(
            instance, *edge, variables["weibull_eta"], processing_max
        )[0]
        for edge in U_index
    }
    transition_load = model.addVars(
        operations, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS,
        name="transition_load"
    )
    for operation in operations:
        model.addConstr(
            transition_load[operation]
            == gp.quicksum(
                transition_edge_load[source, target, machine]
                * U[source, target, machine]
                for source, target, machine in U_index
                if target == operation
            ),
            name=f"transition_load_def[{operation}]",
        )

    variables.update({
        "instance": instance,
        "U": U,
        "U_index": U_index,
        "machine_first": first,
        "machine_last": last,
        "machine_used": machine_used,
        "R": R,
        "R_bounds": R_bounds,
        "transition_load": transition_load,
        "transition_edge_load": transition_edge_load,
        "machine_processing_max": processing_max,
        "age_bound_tightening": bool(tighten_age_bounds),
        "reliability_graph_config": reliability_graph_config_dict(cfg),
        "reliability_graph_schema": RELIABILITY_GRAPH_SCHEMA,
    })
    return variables
