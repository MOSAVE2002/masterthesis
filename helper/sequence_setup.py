"""Direct-predecessor graph shared by training and Gurobi embedding."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import gurobipy as gp
from gurobipy import GRB


RELIABILITY_GRAPH_SCHEMA = "job_local_midpoint_buffer_v7"
RELIABILITY_GNN_GRAPH_SCHEMA = (
    "direct_machine_and_job_predecessor_physical_features_v10"
)
RELIABILITY_GNN_OUTPUT_HEAD = (
    "per_job_local_midpoint_buffer_relu_v6"
)
RELIABILITY_NODE_FEATURE_NAMES = [
    "nominal_midpoint_over_weibull_alpha",
    "repair_rate_times_weibull_alpha_over_10",
    "weibull_beta_over_5",
    "mean_repair_duration_over_60ze",
]


def local_buffer_node_features(midpoint, alpha, beta, repair_rate):
    """Four formula-aligned inputs; times remain in ZE, rates in 1/ZE.

    No realized failures or labels are inputs. The constant 60 is a fixed
    reference duration in ZE, independent of due dates and instance size.
    """
    t, a, b, rate = map(float, (midpoint, alpha, beta, repair_rate))
    if (not all(map(math.isfinite, (t, a, b, rate)))
            or t < 0 or a <= 0 or b <= 1 or rate <= 0):
        raise ValueError("Require finite t>=0, alpha>0, beta>1, repair_rate>0.")
    return [t / a, rate * a / 10.0, b / 5.0, 1.0 / (60.0 * rate)]


@dataclass(frozen=True)
class ReliabilityGraphConfig:
    quadrature_points: int = 12
    service_scope: str = "job"
    gnn_safety_margin: float = 0.0


def normalize_reliability_graph_config(
    config: ReliabilityGraphConfig | dict | None = None,
    **overrides,
) -> ReliabilityGraphConfig:
    if isinstance(config, ReliabilityGraphConfig):
        values = asdict(config)
    else:
        values = dict(config or {})
    values.update(overrides)
    allowed = set(ReliabilityGraphConfig.__dataclass_fields__)
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(
            f"Unknown reliability-graph parameters: {sorted(unknown)}"
        )
    result = ReliabilityGraphConfig(**values)
    if int(result.quadrature_points) < 4:
        raise ValueError("quadrature_points must be at least 4.")
    service_scope = str(result.service_scope).strip().lower()
    if service_scope not in {"all", "job"}:
        raise ValueError("service_scope must be 'all' or 'job'.")
    if float(result.gnn_safety_margin) < 0.0:
        raise ValueError("gnn_safety_margin must be nonnegative.")
    return ReliabilityGraphConfig(
        quadrature_points=int(result.quadrature_points),
        service_scope=service_scope,
        gnn_safety_margin=float(result.gnn_safety_margin),
    )


def reliability_graph_config_dict(config=None) -> dict:
    return asdict(normalize_reliability_graph_config(config))


def reliability_node_feature_names(_config=None) -> list[str]:
    return list(RELIABILITY_NODE_FEATURE_NAMES)


def directed_machine_order(A_plus, A_minus, source, target, machine):
    """Return the active precedence gate for one directed machine pair."""
    if source < target:
        return A_plus[source, target, machine]
    return A_minus[target, source, machine]


def add_order_activations(model, variables):
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
    return A_plus, A_minus


def add_reliability_graph_variables(
    model,
    variables,
    instance,
    config=None,
):
    """Create U_ijk for immediate machine predecessors."""
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
            U[source, target, machine]
            for source in operations
            if (source, target, machine) in U
        ]
        for target, machine in variables["Y_index"]
    }
    outgoing = {
        (source, machine): [
            U[source, target, machine]
            for target in operations
            if (source, target, machine) in U
        ]
        for source, machine in variables["Y_index"]
    }
    for source, target, machine in U_index:
        model.addConstr(
            U[source, target, machine]
            <= directed_machine_order(
                A_plus, A_minus, source, target, machine
            ),
            name=f"direct_predecessor_order[{source},{target},{machine}]",
        )
    for operation, machine in variables["Y_index"]:
        model.addConstr(
            gp.quicksum(incoming[operation, machine])
            + first[operation, machine]
            == Y[operation, machine],
            name=f"direct_in_degree[{operation},{machine}]",
        )
        model.addConstr(
            gp.quicksum(outgoing[operation, machine])
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
        if not eligible:
            model.addConstr(machine_used[machine] == 0)
            continue
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
    variables.update({
        "A_plus": A_plus,
        "A_minus": A_minus,
        "U": U,
        "U_index": U_index,
        "reliability_graph_config": reliability_graph_config_dict(cfg),
        "reliability_graph_schema": RELIABILITY_GRAPH_SCHEMA,
    })
    return variables
