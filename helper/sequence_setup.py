"""Define the reliability graph shared by data generation and GNN embedding.

The graph combines fixed job-precedence arcs with immediate predecessor arcs on
selected machines. This module also owns the normalized physical node features
and the Gurobi variables that reconstruct direct machine sequences.
"""

import math

import gurobipy as gp
from gurobipy import GRB


RELIABILITY_GNN_GRAPH_SCHEMA = (
    "direct_machine_and_job_predecessor_physical_features_v10"
)
RELIABILITY_GNN_OUTPUT_HEAD = (
    "per_job_local_midpoint_buffer_relu_v6"
)
RELIABILITY_QUADRATURE_POINTS = 12
RELIABILITY_SERVICE_SCOPE = "job"
RELIABILITY_GRAPH_CONFIG = {
    "quadrature_points": RELIABILITY_QUADRATURE_POINTS,
    "service_scope": RELIABILITY_SERVICE_SCOPE,
    "gnn_safety_margin": 0.0,
}
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


def reliability_graph_config_dict() -> dict:
    """Return an independent copy of the fixed graph metadata contract.

    The copy is serialized with datasets and model artifacts for compatibility
    checks without exposing the module-level dictionary to mutation.
    """
    return dict(RELIABILITY_GRAPH_CONFIG)


def reliability_node_feature_names() -> list[str]:
    """Return the ordered feature names expected by training and embedding.

    Returns:
        A new list whose order matches :func:`local_buffer_node_features`.
    """
    return list(RELIABILITY_NODE_FEATURE_NAMES)


def directed_machine_order(A_plus, A_minus, source, target, machine):
    """Return the orientation-specific precedence activation for a pair.

    Pairwise order variables are stored only for increasing operation IDs;
    this helper selects the correct forward or reverse activation.
    """
    if source < target:
        return A_plus[source, target, machine]
    return A_minus[target, source, machine]


def add_order_activations(model, variables):
    """Linearize assignment-aware orientations of pairwise machine orders.

    Returns:
        Forward and reverse activation variable dictionaries. An activation is
        one only when both operations select the machine and the corresponding
        pairwise order orientation is active.
    """
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
):
    """Create direct selected-machine predecessor variables ``U``.

    Assignment-aware order activations restrict candidate arcs. Degree
    constraints give every assigned operation either one predecessor or first
    status and either one successor or last status, forming one path on every
    used machine.

    Side Effects:
        Adds order activations, direct-edge variables and their index set to the
        shared ``variables`` dictionary.
    """
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
    })
