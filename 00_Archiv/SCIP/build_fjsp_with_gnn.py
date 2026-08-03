import importlib
import json
import math
import sys
from pathlib import Path

import torch
from pyscipopt import Model, quicksum

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

_base_fjsp = importlib.import_module("00_Archiv.SCIP.build_fjsp")
build_base_fjsp = _base_fjsp.build_fjsp
_nonlinear_fjsp = importlib.import_module("00_Archiv.SCIP.build_fjsp_with_nonlinear")
_directed_a = _nonlinear_fjsp._directed_a
_failure_cost = _nonlinear_fjsp._failure_cost
_first_machine_parameter = _nonlinear_fjsp._first_machine_parameter
from helper.surrogate_constraint import (
    CONSTRAINT_WEIBULL,
    effective_constraint_budget,
    model_stem,
    target_column,
    validate_constraint_type,
)


DEFAULT_MODEL_SEED = 42
DEFAULT_MODEL_DIR = ROOT_DIR / "04_GraphNeuralNetworks" / "models"
GRAPH_MODE_FIXED_CANDIDATE = "fixed_candidate"
LEGACY_FEATURE_NAMES = [
    "chosen_processing_time_over_H",
    "eligible_machine_ratio",
    "assigned_machine_ratio",
    "operation_position_ratio",
    "start_time_over_H",
]
WEIBULL_FEATURE_NAMES = [
    "processing_time_over_eta",
    "machine_age_over_eta",
    "weibull_beta_over_5",
    "failure_cost_over_20",
    "eligible_machine_ratio",
    "assigned_machine_ratio",
    "operation_position_ratio",
    "start_time_over_H",
]


def _default_model_path(seed=DEFAULT_MODEL_SEED, constraint_type=CONSTRAINT_WEIBULL):
    return DEFAULT_MODEL_DIR / f"{model_stem(constraint_type, seed)}.pt"


def _default_metadata_path(seed=DEFAULT_MODEL_SEED, constraint_type=CONSTRAINT_WEIBULL):
    return DEFAULT_MODEL_DIR / f"{model_stem(constraint_type, seed)}_meta.json"


def _resolve_path(path_value, default_path):
    if path_value in (None, ""):
        return Path(default_path)

    path = Path(path_value)
    if not path.is_absolute():
        path = ROOT_DIR / path
    return path


def _load_metadata(metadata_path):
    if not metadata_path.exists():
        raise FileNotFoundError(f"GNN metadata file not found: {metadata_path}")

    with metadata_path.open(encoding="utf-8") as file:
        metadata = json.load(file)

    graph_mode = metadata.get("graph_mode", GRAPH_MODE_FIXED_CANDIDATE)
    if graph_mode != GRAPH_MODE_FIXED_CANDIDATE:
        raise ValueError(
            "The embedded SCIP GNN solver currently supports only "
            "graph_mode='fixed_candidate'."
        )

    aggregation = metadata.get("aggregation", "sum")
    if aggregation not in {"sum", "mean"}:
        raise ValueError(
            "The embedded SCIP GNN solver supports GraphSAGE "
            "aggregation='sum' or 'mean'."
        )

    return metadata


def _load_state_dict(model_path):
    if not model_path.exists():
        raise FileNotFoundError(f"GNN model file not found: {model_path}")

    return {
        key: value.detach().cpu().numpy()
        for key, value in torch.load(model_path, map_location="cpu").items()
    }


def _operation_job_positions(jobs):
    positions = {}
    sorted_jobs = sorted(jobs)
    for job_idx, job in enumerate(sorted_jobs):
        job_operations = jobs[job]
        for position_idx, operation in enumerate(job_operations):
            positions[operation] = (
                job_idx,
                position_idx,
                len(job_operations),
                len(sorted_jobs),
            )
    return positions


def _build_fixed_candidate_neighbors(
    instance, operations, max_machine_neighbors=None
):
    operation_to_idx = {operation: idx for idx, operation in enumerate(operations)}
    neighbors_by_target = {idx: [] for idx in range(len(operations))}
    edges = set()

    def add_edge(source, target):
        source_idx = operation_to_idx[source]
        target_idx = operation_to_idx[target]
        if source_idx == target_idx:
            return
        edges.add((source_idx, target_idx))

    for job_operations in instance.jobs.values():
        for source, target in zip(job_operations, job_operations[1:]):
            add_edge(source, target)

    for machine in range(instance.num_machines):
        candidate_operations = [
            operation
            for operation in operations
            if machine in instance.eligible_machines[operation]
        ]
        for idx_i, operation_i in enumerate(candidate_operations):
            following = candidate_operations[idx_i + 1:]
            if max_machine_neighbors not in (None, 0):
                following = following[:int(max_machine_neighbors)]
            for operation_j in following:
                add_edge(operation_i, operation_j)
                add_edge(operation_j, operation_i)

    for source_idx, target_idx in sorted(edges):
        neighbors_by_target[target_idx].append(source_idx)

    return neighbors_by_target, sorted(edges)


def _linear_expr(coefficients, values, bias=0.0):
    expr = float(bias)
    lower = float(bias)
    upper = float(bias)

    for coefficient, value in zip(coefficients, values):
        coefficient = float(coefficient)
        if abs(coefficient) <= 1e-12:
            continue

        value_expr, value_lower, value_upper = value
        expr += coefficient * value_expr
        if coefficient >= 0.0:
            lower += coefficient * value_lower
            upper += coefficient * value_upper
        else:
            lower += coefficient * value_upper
            upper += coefficient * value_lower

    return expr, lower, upper


def _sum_features(feature_vectors, indices, feature_idx):
    if not indices:
        return 0.0, 0.0, 0.0

    expr = quicksum(feature_vectors[idx][feature_idx][0] for idx in indices)
    lower = sum(feature_vectors[idx][feature_idx][1] for idx in indices)
    upper = sum(feature_vectors[idx][feature_idx][2] for idx in indices)
    return expr, lower, upper


def _add_relu(model, expression, lower, upper, name):
    lower = min(float(lower), 0.0) if abs(lower) < 1e-9 else float(lower)
    upper = max(float(upper), 0.0) if abs(upper) < 1e-9 else float(upper)

    pre_activation = model.addVar(
        vtype="C",
        lb=lower,
        ub=upper,
        name=f"{name}_pre",
    )
    model.addCons(pre_activation == expression, name=f"{name}_pre_def")

    if upper <= 0.0:
        activation = model.addVar(vtype="C", lb=0.0, ub=0.0, name=name)
        return (activation, 0.0, 0.0), pre_activation

    if lower >= 0.0:
        activation = model.addVar(vtype="C", lb=lower, ub=upper, name=name)
        model.addCons(activation == pre_activation, name=f"{name}_relu_pos")
        return (activation, lower, upper), pre_activation

    activation = model.addVar(vtype="C", lb=0.0, ub=upper, name=name)
    active = model.addVar(vtype="B", name=f"{name}_relu_active")

    model.addCons(activation >= pre_activation, name=f"{name}_relu_lb_pre")
    model.addCons(activation <= pre_activation - lower * (1 - active), name=f"{name}_relu_ub_pre")
    model.addCons(activation <= upper * active, name=f"{name}_relu_ub_active")

    return (activation, 0.0, upper), pre_activation


def _add_graphsage_layer(
    model,
    state_dict,
    layer_name,
    input_vectors,
    neighbors_by_target,
    aggregation,
):
    lin_l_weight = state_dict[f"{layer_name}.lin_l.weight"]
    lin_l_bias = state_dict[f"{layer_name}.lin_l.bias"]
    lin_r_weight = state_dict[f"{layer_name}.lin_r.weight"]

    output_vectors = []
    pre_activation_vectors = []
    input_size = lin_l_weight.shape[1]

    for node_idx, root_features in enumerate(input_vectors):
        neighbor_indices = neighbors_by_target[node_idx]
        neighbor_features = [
            _sum_features(input_vectors, neighbor_indices, feature_idx)
            for feature_idx in range(input_size)
        ]
        if aggregation == "mean" and neighbor_indices:
            degree = len(neighbor_indices)
            neighbor_features = [
                (value[0] / degree, value[1] / degree, value[2] / degree)
                for value in neighbor_features
            ]

        node_outputs = []
        node_pre_activations = []
        for channel_idx in range(lin_l_weight.shape[0]):
            neighbor_expr = _linear_expr(
                lin_l_weight[channel_idx],
                neighbor_features,
                bias=lin_l_bias[channel_idx],
            )
            root_expr = _linear_expr(lin_r_weight[channel_idx], root_features)
            expression = neighbor_expr[0] + root_expr[0]
            lower = neighbor_expr[1] + root_expr[1]
            upper = neighbor_expr[2] + root_expr[2]
            activation, pre_activation = _add_relu(
                model,
                expression,
                lower,
                upper,
                name=f"{layer_name}_node{node_idx}_h{channel_idx}",
            )
            node_outputs.append(activation)
            node_pre_activations.append(pre_activation)

        output_vectors.append(node_outputs)
        pre_activation_vectors.append(node_pre_activations)

    return output_vectors, pre_activation_vectors


def _selected_processing_time_expr(Y, instance, operation):
    return quicksum(
        Y[operation, machine] * float(instance.processing_times[operation, machine])
        for machine in instance.eligible_machines[operation]
    )


def _selected_processing_time_feature(Y, instance, operation, H):
    processing_times = [
        float(instance.processing_times[operation, machine])
        for machine in instance.eligible_machines[operation]
    ]
    return (
        _selected_processing_time_expr(Y, instance, operation) / H,
        min(processing_times) / H,
        max(processing_times) / H,
    )


def _selected_machine_ratio_feature(Y, instance, operation):
    denominator = max(instance.num_machines - 1, 1)
    ratios = [
        float(machine) / denominator
        for machine in instance.eligible_machines[operation]
    ]
    expr = quicksum(
        Y[operation, machine] * (float(machine) / denominator)
        for machine in instance.eligible_machines[operation]
    )
    return expr, min(ratios), max(ratios)


def _constant_feature(value):
    value = float(value)
    return value, value, value


def _add_weibull_age_variables(model, variables, instance):
    machines = variables["machines"]
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
    beta = {
        machine: _first_machine_parameter(
            instance, ("weibull_beta", "beta"), machine, 2.0
        )
        for machine in machines
    }
    failure_costs = {
        (operation, machine): _failure_cost(instance, operation, machine)
        for operation, machine in variables["Y_index"]
    }
    A_plus = {
        idx: model.addVar(
            vtype="C", lb=0.0, ub=1.0,
            name=f"A_plus[{idx[0]},{idx[1]},{idx[2]}]"
        )
        for idx in variables["X_index"]
    }
    A_minus = {
        idx: model.addVar(
            vtype="C", lb=0.0, ub=1.0,
            name=f"A_minus[{idx[0]},{idx[1]},{idx[2]}]"
        )
        for idx in variables["X_index"]
    }
    Y, X = variables["Y"], variables["X"]
    for operation_i, operation_j, machine in variables["X_index"]:
        idx = operation_i, operation_j, machine
        ap, am = A_plus[idx], A_minus[idx]
        yi, yj, x = Y[operation_i, machine], Y[operation_j, machine], X[idx]
        model.addCons(ap <= yi, name=f"gnn_ap_yi[{operation_i},{operation_j},{machine}]")
        model.addCons(ap <= yj, name=f"gnn_ap_yj[{operation_i},{operation_j},{machine}]")
        model.addCons(ap <= x, name=f"gnn_ap_x[{operation_i},{operation_j},{machine}]")
        model.addCons(ap >= yi + yj + x - 2, name=f"gnn_ap_lb[{operation_i},{operation_j},{machine}]")
        model.addCons(am <= yi, name=f"gnn_am_yi[{operation_i},{operation_j},{machine}]")
        model.addCons(am <= yj, name=f"gnn_am_yj[{operation_i},{operation_j},{machine}]")
        model.addCons(am <= 1 - x, name=f"gnn_am_x[{operation_i},{operation_j},{machine}]")
        model.addCons(am >= yi + yj - x - 1, name=f"gnn_am_lb[{operation_i},{operation_j},{machine}]")

    R, R_bounds = {}, {}
    for operation, machine in variables["Y_index"]:
        others = [
            other for other in variables["real_operations"]
            if other != operation and machine in instance.eligible_machines[other]
        ]
        upper = initial_age[machine] + sum(
            float(instance.processing_times[other, machine]) for other in others
        )
        R_bounds[operation, machine] = upper
        R[operation, machine] = model.addVar(
            vtype="C", lb=0.0, ub=upper, name=f"R[{operation},{machine}]"
        )
        predecessor_runtime = quicksum(
            float(instance.processing_times[other, machine])
            * _directed_a(A_plus, A_minus, other, operation, machine)
            for other in others
        )
        model.addCons(
            R[operation, machine]
            == initial_age[machine] * Y[operation, machine] + predecessor_runtime,
            name=f"gnn_machine_age[{operation},{machine}]",
        )
    variables.update({
        "A_plus": A_plus, "A_minus": A_minus,
        "A_index": variables["X_index"], "R": R, "R_bounds": R_bounds,
        "machine_initial_age": initial_age, "weibull_eta": eta,
        "weibull_beta": beta, "failure_costs": failure_costs,
    })


def _selected_weighted_feature(Y, instance, operation, values):
    coefficients = [float(values[machine]) for machine in instance.eligible_machines[operation]]
    expr = quicksum(
        Y[operation, machine] * float(values[machine])
        for machine in instance.eligible_machines[operation]
    )
    return expr, min(coefficients), max(coefficients)


def _build_node_feature_expressions(instance, variables, constraint_type):
    H = float(variables["H"])
    if H <= 0.0:
        raise ValueError("Cannot normalize GNN features because H <= 0.")

    operations = list(variables["real_operations"])
    Y = variables["Y"]
    S = variables["S"]
    operation_positions = _operation_job_positions(variables["jobs"])
    num_machines = max(len(variables["machines"]), 1)

    node_features = []
    for operation in operations:
        _job_idx, position_idx, job_length, _num_jobs = operation_positions[operation]
        if constraint_type == CONSTRAINT_WEIBULL:
            p_values = {
                machine: float(instance.processing_times[operation, machine])
                / variables["weibull_eta"][machine]
                for machine in instance.eligible_machines[operation]
            }
            beta_values = {
                machine: variables["weibull_beta"][machine] / 5.0
                for machine in instance.eligible_machines[operation]
            }
            cost_values = {
                machine: variables["failure_costs"][operation, machine] / 20.0
                for machine in instance.eligible_machines[operation]
            }
            age_expr = quicksum(
                variables["R"][operation, machine]
                / variables["weibull_eta"][machine]
                for machine in instance.eligible_machines[operation]
            )
            age_upper = max(
                variables["R_bounds"][operation, machine]
                / variables["weibull_eta"][machine]
                for machine in instance.eligible_machines[operation]
            )
            node_features.append([
                _selected_weighted_feature(Y, instance, operation, p_values),
                (age_expr, 0.0, age_upper),
                _selected_weighted_feature(Y, instance, operation, beta_values),
                _selected_weighted_feature(Y, instance, operation, cost_values),
                _constant_feature(len(instance.eligible_machines[operation]) / num_machines),
                _selected_machine_ratio_feature(Y, instance, operation),
                _constant_feature(position_idx / max(job_length - 1, 1)),
                (S[operation] / H, 0.0, 1.0),
            ])
        else:
            node_features.append([
                _selected_processing_time_feature(Y, instance, operation, H),
                _constant_feature(len(instance.eligible_machines[operation]) / num_machines),
                _selected_machine_ratio_feature(Y, instance, operation),
                _constant_feature(position_idx / max(job_length - 1, 1)),
                (S[operation] / H, 0.0, 1.0),
            ])

    return node_features


def _add_start_time_variables(model, variables, instance):
    operations = list(variables["real_operations"])
    S = {
        operation: model.addVar(vtype="C", lb=0.0, name=f"S[{operation}]")
        for operation in operations
    }

    for operation in operations:
        model.addCons(
            S[operation]
            == variables["C"][operation]
            - _selected_processing_time_expr(variables["Y"], instance, operation),
            name=f"start_time_def[{operation}]",
        )

    variables["S"] = S
    return S


def _set_var_upper_bound(model, variable, upper_bound):
    try:
        model.chgVarUb(variable, upper_bound)
    except Exception:
        variable.ub = upper_bound


def _add_schedule_upper_bounds(model, variables):
    H = float(variables["H"])
    _set_var_upper_bound(model, variables["C_max"], H)
    for operation in variables["real_operations"]:
        _set_var_upper_bound(model, variables["C"][operation], H)
        _set_var_upper_bound(model, variables["S"][operation], H)


def _add_gnn_prediction(
    model,
    variables,
    instance,
    state_dict,
    metadata,
    constraint_type,
):
    operations = list(variables["real_operations"])
    neighbors_by_target, edges = _build_fixed_candidate_neighbors(
        instance,
        operations,
        metadata.get("max_machine_neighbors"),
    )
    node_features = _build_node_feature_expressions(
        instance, variables, constraint_type
    )
    aggregation = metadata.get("aggregation", "sum")

    hidden_1, hidden_1_pre = _add_graphsage_layer(
        model,
        state_dict,
        "conv1",
        node_features,
        neighbors_by_target,
        aggregation,
    )
    if int(metadata.get("num_graphsage_layers", 2)) == 2:
        hidden_2, hidden_2_pre = _add_graphsage_layer(
            model,
            state_dict,
            "conv2",
            hidden_1,
            neighbors_by_target,
            aggregation,
        )
    else:
        hidden_2, hidden_2_pre = hidden_1, hidden_1_pre

    num_nodes = max(len(operations), 1)
    node_predictions, node_raw_predictions = [], []
    input_skip_weight = state_dict.get("out_input.weight")
    input_skip_bias = state_dict.get("out_input.bias")
    if metadata.get("pooling") == "node_output_global_add_pool":
        for node_idx in range(num_nodes):
            raw_expr, raw_lower, raw_upper = _linear_expr(
                state_dict["out.weight"][0], hidden_2[node_idx],
                bias=state_dict["out.bias"][0],
            )
            if input_skip_weight is not None:
                skip_expr, skip_lower, skip_upper = _linear_expr(
                    input_skip_weight[0], node_features[node_idx],
                    bias=input_skip_bias[0],
                )
                raw_expr += skip_expr
                raw_lower += skip_lower
                raw_upper += skip_upper
            node_prediction, node_raw = _add_relu(
                model, raw_expr, raw_lower, raw_upper,
                name=f"predicted_failure_cost_node{node_idx}",
            )
            node_predictions.append(node_prediction)
            node_raw_predictions.append(node_raw)
        prediction = model.addVar(vtype="C", lb=0.0, name="predicted_failure_cost")
        raw_prediction = model.addVar(
            vtype="C",
            lb=sum(var.getLbGlobal() for var in node_raw_predictions),
            ub=sum(var.getUbGlobal() for var in node_raw_predictions),
            name="predicted_failure_cost_raw",
        )
        model.addCons(
            prediction == quicksum(value[0] for value in node_predictions),
            name="predicted_failure_cost_sum",
        )
        model.addCons(
            raw_prediction == quicksum(node_raw_predictions),
            name="predicted_failure_cost_raw_sum",
        )
        prediction_expr = prediction
    else:
        pooled = []
        for channel_idx in range(len(hidden_2[0])):
            expr = quicksum(
                hidden_2[node_idx][channel_idx][0] for node_idx in range(num_nodes)
            ) / num_nodes
            lower = sum(
                hidden_2[node_idx][channel_idx][1] for node_idx in range(num_nodes)
            ) / num_nodes
            upper = sum(
                hidden_2[node_idx][channel_idx][2] for node_idx in range(num_nodes)
            ) / num_nodes
            pooled.append((expr, lower, upper))
        raw_expr, raw_lower, raw_upper = _linear_expr(
            state_dict["out.weight"][0], pooled,
            bias=state_dict["out.bias"][0],
        )
        prediction, raw_prediction = _add_relu(
            model, raw_expr, raw_lower, raw_upper,
            name="predicted_failure_cost",
        )
        prediction_expr = prediction[0]

    variables.update(
        {
            "predicted_failure_cost": prediction_expr,
            "predicted_constraint_value": prediction_expr,
            "predicted_failure_cost_raw": raw_prediction,
            "predicted_operation_failure_costs": [v[0] for v in node_predictions],
            "predicted_operation_failure_costs_raw": node_raw_predictions,
            "gnn_edges": edges,
            "gnn_node_features": node_features,
            "gnn_hidden_1": hidden_1,
            "gnn_hidden_1_pre": hidden_1_pre,
            "gnn_hidden_2": hidden_2,
            "gnn_hidden_2_pre": hidden_2_pre,
            "gnn_feature_names": metadata["feature_names"],
            "gnn_graph_mode": GRAPH_MODE_FIXED_CANDIDATE,
        }
    )

    return prediction_expr


def build_fjsp(
    fjsp: Model,
    instance,
    model_path=None,
    metadata_path=None,
    model_seed=DEFAULT_MODEL_SEED,
    add_schedule_upper_bounds=True,
    use_gnn_objective=True,
    objective_weight=None,
    constraint_type=CONSTRAINT_WEIBULL,
    weibull_budget=10.0,
    safety_margin=0.0,
    enforce_constraint=True,
    safety_margin_per_operation=0.0,
    scale_weibull_budget=False,
    weibull_budget_per_operation=0.4,
):
    """
    Build the FJSP and enforce the configured GNN surrogate as a budget.

    ``use_gnn_objective`` and ``objective_weight`` are retained only for
    backwards-compatible callers. The objective is always ``C_max``.
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
    safety_margin = float(safety_margin)
    safety_margin_per_operation = float(safety_margin_per_operation)
    effective_safety_margin = (
        safety_margin
        + safety_margin_per_operation * len(instance.real_operations)
    )
    if constraint_budget < 0.0 or effective_safety_margin < 0.0:
        raise ValueError("Constraint budget and safety margin must be nonnegative.")

    if not add_schedule_upper_bounds:
        raise ValueError(
            "SCIP GNN embedding needs finite schedule bounds for exact ReLU "
            "linearisation. Keep add_schedule_upper_bounds=True."
        )

    model_seed = int(model_seed)
    model_path = _resolve_path(
        model_path, _default_model_path(model_seed, constraint_type)
    )
    metadata_path = _resolve_path(
        metadata_path, _default_metadata_path(model_seed, constraint_type)
    )
    metadata = _load_metadata(metadata_path)
    expected_target = target_column(constraint_type)
    if metadata.get("target_column") != expected_target:
        raise ValueError(
            f"GNN target mismatch: constraint_type={constraint_type!r} needs "
            f"target_column={expected_target!r}, metadata contains "
            f"{metadata.get('target_column')!r}."
        )
    state_dict = _load_state_dict(model_path)

    expected_features = (
        WEIBULL_FEATURE_NAMES
        if constraint_type == CONSTRAINT_WEIBULL else LEGACY_FEATURE_NAMES
    )
    if metadata.get("feature_names") != expected_features:
        raise ValueError(
            "GNN feature order does not match the selected constraint. "
            f"Expected {expected_features}, got {metadata.get('feature_names')}."
        )
    expected_input_size = int(metadata.get("input_size", len(expected_features)))
    if expected_input_size != len(expected_features):
        raise ValueError(
            f"GNN input_size={expected_input_size} does not match "
            f"{len(expected_features)} embedded FJSP features."
        )

    model, variables = build_base_fjsp(model, instance)
    _add_start_time_variables(model, variables, instance)
    if constraint_type == CONSTRAINT_WEIBULL:
        _add_weibull_age_variables(model, variables, instance)
    _add_schedule_upper_bounds(model, variables)

    predicted_failure_cost = _add_gnn_prediction(
        model,
        variables,
        instance,
        state_dict,
        metadata,
        constraint_type,
    )

    if objective_weight is None:
        objective_weight = float(getattr(instance, "mu_fail", 1.0))
    else:
        objective_weight = float(objective_weight)

    variables.update(
        {
            "gnn_model_path": str(model_path),
            "gnn_metadata_path": str(metadata_path),
            "gnn_metadata": metadata,
            "gnn_objective_weight": objective_weight,
            "gnn_used_in_objective": False,
            "gnn_schedule_upper_bounds": bool(add_schedule_upper_bounds),
            "constraint_type": constraint_type,
            "constraint_target_column": expected_target,
            "constraint_budget": constraint_budget,
            "constraint_safety_margin": safety_margin,
            "constraint_safety_margin_per_operation": safety_margin_per_operation,
            "constraint_effective_safety_margin": effective_safety_margin,
            "scale_weibull_budget": bool(scale_weibull_budget),
            "weibull_budget_per_operation": float(weibull_budget_per_operation),
            "constraint_enforced": bool(enforce_constraint),
        }
    )

    if enforce_constraint:
        model.addCons(
            predicted_failure_cost + effective_safety_margin <= constraint_budget,
            name=f"gnn_{constraint_type}_budget",
        )
    model.setObjective(variables["C_max"], "minimize")

    return model, variables


def write_solution_file(model: Model, variables, _instance, filename="solution.txt"):
    objective = ""
    makespan = ""
    predicted_failure_cost = ""
    predicted_failure_cost_raw = ""
    exact_weibull_postcheck = ""
    solution = None

    if model.getNSols() > 0:
        solution = model.getBestSol()
        objective = f"{float(model.getObjVal()):.4f}"
        makespan = f"{float(model.getSolVal(solution, variables['C_max'])):.4f}"
        predicted_failure_cost = (
            f"{float(model.getSolVal(solution, variables['predicted_failure_cost'])):.4f}"
        )
        predicted_failure_cost_raw = (
            f"{float(model.getSolVal(solution, variables['predicted_failure_cost_raw'])):.4f}"
        )
        if variables["constraint_type"] == CONSTRAINT_WEIBULL:
            exact_value = 0.0
            for operation in variables["real_operations"]:
                machine = max(
                    variables["eligible_machines"][operation],
                    key=lambda k: model.getSolVal(
                        solution, variables["Y"][operation, k]
                    ),
                )
                age = float(model.getSolVal(
                    solution, variables["R"][operation, machine]
                ))
                processing = float(variables["processing_times"][operation, machine])
                eta = variables["weibull_eta"][machine]
                beta = variables["weibull_beta"][machine]
                probability = 1.0 - math.exp(
                    -(((age + processing) / eta) ** beta - (age / eta) ** beta)
                )
                exact_value += variables["failure_costs"][operation, machine] * probability
            exact_weibull_postcheck = f"{exact_value:.6f}"

    with open(filename, "w", encoding="utf-8") as file:
        file.write(f"Status: {model.getStatus()}\n")
        file.write(f"Objective: {objective}\n")
        file.write(f"Makespan: {makespan}\n")
        file.write(f"Constraint type: {variables['constraint_type']}\n")
        file.write(f"Predicted constraint value: {predicted_failure_cost}\n")
        file.write(f"Constraint budget: {variables['constraint_budget']:.6f}\n")
        file.write(f"Safety margin: {variables['constraint_safety_margin']:.6f}\n")
        file.write(
            "Safety margin per operation: "
            f"{variables['constraint_safety_margin_per_operation']:.6f}\n"
        )
        file.write(
            "Effective safety margin: "
            f"{variables['constraint_effective_safety_margin']:.6f}\n"
        )
        file.write(f"Exact Weibull postcheck: {exact_weibull_postcheck}\n")
        file.write(f"Raw predicted constraint value: {predicted_failure_cost_raw}\n")
        file.write(f"Big M: {float(variables['H']):.4f}\n")
        file.write(f"Berechnungszeit: {float(model.getSolvingTime()):.4f}\n")
        file.write("\nGNN parameters:\n")
        file.write(f"model_path: {variables['gnn_model_path']}\n")
        file.write(f"metadata_path: {variables['gnn_metadata_path']}\n")
        file.write(f"graph_mode: {variables['gnn_graph_mode']}\n")
        file.write(f"objective_weight: {variables['gnn_objective_weight']:.4f}\n")
        file.write(
            f"used_in_objective: {str(variables['gnn_used_in_objective'])}\n"
        )
        file.write(
            f"schedule_upper_bounds: {str(variables['gnn_schedule_upper_bounds'])}\n"
        )
        file.write(f"num_gnn_edges: {len(variables['gnn_edges'])}\n")

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
                    round(model.getSolVal(solution, variables["Y"][operation, machine]))
                )
                p_value = float(variables["processing_times"][operation, machine])
                file.write(
                    f"Y[{operation},{machine}] = {y_value}, p={p_value:.4f}\n"
                )
