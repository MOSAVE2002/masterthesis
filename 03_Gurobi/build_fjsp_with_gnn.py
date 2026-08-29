import importlib
import json
import math
import sys
from pathlib import Path

import gurobipy as gp
from gurobipy import GRB
from helper.gurobi_solution_writer import write_comparable_solution
from helper.economic_objective import (
    add_economic_cost_objective,
    add_robust_due_date_constraints,
)
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

_base_fjsp = importlib.import_module("03_Gurobi.build_fjsp")
STATUS_NAMES = _base_fjsp.STATUS_NAMES
build_base_fjsp = _base_fjsp.build_fjsp  

from helper.surrogate_constraint import (
    CONSTRAINT_WEIBULL,
    target_column,
    validate_constraint_type,
)
from helper.sequence_setup import (
    RELIABILITY_GNN_GRAPH_SCHEMA,
    RELIABILITY_GNN_OUTPUT_HEAD,
    add_reliability_graph_variables,
    normalize_reliability_graph_config,
    reliability_graph_config_dict,
    reliability_node_feature_names,
)
from helper.stochastic_fjsp import (
    ensure_stochastic_parameters,
    normalize_machine_profile_config,
    stochastic_parameters,
)
_gnn_architecture = importlib.import_module(
    "04_GraphNeuralNetworks.models.gnn_architecture"
)
CONV_LINEAR = _gnn_architecture.CONV_LINEAR
CONV_SAGE = _gnn_architecture.CONV_SAGE
CONV_JOB = _gnn_architecture.CONV_JOB
POOL_ADD = _gnn_architecture.POOL_ADD
VALID_LAYER_COUNTS = _gnn_architecture.VALID_LAYER_COUNTS
validate_architecture = _gnn_architecture.validate_architecture

GRAPH_MODE_FIXED_CANDIDATE = "fixed_candidate"
EXPECTED_OUTPUT_HEAD = RELIABILITY_GNN_OUTPUT_HEAD
EXPECTED_GRAPH_SCHEMA = RELIABILITY_GNN_GRAPH_SCHEMA


def _resolve_path(path_value):
    if path_value in (None, ""):
        raise ValueError("Both model_path and metadata_path are required.")
    path = Path(path_value)
    if not path.is_absolute():
        path = ROOT_DIR / path
    return path


def _load_metadata(metadata_path):
    # load data from trained model
    if not metadata_path.exists():
        raise FileNotFoundError(f"GNN metadata file not found: {metadata_path}")

    with metadata_path.open(encoding="utf-8") as file:
        metadata = json.load(file)

    graph_mode = metadata.get("graph_mode", GRAPH_MODE_FIXED_CANDIDATE)
    if graph_mode != GRAPH_MODE_FIXED_CANDIDATE:
        raise ValueError(
            "The embedded Gurobi GNN solver currently supports only "
            "graph_mode='fixed_candidate'. Please retrain/use a fixed-candidate "
            "GNN model for direct MILP embedding."
        )

    architecture = validate_architecture(
        GRAPH_MODE_FIXED_CANDIDATE,
        metadata.get("convolution", CONV_SAGE),
        metadata.get("aggregation", "sum"),
        metadata.get("pooling", "global_add"),
    )
    convolution = architecture["convolution"]
    if metadata.get("output_head") != EXPECTED_OUTPUT_HEAD:
        raise ValueError(
            "The embedded repair-buffer GNN requires output_head="
            f"{EXPECTED_OUTPUT_HEAD!r}."
        )
    if metadata.get("target_column") != target_column(CONSTRAINT_WEIBULL):
        raise ValueError(
            "The embedded GNN requires jobspecific expected repair buffers."
        )
    if metadata.get("job_target") != "job_expected_repair_buffer":
        raise ValueError("The embedded GNN requires one repair buffer per job.")
    if (
        convolution in {CONV_SAGE, CONV_JOB}
        and not metadata.get("include_job_precedence_edges", False)
    ):
        raise ValueError("The job-buffer GNN requires fixed job edges.")
    if (
        convolution == CONV_SAGE
        and not metadata.get("include_machine_predecessor_edges", False)
    ):
        raise ValueError("The active GNN pipeline requires machine edges.")
    if (
        convolution != CONV_SAGE
        and metadata.get("include_machine_predecessor_edges", False)
    ):
        raise ValueError(
            f"{convolution} metadata must not enable machine messages."
        )
    expected_machine_scope = "direct" if convolution == CONV_SAGE else "none"
    if metadata.get("machine_predecessor_edge_scope") != expected_machine_scope:
        raise ValueError(
            f"The embedded {convolution} model requires "
            "machine_predecessor_edge_scope="
            f"{expected_machine_scope!r}."
        )
    if metadata.get("graph_schema") != EXPECTED_GRAPH_SCHEMA:
        raise ValueError(
            "The embedded GNN requires graph_schema="
            f"{EXPECTED_GRAPH_SCHEMA!r}."
        )
    expected_message_passing = (
        "none" if convolution == CONV_LINEAR else "source_node_states_only"
    )
    if metadata.get("message_passing") != expected_message_passing:
        raise ValueError(
            f"The embedded {convolution} model requires message_passing="
            f"{expected_message_passing!r}."
        )

    return metadata


def _load_state_dict(model_path):
    if not model_path.exists():
        raise FileNotFoundError(f"GNN model file not found: {model_path}")

    return {
        key: value.detach().cpu().numpy()
        for key, value in torch.load(model_path, map_location="cpu").items() # Jeder Pytorch Tensor wird in ein Numpy Array umgewandelt
    }


def _add_stochastic_machine_state(
    model,
    variables,
    instance,
    reliability_graph_config=None,
):
    """Attach the machine parameters used by every surrogate feature set."""
    graph_cfg = normalize_reliability_graph_config(reliability_graph_config)
    ensure_stochastic_parameters(instance)
    parameters = stochastic_parameters(instance)
    alpha = parameters["alpha"]
    beta = parameters["beta"]
    repair_durations = {
        (operation, machine): parameters["repair_duration"][machine]
        for operation, machine in variables["Y_index"]
    }
    variables.update(
        {
            "machine_modernity": parameters["theta"],
            "machine_speed": parameters["speed"],
            "weibull_alpha": alpha,
            "weibull_beta": beta,
            "repair_rate": parameters["repair_rate"],
            "repair_durations": repair_durations,
            "reliability_graph_config": reliability_graph_config_dict(
                graph_cfg
            ),
        }
    )
    return variables


def _add_reliability_graph_state(
    model,
    variables,
    instance,
    reliability_graph_config=None,
):
    """Create direct U_ijk edges for SAGE and shared machine parameters."""
    graph_cfg = normalize_reliability_graph_config(reliability_graph_config)
    _add_stochastic_machine_state(
        model, variables, instance, graph_cfg
    )
    add_reliability_graph_variables(
        model,
        variables,
        instance,
        graph_cfg,
    )
    return variables


def _linear_expr(coefficients, values, bias=0.0):
    expr = gp.LinExpr(float(bias))
    for coefficient, value in zip(coefficients, values):
        coefficient = float(coefficient)
        if abs(coefficient) > 1e-12:
            expr += coefficient * value
    return expr


def _linear_bounds(coefficients, bounds, bias=0.0):
    lower = upper = float(bias)
    for coefficient, (value_lower, value_upper) in zip(coefficients, bounds):
        coefficient = float(coefficient)
        if coefficient >= 0.0:
            lower += coefficient * value_lower
            upper += coefficient * value_upper
        else:
            lower += coefficient * value_upper
            upper += coefficient * value_lower
    return lower, upper


def _add_relu(model, expression, name, lower=None, upper=None):
    if lower is None or upper is None:
        raise ValueError(
            f"Exact ReLU formulation requires finite bounds for {name}."
        )
    lower = float(lower)
    upper = float(upper)
    if not math.isfinite(lower) or not math.isfinite(upper):
        raise ValueError(
            f"Exact ReLU formulation requires finite bounds for {name}, "
            f"received [{lower}, {upper}]."
        )
    bound_tolerance = 1e-9 * max(1.0, abs(lower), abs(upper))
    if lower > upper + bound_tolerance:
        raise ValueError(
            f"Invalid ReLU bounds for {name}: lower={lower} > upper={upper}."
        )
    if lower > upper:
        center = 0.5 * (lower + upper)
        lower = upper = center

    pre_activation = model.addVar(
        lb=lower,
        ub=upper,
        vtype=GRB.CONTINUOUS,
        name=f"{name}_pre",
    )
    activation = model.addVar(
        lb=max(0.0, lower),
        ub=max(0.0, upper),
        vtype=GRB.CONTINUOUS,
        name=name,
    )
    if not hasattr(model, "_gnn_proven_variable_bounds"):
        model._gnn_proven_variable_bounds = {}
    model._gnn_proven_variable_bounds[id(activation)] = (
        max(0.0, lower),
        max(0.0, upper),
    )
    model.addConstr(pre_activation == expression, name=f"{name}_pre_def")
    if upper <= 0.0:
        activation.lb = 0.0
        activation.ub = 0.0
    elif lower >= 0.0:
        model.addConstr(activation == pre_activation, name=f"{name}_relu_linear")
    else:
        # Ideal single-neuron ReLU formulation on the neuron-specific
        # interval lower <= pre_activation <= upper.  The non-negativity
        # inequality is already represented by activation.lb = 0.
        phase = model.addVar(
            vtype=GRB.BINARY,
            name=f"{name}_relu_phase",
        )
        model.addConstr(
            activation >= pre_activation,
            name=f"{name}_relu_ge_pre",
        )
        model.addConstr(
            activation
            <= pre_activation - lower * (1.0 - phase),
            name=f"{name}_relu_upper_active",
        )
        model.addConstr(
            activation <= upper * phase,
            name=f"{name}_relu_upper_inactive",
        )
    return activation


def _add_linear_layer(
    model,
    state_dict,
    layer_name,
    input_vectors,
    input_bounds,
):
    weight = state_dict[f"{layer_name}.weight"]
    bias = state_dict[f"{layer_name}.bias"]
    outputs, output_bounds = [], []
    for node_idx, values in enumerate(input_vectors):
        node_outputs, node_bounds = [], []
        for channel_idx in range(weight.shape[0]):
            lower, upper = _linear_bounds(
                weight[channel_idx], input_bounds[node_idx], bias=bias[channel_idx]
            )
            activation = _add_relu(
                model,
                _linear_expr(weight[channel_idx], values, bias=bias[channel_idx]),
                name=f"{layer_name}_linear_node{node_idx}_h{channel_idx}",
                lower=lower,
                upper=upper,
            )
            node_outputs.append(activation)
            node_bounds.append((max(0.0, lower), max(0.0, upper)))
        outputs.append(node_outputs)
        output_bounds.append(node_bounds)
    return outputs, output_bounds


def _build_node_feature_expressions(instance, variables, constraint_type):
    operations = list(variables["real_operations"])
    Y = variables["Y"]
    horizon = float(variables["service_horizon"])
    node_features = []
    node_bounds = []
    for operation in operations:
        if constraint_type == CONSTRAINT_WEIBULL:
            processing_over_alpha = gp.quicksum(
                Y[operation, machine]
                * float(instance.processing_times[operation, machine])
                / variables["weibull_alpha"][machine]
                for machine in instance.eligible_machines[operation]
            )
            rate_times_alpha = gp.quicksum(
                Y[operation, machine]
                * variables["repair_rate"][machine]
                * variables["weibull_alpha"][machine]
                / 30.0
                for machine in instance.eligible_machines[operation]
            )
            beta_scaled = gp.quicksum(
                Y[operation, machine] * variables["weibull_beta"][machine] / 5.0
                for machine in instance.eligible_machines[operation]
            )
            node_features.append([
                variables["S"][operation] / horizon,
                variables["C"][operation] / horizon,
                processing_over_alpha,
                rate_times_alpha,
                beta_scaled,
            ])
            node_bounds.append([
                (0.0, float(variables["H"]) / horizon),
                (0.0, float(variables["H"]) / horizon),
                (
                    min(
                        float(instance.processing_times[operation, machine])
                        / variables["weibull_alpha"][machine]
                        for machine in instance.eligible_machines[operation]
                    ),
                    max(
                        float(instance.processing_times[operation, machine])
                        / variables["weibull_alpha"][machine]
                        for machine in instance.eligible_machines[operation]
                    ),
                ),
                (
                    min(
                        variables["repair_rate"][machine]
                        * variables["weibull_alpha"][machine]
                        / 30.0
                        for machine in instance.eligible_machines[operation]
                    ),
                    max(
                        variables["repair_rate"][machine]
                        * variables["weibull_alpha"][machine]
                        / 30.0
                        for machine in instance.eligible_machines[operation]
                    ),
                ),
                (
                    min(
                        variables["weibull_beta"][machine] / 5.0
                        for machine in instance.eligible_machines[operation]
                    ),
                    max(
                        variables["weibull_beta"][machine] / 5.0
                        for machine in instance.eligible_machines[operation]
                    ),
                ),
            ])

    return node_features, node_bounds


def _add_schedule_upper_bounds(variables):
    H = float(variables["H"])
    for operation in variables["real_operations"]:
        variables["C"][operation].ub = H


def _add_job_time_bounds(variables, instance):
    """Tighten completion times using only mandatory job precedence."""
    operations = list(variables["real_operations"])
    operation_set = set(operations)
    minimum_duration = {
        operation: min(
            float(instance.processing_times[operation, machine])
            for machine in instance.eligible_machines[operation]
        )
        for operation in operations
    }
    predecessors = {
        operation: [
            predecessor
            for predecessor in instance.predecessors.get(operation, [])
            if predecessor in operation_set
        ]
        for operation in operations
    }
    successors = {operation: [] for operation in operations}
    for operation in operations:
        for predecessor in predecessors[operation]:
            successors[predecessor].append(operation)

    earliest_cache, suffix_cache = {}, {}

    def earliest_completion(operation, visiting=None):
        if operation in earliest_cache:
            return earliest_cache[operation]
        visiting = set(visiting or ())
        if operation in visiting:
            raise ValueError("Job precedence graph must be acyclic.")
        visiting.add(operation)
        value = minimum_duration[operation] + max(
            (
                earliest_completion(predecessor, visiting)
                for predecessor in predecessors[operation]
            ),
            default=0.0,
        )
        earliest_cache[operation] = value
        return value

    def mandatory_suffix_after(operation, visiting=None):
        if operation in suffix_cache:
            return suffix_cache[operation]
        visiting = set(visiting or ())
        if operation in visiting:
            raise ValueError("Job precedence graph must be acyclic.")
        visiting.add(operation)
        value = max(
            (
                minimum_duration[successor]
                + mandatory_suffix_after(successor, visiting)
                for successor in successors[operation]
            ),
            default=0.0,
        )
        suffix_cache[operation] = value
        return value

    horizon = float(variables["H"])
    for operation in operations:
        variables["C"][operation].lb = max(
            float(variables["C"][operation].lb),
            earliest_completion(operation),
        )
        variables["C"][operation].ub = min(
            float(variables["C"][operation].ub),
            horizon - mandatory_suffix_after(operation),
        )
    variables["job_earliest_completion_bounds"] = dict(earliest_cache)
    variables["job_mandatory_suffix_bounds"] = dict(suffix_cache)




def _relational_incoming_edges(
    instance,
    variables,
    convolution,
):
    operations = list(variables["real_operations"])
    operation_to_idx = {
        operation: index for index, operation in enumerate(operations)
    }
    incoming = {index: [] for index in range(len(operations))}
    if convolution == CONV_SAGE:
        for source, target, machine in variables["U_index"]:
            source_idx = operation_to_idx[source]
            target_idx = operation_to_idx[target]
            gate = variables["U"][source, target, machine]
            incoming[target_idx].append(
                (source_idx, gate, source, target, machine)
            )
    if convolution in {CONV_SAGE, CONV_JOB}:
        for target in operations:
            for source in instance.predecessors.get(target, []):
                if source not in operation_to_idx:
                    continue
                incoming[operation_to_idx[target]].append(
                    (
                        operation_to_idx[source],
                        1.0,
                        source,
                        target,
                        -1,
                    )
                )
    return incoming


def _gate_hidden_value(
    model,
    value,
    gate,
    lower,
    upper,
    name,
):
    if isinstance(gate, (int, float)):
        if float(gate) == 1.0:
            return value
        if float(gate) == 0.0:
            return 0.0
    gated = model.addVar(
        lb=min(0.0, float(lower)),
        ub=max(0.0, float(upper)),
        vtype=GRB.CONTINUOUS,
        name=name,
    )
    model.addConstr(gated >= lower * gate)
    model.addConstr(gated <= upper * gate)
    model.addConstr(gated >= value - upper * (1.0 - gate))
    model.addConstr(gated <= value - lower * (1.0 - gate))
    return gated
















def _add_relational_node_layer(
    model,
    state_dict,
    layer_name,
    convolution,
    inputs,
    input_bounds,
    incoming,
):
    if convolution == CONV_LINEAR:
        return _add_linear_layer(
            model, state_dict, layer_name, inputs, input_bounds
        )
    if convolution not in {CONV_SAGE, CONV_JOB}:
        raise ValueError(f"Unsupported convolution: {convolution}")
    root_weight = state_dict[f"{layer_name}.lin_root.weight"]
    message_weight = state_dict[f"{layer_name}.lin_message.weight"]
    bias = state_dict[f"{layer_name}.lin_root.bias"]

    output_vectors, output_bounds = [], []
    input_size = len(inputs[0])
    for target_idx, root_values in enumerate(inputs):
        aggregated = []
        aggregated_bounds = []
        for feature_idx in range(input_size):
            gated_values = []
            fixed_lower = fixed_upper = 0.0
            gated_lower = gated_upper = 0.0
            for (
                source_idx,
                gate,
                source,
                target,
                machine,
            ) in incoming[target_idx]:
                lower, upper = input_bounds[source_idx][feature_idx]
                gated_values.append(
                    _gate_hidden_value(
                        model,
                        inputs[source_idx][feature_idx],
                        gate,
                        lower,
                        upper,
                        name=(
                            f"{layer_name}_direct_msg"
                            f"[{source},{target},{machine},{feature_idx}]"
                        ),
                    )
                )
                if isinstance(gate, (int, float)) and float(gate) == 1.0:
                    fixed_lower += lower
                    fixed_upper += upper
                else:
                    gated_lower += min(0.0, lower)
                    gated_upper += max(0.0, upper)
            aggregated.append(
                gp.quicksum(gated_values) if gated_values else 0.0
            )
            aggregated_bounds.append(
                (
                    fixed_lower + gated_lower,
                    fixed_upper + gated_upper,
                )
            )
        node_outputs, node_output_bounds = [], []
        for channel_idx in range(root_weight.shape[0]):
            expression = _linear_expr(
                root_weight[channel_idx],
                root_values,
                bias=bias[channel_idx],
            )
            expression += _linear_expr(
                message_weight[channel_idx], aggregated
            )
            lower, upper = _linear_bounds(
                root_weight[channel_idx],
                input_bounds[target_idx],
                bias=bias[channel_idx],
            )
            message_lower, message_upper = _linear_bounds(
                message_weight[channel_idx], aggregated_bounds
            )
            lower += message_lower
            upper += message_upper
            activation = _add_relu(
                model,
                expression,
                name=(
                    f"{layer_name}_relational_node"
                    f"{target_idx}_h{channel_idx}"
                ),
                lower=lower,
                upper=upper,
            )
            node_outputs.append(activation)
            node_output_bounds.append(
                (max(0.0, lower), max(0.0, upper))
            )
        output_vectors.append(node_outputs)
        output_bounds.append(node_output_bounds)
    return output_vectors, output_bounds


def _add_relational_gnn_output(
    model,
    variables,
    instance,
    state_dict,
    metadata,
    constraint_type,
):
    operations = list(variables["real_operations"])
    node_features, node_bounds = _build_node_feature_expressions(
        instance, variables, constraint_type
    )
    local_features, local_bounds = node_features, node_bounds
    architecture = validate_architecture(
        GRAPH_MODE_FIXED_CANDIDATE,
        metadata.get("convolution", CONV_SAGE),
        metadata.get("aggregation", "sum"),
        metadata.get("pooling", POOL_ADD),
    )
    convolution = architecture["convolution"]
    incoming = _relational_incoming_edges(
        instance, variables, convolution
    )

    hidden, hidden_bounds = (
        _add_relational_node_layer(
            model,
            state_dict,
            "conv1",
            convolution,
            local_features,
            local_bounds,
            incoming,
        )
    )
    num_layers = int(metadata.get("num_graphsage_layers", 2))
    if num_layers not in VALID_LAYER_COUNTS:
        choices = ", ".join(map(str, sorted(VALID_LAYER_COUNTS)))
        raise ValueError(
            f"GNN metadata must specify one of {choices} layers."
        )
    for layer_number in range(2, num_layers + 1):
        hidden, hidden_bounds = (
            _add_relational_node_layer(
                model,
                state_dict,
                f"conv{layer_number}",
                convolution,
                hidden,
                hidden_bounds,
                incoming,
            )
        )

    output_weight = state_dict["out.weight"][0]
    output_bias = state_dict["out.bias"][0]
    skip_weight = state_dict.get("out_input.weight")
    skip_bias = state_dict.get("out_input.bias")
    operation_to_index = {
        operation: index for index, operation in enumerate(operations)
    }
    output_expressions, raw_outputs = {}, {}
    for job in sorted(instance.jobs):
        node_indices = [
            operation_to_index[operation]
            for operation in instance.jobs[job]
        ]
        pooled_hidden = [
            gp.quicksum(hidden[index][channel] for index in node_indices)
            for channel in range(len(hidden[0]))
        ]
        pooled_local = [
            gp.quicksum(local_features[index][channel] for index in node_indices)
            for channel in range(len(local_features[0]))
        ]
        expression = _linear_expr(
            output_weight, pooled_hidden, bias=output_bias
        )
        if skip_weight is not None:
            expression += _linear_expr(
                skip_weight[0], pooled_local, bias=skip_bias[0]
            )
        raw = model.addVar(
            lb=-GRB.INFINITY, name=f"gnn_raw_job_repair_buffer[{job}]"
        )
        buffer = model.addVar(
            lb=0.0, name=f"gnn_job_expected_repair_buffer[{job}]"
        )
        model.addConstr(raw == expression, name=f"gnn_raw_output_def[{job}]")
        model.addGenConstrMax(
            buffer, [raw], constant=0.0,
            name=f"gnn_repair_buffer_relu[{job}]",
        )
        raw_outputs[job] = raw
        output_expressions[job] = buffer
    variables.update({
        "gnn_job_output_expressions": output_expressions,
        "gnn_raw_job_outputs": raw_outputs,
        "job_expected_delays": output_expressions,
        "job_repair_buffer_postprocess": "relu_inside_model",
    })
    return output_expressions


def _add_midpoint_state(model, variables, instance):
    """Create nominal starts, operation midpoints and T*Y products."""
    horizon = float(variables["H"])
    starts, midpoints, durations, midpoint_times_assignment = {}, {}, {}, {}
    for operation in variables["real_operations"]:
        duration = gp.quicksum(
            float(instance.processing_times[operation, machine])
            * variables["Y"][operation, machine]
            for machine in instance.eligible_machines[operation]
        )
        durations[operation] = duration
        starts[operation] = model.addVar(
            lb=0.0, ub=horizon, name=f"S_nominal[{operation}]"
        )
        midpoints[operation] = model.addVar(
            lb=0.0, ub=horizon, name=f"t_midpoint[{operation}]"
        )
        model.addConstr(
            variables["C"][operation] == starts[operation] + duration,
            name=f"gnn_nominal_start_def[{operation}]",
        )
        model.addConstr(
            midpoints[operation] == starts[operation] + 0.5 * duration,
            name=f"gnn_midpoint_def[{operation}]",
        )
        for machine in instance.eligible_machines[operation]:
            product = model.addVar(
                lb=0.0,
                ub=horizon,
                name=f"midpoint_times_Y[{operation},{machine}]",
            )
            y = variables["Y"][operation, machine]
            model.addConstr(product <= midpoints[operation])
            model.addConstr(product <= horizon * y)
            model.addConstr(
                product >= midpoints[operation] - horizon * (1.0 - y)
            )
            midpoint_times_assignment[operation, machine] = product
    variables.update({
        "S": starts,
        "T": midpoints,
        "D": durations,
        "midpoint_times_assignment": midpoint_times_assignment,
    })


def _service_operations(instance, job, scope):
    return (
        list(instance.jobs[job])
        if scope == "job"
        else list(instance.real_operations)
    )


def _add_gnn_service_metadata(
    model, variables, instance, graph_cfg, metadata
):
    variables.update({
        "service_constraints": {},
        "service_scope": graph_cfg.service_scope,
        "due_dates": dict(instance.due_dates),
        "job_repair_buffer_label_method": metadata.get(
            "job_repair_buffer_label_method"
        ),
    })


def build_fjsp(
    fjsp,
    instance,
    model_path,
    metadata_path,
    convolution=CONV_SAGE,
    aggregation="sum",
    pooling="global_add",
    layers=None,
    hidden_channels=None,
    add_schedule_upper_bounds=True,
    constraint_type=CONSTRAINT_WEIBULL,
    reliability_graph_config=None,
    analytic_bounds=True,
    facility_cost_per_time=1.0,
    tardiness_cost_per_time=1.0,
):
    """Build the ReLU-GNN MILP with one expected repair buffer per job."""
    model = fjsp
    analytic_bounds = bool(analytic_bounds)
    constraint_type = validate_constraint_type(constraint_type)
    ensure_stochastic_parameters(instance)
    graph_cfg = normalize_reliability_graph_config(reliability_graph_config)

    requested_architecture = validate_architecture(
        GRAPH_MODE_FIXED_CANDIDATE,
        convolution,
        aggregation,
        pooling,
    )
    if layers is None or hidden_channels is None:
        raise ValueError("layers and hidden_channels are required.")
    model_path = _resolve_path(model_path)
    metadata_path = _resolve_path(metadata_path)
    metadata = _load_metadata(metadata_path)
    metadata_architecture = validate_architecture(
        metadata.get("graph_mode", GRAPH_MODE_FIXED_CANDIDATE),
        metadata.get("convolution", CONV_SAGE),
        metadata.get("aggregation", "sum"),
        metadata.get("pooling", "global_add"),
    )
    if metadata_architecture != requested_architecture:
        raise ValueError(
            "Configured GNN architecture does not match metadata: "
            f"config={requested_architecture}, metadata={metadata_architecture}."
        )
    metadata_layers = int(metadata.get("num_graphsage_layers", 2))
    metadata_hidden = int(metadata.get("hidden_channels", 16))
    if layers is not None and metadata_layers != int(layers):
        raise ValueError(
            f"Configured layers={int(layers)} do not match model metadata "
            f"layers={metadata_layers}."
        )
    if (
        hidden_channels is not None
        and metadata_hidden != int(hidden_channels)
    ):
        raise ValueError(
            "Configured hidden_channels="
            f"{int(hidden_channels)} do not match model metadata "
            f"hidden_channels={metadata_hidden}."
        )
    expected_target = target_column(constraint_type)
    if metadata.get("target_column") != expected_target:
        raise ValueError(
            f"GNN target mismatch: constraint_type={constraint_type!r} needs "
            f"target_column={expected_target!r}, metadata contains "
            f"{metadata.get('target_column')!r}."
        )
    state_dict = _load_state_dict(model_path)

    expected_features = reliability_node_feature_names(graph_cfg)
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
    metadata_graph_config_values = dict(
        metadata.get("reliability_graph_config") or {}
    )
    # Models trained before removal of C_max may contain this obsolete
    # objective-only field. It never affected graph features or predictions.
    metadata_graph_config_values.pop("makespan_weight", None)
    metadata_graph_config = normalize_reliability_graph_config(
        metadata_graph_config_values
    )
    metadata_prediction_config = reliability_graph_config_dict(
        metadata_graph_config
    )
    solver_prediction_config = reliability_graph_config_dict(graph_cfg)
    if metadata_prediction_config != solver_prediction_config:
        raise ValueError(
            "GNN reliability-graph parameters differ from the solver model: "
            f"metadata={metadata_prediction_config}, "
            f"solver={solver_prediction_config}."
        )
    metadata_profiles = metadata.get("machine_profile_config")
    if metadata_profiles is None:
        raise ValueError(
            "GNN metadata does not contain machine_profile_config. "
            "Regenerate the training data and retrain the GNN before solving."
        )
    expected_profiles = normalize_machine_profile_config(metadata_profiles)
    actual_profiles = normalize_machine_profile_config(
        getattr(instance, "machine_profile_config", None)
    )
    if expected_profiles != actual_profiles:
        raise ValueError(
            "GNN machine profiles differ from the solved instance. "
            "Regenerate the training data and retrain the GNN."
        )
    metadata_time_unit = metadata.get("time_unit_minutes")
    if metadata_time_unit is None or not math.isclose(
        float(metadata_time_unit),
        float(getattr(instance, "time_unit_minutes", 1.0)),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "GNN time-unit metadata differs from the solved instance. "
            "Regenerate the training data and retrain the GNN."
        )
    model, variables = build_base_fjsp(
        model,
        instance,
        include_makespan=False,
        horizon_upper_bound=None,
        enforce_due_dates=False,
        economic_objective=False,
    )
    for operation in variables["real_operations"]:
        variables["C"][operation].ub = float(variables["H"])
    _add_midpoint_state(model, variables, instance)
    variables["service_horizon"] = max(
        float(value) for value in instance.due_dates.values()
    )
    if requested_architecture["convolution"] == CONV_SAGE:
        _add_reliability_graph_state(
            model,
            variables,
            instance,
            graph_cfg,
        )
    else:
        _add_stochastic_machine_state(
            model,
            variables,
            instance,
            graph_cfg,
        )
    if add_schedule_upper_bounds:
        _add_schedule_upper_bounds(variables)
        if analytic_bounds:
            _add_job_time_bounds(variables, instance)
    job_repair_buffers = _add_relational_gnn_output(
        model, variables, instance, state_dict, metadata,
        constraint_type,
    )
    _add_gnn_service_metadata(
        model, variables, instance, graph_cfg, metadata
    )
    variables.update(
        {
            "gnn_model_path": str(model_path),
            "gnn_metadata_path": str(metadata_path),
            "gnn_metadata": metadata,
            "constraint_type": constraint_type,
        }
    )
    add_robust_due_date_constraints(
        model,
        variables,
        instance,
        job_repair_buffers,
    )
    add_economic_cost_objective(
        model,
        variables,
        instance,
        facility_cost_per_time=facility_cost_per_time,
        tardiness_cost_per_time=tardiness_cost_per_time,
    )
    formulation = "gnn_expected_repair_buffer_tardiness_cost_v13"
    variables.update({
        "formulation": formulation,
    })

    model.update()
    return model, variables

def write_solution_file(model, variables, instance, filename="solution.txt"):
    """Write the common exact/GNN comparison format."""
    return write_comparable_solution(
        model,
        variables,
        filename,
        instance=instance,
    )


if __name__ == "__main__":
    pass
