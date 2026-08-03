import importlib
import json
import math
import sys
from pathlib import Path

import gurobipy as gp
from gurobipy import GRB
from helper.gurobi_solution_writer import write_comparable_solution
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

_base_fjsp = importlib.import_module("03_Gurobi.build_fjsp")
STATUS_NAMES = _base_fjsp.STATUS_NAMES
build_base_fjsp = _base_fjsp.build_fjsp  

_nonlinear_fjsp = importlib.import_module("03_Gurobi.build_fjsp_with_nonlinear")
_first_machine_parameter = _nonlinear_fjsp._first_machine_parameter
_repair_duration = _nonlinear_fjsp._repair_duration
from helper.surrogate_constraint import (
    CONSTRAINT_WEIBULL,
    model_stem,
    target_column,
    validate_constraint_type,
)
from helper.sequence_setup import (
    RELIABILITY_EDGE_FEATURE_NAMES,
    RELIABILITY_GNN_GRAPH_SCHEMA,
    RELIABILITY_GNN_GRAPH_SCHEMA_JOB_ONLY,
    RELIABILITY_GNN_GRAPH_SCHEMA_WITH_JOB_EDGES,
    RELIABILITY_GNN_OUTPUT_HEAD,
    add_reliability_graph_variables,
    normalize_reliability_graph_config,
    reliability_graph_config_dict,
    reliability_edge_feature_names,
    reliability_node_feature_names,
    transition_edge_values,
)
_gnn_architecture = importlib.import_module(
    "04_GraphNeuralNetworks.models.gnn_architecture"
)
CONV_GCN = _gnn_architecture.CONV_GCN
CONV_GINE = _gnn_architecture.CONV_GINE
CONV_LINEAR = _gnn_architecture.CONV_LINEAR
CONV_MPNN = _gnn_architecture.CONV_MPNN
CONV_SAGE = _gnn_architecture.CONV_SAGE
POOL_ADD = _gnn_architecture.POOL_ADD
architecture_model_dir = _gnn_architecture.architecture_model_dir
architecture_stem = _gnn_architecture.architecture_stem
validate_architecture = _gnn_architecture.validate_architecture

DEFAULT_MODEL_SEED = 42
DEFAULT_MODEL_DIR = (
    ROOT_DIR / "04_GraphNeuralNetworks" / "trained_gnn_models"
)
GRAPH_MODE_FIXED_CANDIDATE = "fixed_candidate"
EXPECTED_OUTPUT_HEAD = RELIABILITY_GNN_OUTPUT_HEAD
EXPECTED_GRAPH_SCHEMA = RELIABILITY_GNN_GRAPH_SCHEMA
RELU_BIG_M = "big_m"
RELU_SOS1 = "sos1"
RELU_FORMULATIONS = {RELU_BIG_M, RELU_SOS1}


def _validate_relu_formulation(value):
    formulation = str(value).strip().lower()
    if formulation not in RELU_FORMULATIONS:
        raise ValueError(
            "relu_formulation must be 'big_m' or 'sos1', received "
            f"{value!r}."
        )
    return formulation


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

    validate_architecture(
        GRAPH_MODE_FIXED_CANDIDATE,
        metadata.get("convolution", CONV_SAGE),
        metadata.get("aggregation", "sum"),
        metadata.get("pooling", "global_add"),
    )
    if metadata.get("output_head") != EXPECTED_OUTPUT_HEAD:
        raise ValueError(
            "The embedded failure-probability GNN requires output_head="
            f"{EXPECTED_OUTPUT_HEAD!r}."
        )
    if metadata.get("node_target") != "operation_failure_probability":
        raise ValueError(
            "The embedded GNN requires node_target="
            "'operation_failure_probability'. Retrain the legacy "
            "expected-delay model first."
        )
    include_job_edges = bool(
        metadata.get("include_job_precedence_edges", False)
    )
    include_machine_edges = bool(
        metadata.get("include_machine_predecessor_edges", True)
    )
    if not include_machine_edges and not include_job_edges:
        raise ValueError("At least one predecessor edge type must be enabled.")
    expected_schema = (
        RELIABILITY_GNN_GRAPH_SCHEMA_WITH_JOB_EDGES
        if include_job_edges and include_machine_edges
        else RELIABILITY_GNN_GRAPH_SCHEMA_JOB_ONLY
        if include_job_edges
        else EXPECTED_GRAPH_SCHEMA
    )
    if metadata.get("graph_schema") != expected_schema:
        raise ValueError(
            "The embedded GNN requires graph_schema="
            f"{expected_schema!r}."
        )

    return metadata


def _load_state_dict(model_path):
    if not model_path.exists():
        raise FileNotFoundError(f"GNN model file not found: {model_path}")

    return {
        key: value.detach().cpu().numpy()
        for key, value in torch.load(model_path, map_location="cpu").items() # Jeder Pytorch Tensor wird in ein Numpy Array umgewandelt
    }


def _add_reliability_graph_state(
    model,
    variables,
    instance,
    reliability_graph_config=None,
    bound_tightening=False,
):
    """Create exactly the same reliability state as the nonlinear model."""
    graph_cfg = normalize_reliability_graph_config(reliability_graph_config)
    if not hasattr(instance, "mu_fail"):
        instance._set_default_nonlinear_parameters()
    machines = list(variables["machines"])
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
        (operation, machine): _repair_duration(
            instance, operation, machine
        )
        for operation, machine in variables["Y_index"]
    }
    if any(value < 0.0 for value in repair_durations.values()):
        raise ValueError("repair_duration must be nonnegative.")
    variables.update(
        {
            "mu_fail": float(getattr(instance, "mu_fail", 1.0)),
            "machine_initial_age": initial_age,
            "weibull_eta": eta,
            "weibull_beta": beta,
            "repair_durations": repair_durations,
        }
    )
    add_reliability_graph_variables(
        model,
        variables,
        instance,
        graph_cfg,
        tighten_age_bounds=bound_tightening,
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
    obbt_bounds = getattr(model, "_gnn_obbt_bounds", {})
    if name in obbt_bounds:
        obbt_lower, obbt_upper = map(float, obbt_bounds[name])
        if not math.isfinite(obbt_lower) or not math.isfinite(obbt_upper):
            raise ValueError(f"Non-finite OBBT bounds for {name}.")
        # OBBT is allowed to tighten, never enlarge, the analytically proven
        # interval.  The caller is responsible for deriving these bounds from
        # a relaxation containing every integer-feasible schedule.
        lower = max(lower, obbt_lower)
        upper = min(upper, obbt_upper)
        model._gnn_obbt_applied = getattr(
            model, "_gnn_obbt_applied", 0
        ) + 1
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
        midpoint = 0.5 * (lower + upper)
        lower = upper = midpoint

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
    elif getattr(model, "_gnn_relu_formulation", RELU_BIG_M) == RELU_SOS1:
        negative_part = model.addVar(
            lb=0.0,
            ub=-lower,
            vtype=GRB.CONTINUOUS,
            name=f"{name}_relu_negative",
        )
        model.addConstr(
            pre_activation == activation - negative_part,
            name=f"{name}_relu_split",
        )
        model.addSOS(GRB.SOS_TYPE1, [activation, negative_part], [1.0, 2.0])
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
    return activation, pre_activation


def _add_upper_clip(model, value, name, lower, upper, cap=1.0):
    """Add ``min(value, cap)`` using only bounds and linear constraints."""
    lower = float(lower)
    upper = float(upper)
    cap = float(cap)
    # In particular after OBBT, the variable carrying the preceding ReLU can
    # be tighter than the original interval-arithmetic bound.  Intersecting
    # with its proven bounds can fix the clip phase without changing min(x, 1).
    proven_bounds = getattr(model, "_gnn_proven_variable_bounds", {}).get(
        id(value)
    )
    if proven_bounds is not None:
        lower = max(lower, float(proven_bounds[0]))
        upper = min(upper, float(proven_bounds[1]))
    if not all(math.isfinite(bound) for bound in (lower, upper, cap)):
        raise ValueError(
            f"Exact upper clip requires finite bounds for {name}, "
            f"received [{lower}, {upper}] with cap={cap}."
        )
    bound_tolerance = 1e-9 * max(
        1.0,
        abs(lower),
        abs(upper),
        abs(cap),
    )
    if lower > upper + bound_tolerance:
        raise ValueError(
            f"Invalid clip bounds for {name}: lower={lower} > upper={upper}."
        )
    if lower > upper:
        midpoint = 0.5 * (lower + upper)
        lower = upper = midpoint

    clipped = model.addVar(
        lb=min(lower, cap),
        ub=min(upper, cap),
        vtype=GRB.CONTINUOUS,
        name=name,
    )
    if upper <= cap:
        model.addConstr(
            clipped == value,
            name=f"{name}_clip_inactive",
        )
    elif lower >= cap:
        clipped.lb = cap
        clipped.ub = cap
    else:
        # Convex-hull formulation for the two branches value <= cap and
        # value >= cap. The interval-specific constants are the smallest
        # valid Big-M values for this single-variable upper clip.
        above_cap = model.addVar(
            vtype=GRB.BINARY,
            name=f"{name}_clip_above_cap",
        )
        model.addConstr(
            clipped <= value,
            name=f"{name}_clip_le_value",
        )
        model.addConstr(
            clipped
            >= value - (upper - cap) * above_cap,
            name=f"{name}_clip_ge_value_branch",
        )
        model.addConstr(
            clipped
            >= cap - (cap - lower) * (1.0 - above_cap),
            name=f"{name}_clip_ge_cap_branch",
        )
    return clipped


def _add_linear_layer(
    model,
    state_dict,
    layer_name,
    input_vectors,
    input_bounds,
):
    weight = state_dict[f"{layer_name}.weight"]
    bias = state_dict[f"{layer_name}.bias"]
    outputs, pre_activations, output_bounds = [], [], []
    for node_idx, values in enumerate(input_vectors):
        node_outputs, node_pre, node_bounds = [], [], []
        for channel_idx in range(weight.shape[0]):
            lower, upper = _linear_bounds(
                weight[channel_idx], input_bounds[node_idx], bias=bias[channel_idx]
            )
            activation, pre = _add_relu(
                model,
                _linear_expr(weight[channel_idx], values, bias=bias[channel_idx]),
                name=f"{layer_name}_linear_node{node_idx}_h{channel_idx}",
                lower=lower,
                upper=upper,
            )
            node_outputs.append(activation)
            node_pre.append(pre)
            node_bounds.append((max(0.0, lower), max(0.0, upper)))
        outputs.append(node_outputs)
        pre_activations.append(node_pre)
        output_bounds.append(node_bounds)
    return outputs, pre_activations, output_bounds


def _build_node_feature_expressions(instance, variables, constraint_type):
    # FÜr jede Operation den GNN-Eingabevektor asl Gurobi-Ausdrücke
    operations = list(variables["real_operations"])
    # Y Maschinenzuordnungen
    Y = variables["Y"]

    node_features = []
    node_bounds = []
    for operation in operations:
        if constraint_type == CONSTRAINT_WEIBULL:
            # Feature 1, processingtime over eta
            processing_over_eta = gp.quicksum(
                Y[operation, machine]
                * float(instance.processing_times[operation, machine])
                / variables["weibull_eta"][machine]
                for machine in instance.eligible_machines[operation]
            )
            # Feature 2: full age at the beginning of the selected setup.
            machine_age_over_eta = gp.quicksum(
                variables["R"][operation, machine]
                / variables["weibull_eta"][machine]
                for machine in instance.eligible_machines[operation]
            )
            # Feature 3: β / 5
            beta_scaled = gp.quicksum(
                Y[operation, machine] * variables["weibull_beta"][machine] / 5.0
                for machine in instance.eligible_machines[operation]
            )
            node_features.append([
                processing_over_eta,
                machine_age_over_eta,
                beta_scaled,
            ])
            # Bounds for Weibull features
            node_bounds.append([
                (
                    min(
                        float(instance.processing_times[operation, machine])
                        / variables["weibull_eta"][machine]
                        for machine in instance.eligible_machines[operation]
                    ),
                    max(
                        float(instance.processing_times[operation, machine])
                        / variables["weibull_eta"][machine]
                        for machine in instance.eligible_machines[operation]
                    ),
                ),
                (
                    (
                        min(
                            variables["machine_initial_age"][machine]
                            / variables["weibull_eta"][machine]
                            for machine in instance.eligible_machines[operation]
                        )
                        if variables.get("gnn_analytic_bounds", False)
                        else 0.0
                    ),
                    max(
                        variables["R_bounds"][operation, machine]
                        / variables["weibull_eta"][machine]
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
    variables["C_max"].ub = H
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
    variables["C_max"].lb = max(
        float(variables["C_max"].lb),
        max(
            earliest_completion(instance.job_end_operations[job])
            for job in instance.jobs
        ),
    )
    variables["job_earliest_completion_bounds"] = dict(earliest_cache)
    variables["job_mandatory_suffix_bounds"] = dict(suffix_cache)


def _add_effective_duration_constraints(
    model,
    variables,
    instance,
):
    """Put every node-level GNN delay directly on the makespan path."""
    operations = list(variables["real_operations"])
    Y, X = variables["Y"], variables["X"]
    node_predictions = variables["predicted_operation_failure_delays"]

    Delta, D = {}, {}
    for node_idx, operation in enumerate(operations):
        prediction = node_predictions[node_idx]
        selected_processing = gp.quicksum(
            Y[operation, machine]
            * float(instance.processing_times[operation, machine])
            for machine in instance.eligible_machines[operation]
        )
        selected_repair_duration = gp.quicksum(
            Y[operation, machine]
            * variables["repair_durations"][operation, machine]
            for machine in instance.eligible_machines[operation]
        )
        # A failure probability cannot exceed one, hence its expected delay
        # cannot exceed the selected repair duration.
        model.addConstr(
            prediction <= selected_repair_duration,
            name=f"predicted_delay_physical_ub[{operation}]",
        )
        if variables.get("gnn_analytic_bounds", False):
            min_processing = min(
                float(instance.processing_times[operation, machine])
                for machine in instance.eligible_machines[operation]
            )
            max_effective_duration = max(
                float(instance.processing_times[operation, machine])
                + variables["repair_durations"][operation, machine]
                for machine in instance.eligible_machines[operation]
            )
        else:
            min_processing = 0.0
            max_effective_duration = (
                max(
                    float(instance.processing_times[operation, machine])
                    for machine in instance.eligible_machines[operation]
                )
                + max(
                    variables["repair_durations"][operation, machine]
                    for machine in instance.eligible_machines[operation]
                )
            )
        # Keep the public Delta mapping for reporting compatibility, but do
        # not create an alias variable and equality for the GNN prediction.
        Delta[operation] = prediction
        D[operation] = model.addVar(
            lb=min_processing,
            ub=max_effective_duration,
            vtype=GRB.CONTINUOUS,
            name=f"D_gnn[{operation}]",
        )
        model.addConstr(
            D[operation]
            == selected_processing + prediction,
            name=f"effective_duration_gnn_def[{operation}]",
        )

    for operation in operations:
        for predecessor in instance.predecessors.get(operation, []):
            model.addConstr(
                variables["C"][operation]
                >= variables["C"][predecessor] + D[operation],
                name=f"effective_gnn_precedence[{predecessor}_before_{operation}]",
            )
        model.addConstr(
            variables["C"][operation] >= D[operation],
            name=f"effective_gnn_completion_lb[{operation}]",
        )

    H = float(variables["H"])
    for operation_i, operation_j, machine in variables["X_index"]:
        model.addConstr(
            variables["C"][operation_i]
            >= variables["C"][operation_j] + D[operation_i]
            - H * (
                2 + X[operation_i, operation_j, machine]
                - Y[operation_i, machine] - Y[operation_j, machine]
            ),
            name=(
                f"effective_gnn_nonoverlap_j_before_i["
                f"{operation_j}_{operation_i}_{machine}]"
            ),
        )
        model.addConstr(
            variables["C"][operation_j]
            >= variables["C"][operation_i] + D[operation_j]
            - H * (
                3 - X[operation_i, operation_j, machine]
                - Y[operation_i, machine] - Y[operation_j, machine]
            ),
            name=(
                f"effective_gnn_nonoverlap_i_before_j["
                f"{operation_i}_{operation_j}_{machine}]"
            ),
        )

    variables.update({
        "Delta": Delta,
        "D": D,
    })
    return D


def _build_relational_local_features(
    instance,
    variables,
    node_features,
    node_bounds,
):
    # The Node-NN baseline receives precisely these three node features.
    # Edge information is available only inside message-passing layers.
    return node_features, node_bounds


def _relational_incoming_edges(
    instance,
    variables,
    include_job_precedence_edges=False,
    include_machine_predecessor_edges=True,
):
    operations = list(variables["real_operations"])
    operation_to_idx = {
        operation: index for index, operation in enumerate(operations)
    }
    incoming = {index: [] for index in range(len(operations))}
    edges, gates = [], []
    for source, target, machine in (
        variables["U_index"] if include_machine_predecessor_edges else []
    ):
        source_idx = operation_to_idx[source]
        target_idx = operation_to_idx[target]
        gate = variables["U"][source, target, machine]
        features = list(transition_edge_values(
            instance,
            source,
            target,
            machine,
            variables["weibull_eta"],
            variables["machine_processing_max"],
        ))
        if include_job_precedence_edges:
            features.append(0.0)
        incoming[target_idx].append(
            (source_idx, gate, features, source, target, machine)
        )
        edge = (source_idx, target_idx, machine)
        edges.append(edge)
        gates.append(gate)
    if include_job_precedence_edges:
        for target in operations:
            for source in instance.predecessors.get(target, []):
                if source not in operation_to_idx:
                    continue
                source_idx = operation_to_idx[source]
                target_idx = operation_to_idx[target]
                features = (
                    [0.0, 0.0, 1.0]
                    if include_machine_predecessor_edges
                    else [1.0]
                )
                incoming[target_idx].append(
                    (source_idx, 1.0, features, source, target, "job")
                )
                edges.append((source_idx, target_idx, "job"))
                gates.append(1.0)
    return incoming, edges, gates


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


def _structured_first_sage_channel_bounds(
    channel_idx,
    target,
    target_incoming,
    root_weight,
    message_weight,
    edge_weight,
    bias,
    variables,
    instance,
):
    """Joint bounds over the feasible first/predecessor machine cases.

    Coordinate-wise interval arithmetic can combine processing, age, beta,
    and edge extrema that belong to different machines.  In the first SAGE
    layer, however, a target is either first on its selected machine or has
    exactly one selected direct predecessor on that same machine.  Enumerating
    these alternatives gives a valid and substantially tighter interval at no
    additional solve cost.
    """
    root = root_weight[channel_idx]
    message = message_weight[channel_idx]
    edge = edge_weight[channel_idx]
    scenarios = []

    for machine in instance.eligible_machines[target]:
        eta = float(variables["weibull_eta"][machine])
        processing = float(instance.processing_times[target, machine]) / eta
        initial_age = float(variables["machine_initial_age"][machine]) / eta
        beta = float(variables["weibull_beta"][machine]) / 5.0
        value = (
            float(bias[channel_idx])
            + float(root[0]) * processing
            + float(root[1]) * initial_age
            + float(root[2]) * beta
        )
        scenarios.append((value, value))

    for (
        _source_idx,
        _gate,
        edge_features,
        source,
        _target,
        machine,
    ) in target_incoming:
        eta = float(variables["weibull_eta"][machine])
        target_processing = (
            float(instance.processing_times[target, machine]) / eta
        )
        source_processing = (
            float(instance.processing_times[source, machine]) / eta
        )
        beta = float(variables["weibull_beta"][machine]) / 5.0
        edge_contribution = sum(
            float(coefficient) * float(feature)
            for coefficient, feature in zip(edge, edge_features)
        )
        constant = (
            float(bias[channel_idx])
            + float(root[0]) * target_processing
            + float(root[1]) * source_processing
            + float(root[2]) * beta
            + float(message[0]) * source_processing
            + float(message[2]) * beta
            + edge_contribution
        )
        source_age_lower = (
            float(variables["machine_initial_age"][machine]) / eta
        )
        source_age_upper = (
            float(variables["R_bounds"][source, machine]) / eta
        )
        scenarios.append(
            _linear_bounds(
                [float(root[1]) + float(message[1])],
                [(source_age_lower, source_age_upper)],
                bias=constant,
            )
        )

    if not scenarios:
        raise ValueError(
            f"No valid SAGE bound scenarios for target operation {target}."
        )
    return (
        min(lower for lower, _upper in scenarios),
        max(upper for _lower, upper in scenarios),
    )


def _add_structured_first_sage_layer(
    model,
    state_dict,
    layer_name,
    inputs,
    input_bounds,
    incoming,
    variables,
    instance,
    convolution=CONV_SAGE,
    include_job_precedence_edges=False,
):
    """Embed the first SAGE/GCN layer without edge-by-feature products.

    Every operation has at most one active direct machine predecessor.  For
    the three reliability input features, the selected predecessor message
    can therefore be reconstructed linearly from U, the target age recursion,
    and the first-on-machine indicator.  This removes one auxiliary variable
    and four McCormick constraints per candidate edge and input feature while
    preserving the trained SAGE forward pass exactly.
    """
    if len(inputs[0]) != 3:
        raise ValueError(
            "Structured first-layer SAGE/GCN embedding requires exactly the "
            "three reliability node features."
        )

    if convolution == CONV_GCN:
        root_weight = state_dict[f"{layer_name}.lin.weight"]
        message_weight = root_weight
        edge_weight = state_dict[f"{layer_name}.lin_edge.weight"]
        bias = state_dict[f"{layer_name}.bias"]
    elif convolution == CONV_SAGE:
        root_weight = state_dict[f"{layer_name}.lin_root.weight"]
        message_weight = state_dict[f"{layer_name}.lin_message.weight"]
        edge_weight = state_dict[f"{layer_name}.lin_edge.weight"]
        bias = state_dict[f"{layer_name}.lin_root.bias"]
    else:
        raise ValueError(
            "Structured first-layer embedding supports only SAGE and GCN."
        )
    operations = list(variables["real_operations"])

    output_vectors, pre_vectors, output_bounds = [], [], []
    for target_idx, target in enumerate(operations):
        target_incoming = incoming[target_idx]
        machine_incoming = [
            edge for edge in target_incoming if edge[-1] != "job"
        ]
        job_incoming = [
            edge for edge in target_incoming if edge[-1] == "job"
        ]
        if len(job_incoming) > 1:
            raise ValueError(
                "Every operation may have at most one direct job predecessor."
            )

        predecessor_processing = gp.quicksum(
            gate
            * float(instance.processing_times[source, machine])
            / variables["weibull_eta"][machine]
            for (
                _source_idx,
                gate,
                _edge_features,
                source,
                _target,
                machine,
            ) in machine_incoming
        )
        predecessor_age = (
            inputs[target_idx][1]
            - gp.quicksum(
                variables["machine_first"][target, machine]
                * variables["machine_initial_age"][machine]
                / variables["weibull_eta"][machine]
                for machine in instance.eligible_machines[target]
            )
            - predecessor_processing
        )
        predecessor_beta = gp.quicksum(
            gate * variables["weibull_beta"][machine] / 5.0
            for (
                _source_idx,
                gate,
                _edge_features,
                _source,
                _target,
                machine,
            ) in machine_incoming
        )
        aggregated = [
            predecessor_processing,
            predecessor_age,
            predecessor_beta,
        ]
        if job_incoming:
            job_source_idx = job_incoming[0][0]
            aggregated = [
                value + inputs[job_source_idx][feature_idx]
                for feature_idx, value in enumerate(aggregated)
            ]
        edge_aggregate = []
        for edge_idx in range(edge_weight.shape[1]):
            terms = [
                edge_features[edge_idx] * gate
                for _source_idx, gate, edge_features, *_rest
                in target_incoming
            ]
            edge_aggregate.append(
                gp.quicksum(terms) if terms else 0.0
            )

        node_outputs, node_pre, node_output_bounds = [], [], []
        for channel_idx in range(root_weight.shape[0]):
            expression = _linear_expr(
                root_weight[channel_idx],
                inputs[target_idx],
                bias=bias[channel_idx],
            )
            expression += _linear_expr(
                message_weight[channel_idx], aggregated
            )
            expression += _linear_expr(
                edge_weight[channel_idx], edge_aggregate
            )
            lower, upper = _structured_first_sage_channel_bounds(
                channel_idx,
                target,
                machine_incoming,
                root_weight,
                message_weight,
                edge_weight,
                bias,
                variables,
                instance,
            )
            if job_incoming:
                job_source_idx = job_incoming[0][0]
                job_lower, job_upper = _linear_bounds(
                    message_weight[channel_idx],
                    input_bounds[job_source_idx],
                    bias=edge_weight[channel_idx][-1],
                )
                lower += job_lower
                upper += job_upper
            activation, pre = _add_relu(
                model,
                expression,
                name=(
                    f"{layer_name}_structured_node"
                    f"{target_idx}_h{channel_idx}"
                ),
                lower=lower,
                upper=upper,
            )
            node_outputs.append(activation)
            node_pre.append(pre)
            node_output_bounds.append(
                (max(0.0, lower), max(0.0, upper))
            )
        output_vectors.append(node_outputs)
        pre_vectors.append(node_pre)
        output_bounds.append(node_output_bounds)
    return output_vectors, pre_vectors, output_bounds


def _direct_predecessor_aggregates(
    model,
    layer_name,
    inputs,
    input_bounds,
    incoming,
):
    """Build the active predecessor and edge sums for every target node."""
    input_size = len(inputs[0])
    all_aggregated = []
    all_aggregated_bounds = []
    all_edge_aggregates = []
    all_edge_bounds = []
    for target_idx in range(len(inputs)):
        aggregated = []
        aggregated_bounds = []
        for feature_idx in range(input_size):
            gated_values = []
            fixed_lower = fixed_upper = 0.0
            lower_candidates = [0.0]
            upper_candidates = [0.0]
            for (
                source_idx,
                gate,
                _edge_features,
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
                    lower_candidates.append(lower)
                    upper_candidates.append(upper)
            aggregated.append(
                gp.quicksum(gated_values) if gated_values else 0.0
            )
            aggregated_bounds.append(
                (
                    fixed_lower + min(lower_candidates),
                    fixed_upper + max(upper_candidates),
                )
            )

        edge_aggregate = []
        edge_bounds = []
        edge_feature_size = int(getattr(
            model,
            "_gnn_edge_feature_size",
            len(RELIABILITY_EDGE_FEATURE_NAMES),
        ))
        for edge_idx in range(edge_feature_size):
            terms = [
                edge_features[edge_idx] * gate
                for _source_idx, gate, edge_features, *_rest
                in incoming[target_idx]
            ]
            fixed_value = sum(
                edge_features[edge_idx]
                for _source_idx, gate, edge_features, *_rest
                in incoming[target_idx]
                if isinstance(gate, (int, float)) and float(gate) == 1.0
            )
            candidates = [
                edge_features[edge_idx]
                for _source_idx, gate, edge_features, *_rest
                in incoming[target_idx]
                if not isinstance(gate, (int, float))
            ]
            edge_aggregate.append(
                gp.quicksum(terms) if terms else 0.0
            )
            edge_bounds.append(
                (
                    fixed_value + min([0.0, *candidates]),
                    fixed_value + max([0.0, *candidates]),
                )
            )
        all_aggregated.append(aggregated)
        all_aggregated_bounds.append(aggregated_bounds)
        all_edge_aggregates.append(edge_aggregate)
        all_edge_bounds.append(edge_bounds)
    return (
        all_aggregated,
        all_aggregated_bounds,
        all_edge_aggregates,
        all_edge_bounds,
    )


def _structured_first_predecessor_aggregates(
    inputs,
    incoming,
    variables,
    instance,
):
    """Use the age recursion to avoid first-layer edge-by-feature products."""
    all_aggregated = []
    all_aggregated_bounds = []
    all_edge_aggregates = []
    all_edge_bounds = []
    operations = list(variables["real_operations"])
    for target_idx, target in enumerate(operations):
        target_incoming = incoming[target_idx]
        predecessor_processing = gp.quicksum(
            gate
            * float(instance.processing_times[source, machine])
            / variables["weibull_eta"][machine]
            for (
                _source_idx,
                gate,
                _edge_features,
                source,
                _target,
                machine,
            ) in target_incoming
        )
        predecessor_age = (
            inputs[target_idx][1]
            - gp.quicksum(
                variables["machine_first"][target, machine]
                * variables["machine_initial_age"][machine]
                / variables["weibull_eta"][machine]
                for machine in instance.eligible_machines[target]
            )
            - predecessor_processing
        )
        predecessor_beta = gp.quicksum(
            gate * variables["weibull_beta"][machine] / 5.0
            for (
                _source_idx,
                gate,
                _edge_features,
                _source,
                _target,
                machine,
            ) in target_incoming
        )
        all_aggregated.append([
            predecessor_processing,
            predecessor_age,
            predecessor_beta,
        ])
        processing_candidates = [0.0]
        age_candidates = [0.0]
        beta_candidates = [0.0]
        for _idx, _gate, _edge, source, _target, machine in target_incoming:
            eta = variables["weibull_eta"][machine]
            processing_candidates.append(
                float(instance.processing_times[source, machine]) / eta
            )
            age_candidates.append(
                float(variables["R_bounds"][source, machine]) / eta
            )
            beta_candidates.append(
                variables["weibull_beta"][machine] / 5.0
            )
        all_aggregated_bounds.append([
            (min(processing_candidates), max(processing_candidates)),
            (0.0, max(age_candidates)),
            (min(beta_candidates), max(beta_candidates)),
        ])

        edge_aggregate = []
        edge_bounds = []
        for edge_idx in range(len(RELIABILITY_EDGE_FEATURE_NAMES)):
            terms = [
                edge_features[edge_idx] * gate
                for _source_idx, gate, edge_features, *_rest
                in target_incoming
            ]
            candidates = [
                edge_features[edge_idx]
                for _source_idx, _gate, edge_features, *_rest
                in target_incoming
            ]
            edge_aggregate.append(
                gp.quicksum(terms) if terms else 0.0
            )
            edge_bounds.append(
                (min([0.0, *candidates]), max([0.0, *candidates]))
            )
        all_edge_aggregates.append(edge_aggregate)
        all_edge_bounds.append(edge_bounds)
    return (
        all_aggregated,
        all_aggregated_bounds,
        all_edge_aggregates,
        all_edge_bounds,
    )


def _add_mpnn_layer(
    model,
    state_dict,
    layer_name,
    inputs,
    input_bounds,
    incoming,
    variables=None,
    instance=None,
):
    root_weight = state_dict[f"{layer_name}.lin_root.weight"]
    root_bias = state_dict[f"{layer_name}.lin_root.bias"]
    message_weight = state_dict[f"{layer_name}.lin_message.weight"]
    edge_weight = state_dict[f"{layer_name}.lin_edge.weight"]
    (
        aggregates,
        aggregate_bounds,
        edge_aggregates,
        edge_bounds,
    ) = (
        _structured_first_predecessor_aggregates(
            inputs, incoming, variables, instance
        )
        if variables is not None and instance is not None
        else _direct_predecessor_aggregates(
            model, layer_name, inputs, input_bounds, incoming
        )
    )

    output_vectors, pre_vectors, output_bounds = [], [], []
    for target_idx, root_values in enumerate(inputs):
        message_outputs = []
        message_output_bounds = []
        for channel_idx in range(message_weight.shape[0]):
            expression = _linear_expr(
                message_weight[channel_idx], aggregates[target_idx]
            )
            expression += _linear_expr(
                edge_weight[channel_idx], edge_aggregates[target_idx]
            )
            lower, upper = _linear_bounds(
                message_weight[channel_idx], aggregate_bounds[target_idx]
            )
            edge_lower, edge_upper = _linear_bounds(
                edge_weight[channel_idx], edge_bounds[target_idx]
            )
            lower += edge_lower
            upper += edge_upper
            activation, _pre = _add_relu(
                model,
                expression,
                name=f"{layer_name}_mpnn_message{target_idx}_h{channel_idx}",
                lower=lower,
                upper=upper,
            )
            message_outputs.append(activation)
            message_output_bounds.append(
                (max(0.0, lower), max(0.0, upper))
            )

        node_outputs, node_pre, node_output_bounds = [], [], []
        for channel_idx in range(root_weight.shape[0]):
            expression = _linear_expr(
                root_weight[channel_idx],
                root_values,
                bias=root_bias[channel_idx],
            )
            expression += message_outputs[channel_idx]
            lower, upper = _linear_bounds(
                root_weight[channel_idx],
                input_bounds[target_idx],
                bias=root_bias[channel_idx],
            )
            lower += message_output_bounds[channel_idx][0]
            upper += message_output_bounds[channel_idx][1]
            activation, pre = _add_relu(
                model,
                expression,
                name=f"{layer_name}_mpnn_node{target_idx}_h{channel_idx}",
                lower=lower,
                upper=upper,
            )
            node_outputs.append(activation)
            node_pre.append(pre)
            node_output_bounds.append(
                (max(0.0, lower), max(0.0, upper))
            )
        output_vectors.append(node_outputs)
        pre_vectors.append(node_pre)
        output_bounds.append(node_output_bounds)
    return output_vectors, pre_vectors, output_bounds


def _scaled_bounds(scale, bounds):
    lower, upper = bounds
    values = (float(scale) * float(lower), float(scale) * float(upper))
    return min(values), max(values)


def _add_gine_layer(
    model,
    state_dict,
    layer_name,
    inputs,
    input_bounds,
    incoming,
    variables=None,
    instance=None,
):
    epsilon = 1.0 + float(state_dict[f"{layer_name}.eps"][0])
    edge_weight = state_dict[f"{layer_name}.lin_edge.weight"]
    first_weight = state_dict[f"{layer_name}.lin1.weight"]
    first_bias = state_dict[f"{layer_name}.lin1.bias"]
    second_weight = state_dict[f"{layer_name}.lin2.weight"]
    second_bias = state_dict[f"{layer_name}.lin2.bias"]
    (
        aggregates,
        aggregate_bounds,
        edge_aggregates,
        edge_bounds,
    ) = (
        _structured_first_predecessor_aggregates(
            inputs, incoming, variables, instance
        )
        if variables is not None and instance is not None
        else _direct_predecessor_aggregates(
            model, layer_name, inputs, input_bounds, incoming
        )
    )

    output_vectors, pre_vectors, output_bounds = [], [], []
    for target_idx, root_values in enumerate(inputs):
        messages = []
        message_bounds = []
        for feature_idx in range(len(root_values)):
            expression = aggregates[target_idx][feature_idx]
            expression += _linear_expr(
                edge_weight[feature_idx], edge_aggregates[target_idx]
            )
            lower, upper = aggregate_bounds[target_idx][feature_idx]
            edge_lower, edge_upper = _linear_bounds(
                edge_weight[feature_idx], edge_bounds[target_idx]
            )
            lower += edge_lower
            upper += edge_upper
            activation, _pre = _add_relu(
                model,
                expression,
                name=f"{layer_name}_gine_message{target_idx}_f{feature_idx}",
                lower=lower,
                upper=upper,
            )
            messages.append(activation)
            message_bounds.append((max(0.0, lower), max(0.0, upper)))

        combined = [
            epsilon * root_values[feature_idx] + messages[feature_idx]
            for feature_idx in range(len(root_values))
        ]
        combined_bounds = []
        for feature_idx in range(len(root_values)):
            root_lower, root_upper = _scaled_bounds(
                epsilon, input_bounds[target_idx][feature_idx]
            )
            combined_bounds.append((
                root_lower + message_bounds[feature_idx][0],
                root_upper + message_bounds[feature_idx][1],
            ))

        first_outputs = []
        first_output_bounds = []
        for channel_idx in range(first_weight.shape[0]):
            expression = _linear_expr(
                first_weight[channel_idx],
                combined,
                bias=first_bias[channel_idx],
            )
            lower, upper = _linear_bounds(
                first_weight[channel_idx],
                combined_bounds,
                bias=first_bias[channel_idx],
            )
            activation, _pre = _add_relu(
                model,
                expression,
                name=f"{layer_name}_gine_mlp{target_idx}_h{channel_idx}",
                lower=lower,
                upper=upper,
            )
            first_outputs.append(activation)
            first_output_bounds.append(
                (max(0.0, lower), max(0.0, upper))
            )

        node_outputs, node_pre, node_output_bounds = [], [], []
        for channel_idx in range(second_weight.shape[0]):
            expression = _linear_expr(
                second_weight[channel_idx],
                first_outputs,
                bias=second_bias[channel_idx],
            )
            lower, upper = _linear_bounds(
                second_weight[channel_idx],
                first_output_bounds,
                bias=second_bias[channel_idx],
            )
            activation, pre = _add_relu(
                model,
                expression,
                name=f"{layer_name}_gine_node{target_idx}_h{channel_idx}",
                lower=lower,
                upper=upper,
            )
            node_outputs.append(activation)
            node_pre.append(pre)
            node_output_bounds.append(
                (max(0.0, lower), max(0.0, upper))
            )
        output_vectors.append(node_outputs)
        pre_vectors.append(node_pre)
        output_bounds.append(node_output_bounds)
    return output_vectors, pre_vectors, output_bounds


def _add_relational_edge_layer(
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
    if convolution == CONV_MPNN:
        return _add_mpnn_layer(
            model, state_dict, layer_name, inputs, input_bounds, incoming
        )
    if convolution == CONV_GINE:
        return _add_gine_layer(
            model, state_dict, layer_name, inputs, input_bounds, incoming
        )
    if convolution == CONV_GCN:
        root_weight = state_dict[f"{layer_name}.lin.weight"]
        message_weight = root_weight
        edge_weight = state_dict[f"{layer_name}.lin_edge.weight"]
        bias = state_dict[f"{layer_name}.bias"]
    elif convolution == CONV_SAGE:
        root_weight = state_dict[f"{layer_name}.lin_root.weight"]
        message_weight = state_dict[
            f"{layer_name}.lin_message.weight"
        ]
        edge_weight = state_dict[f"{layer_name}.lin_edge.weight"]
        bias = state_dict[f"{layer_name}.lin_root.bias"]
    else:
        raise ValueError(f"Unsupported relational convolution: {convolution}")

    output_vectors, pre_vectors, output_bounds = [], [], []
    input_size = len(inputs[0])
    for target_idx, root_values in enumerate(inputs):
        aggregated = []
        aggregated_bounds = []
        for feature_idx in range(input_size):
            gated_values = []
            fixed_lower = fixed_upper = 0.0
            lower_candidates = [0.0]
            upper_candidates = [0.0]
            for (
                source_idx,
                gate,
                _edge_features,
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
                    lower_candidates.append(lower)
                    upper_candidates.append(upper)
            aggregated.append(
                gp.quicksum(gated_values) if gated_values else 0.0
            )
            aggregated_bounds.append(
                (
                    fixed_lower + min(lower_candidates),
                    fixed_upper + max(upper_candidates),
                )
            )
        edge_aggregate = []
        edge_bounds = []
        for edge_idx in range(edge_weight.shape[1]):
            terms = [
                edge_features[edge_idx] * gate
                for _source_idx, gate, edge_features, *_rest
                in incoming[target_idx]
            ]
            fixed_value = sum(
                edge_features[edge_idx]
                for _source_idx, gate, edge_features, *_rest
                in incoming[target_idx]
                if isinstance(gate, (int, float)) and float(gate) == 1.0
            )
            candidates = [
                edge_features[edge_idx]
                for _source_idx, gate, edge_features, *_rest
                in incoming[target_idx]
                if not isinstance(gate, (int, float))
            ]
            edge_aggregate.append(
                gp.quicksum(terms) if terms else 0.0
            )
            edge_bounds.append(
                (
                    fixed_value + min([0.0, *candidates]),
                    fixed_value + max([0.0, *candidates]),
                )
            )

        node_outputs, node_pre, node_output_bounds = [], [], []
        for channel_idx in range(root_weight.shape[0]):
            expression = _linear_expr(
                root_weight[channel_idx],
                root_values,
                bias=bias[channel_idx],
            )
            expression += _linear_expr(
                message_weight[channel_idx], aggregated
            )
            expression += _linear_expr(
                edge_weight[channel_idx], edge_aggregate
            )
            lower, upper = _linear_bounds(
                root_weight[channel_idx],
                input_bounds[target_idx],
                bias=bias[channel_idx],
            )
            message_lower, message_upper = _linear_bounds(
                message_weight[channel_idx], aggregated_bounds
            )
            edge_lower, edge_upper = _linear_bounds(
                edge_weight[channel_idx], edge_bounds
            )
            lower += message_lower + edge_lower
            upper += message_upper + edge_upper
            activation, pre = _add_relu(
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
            node_pre.append(pre)
            node_output_bounds.append(
                (max(0.0, lower), max(0.0, upper))
            )
        output_vectors.append(node_outputs)
        pre_vectors.append(node_pre)
        output_bounds.append(node_output_bounds)
    return output_vectors, pre_vectors, output_bounds


def _add_relational_gnn_prediction(
    model,
    variables,
    instance,
    state_dict,
    metadata,
    constraint_type,
):
    include_job_edges = bool(
        metadata.get("include_job_precedence_edges", False)
    )
    include_machine_edges = bool(
        metadata.get("include_machine_predecessor_edges", True)
    )
    operations = list(variables["real_operations"])
    node_features, node_bounds = _build_node_feature_expressions(
        instance, variables, constraint_type
    )
    local_features, local_bounds = _build_relational_local_features(
        instance, variables, node_features, node_bounds
    )
    incoming, edges, edge_gates = _relational_incoming_edges(
        instance,
        variables,
        include_job_precedence_edges=include_job_edges,
        include_machine_predecessor_edges=include_machine_edges,
    )
    architecture = validate_architecture(
        GRAPH_MODE_FIXED_CANDIDATE,
        metadata.get("convolution", CONV_SAGE),
        metadata.get("aggregation", "sum"),
        metadata.get("pooling", POOL_ADD),
    )
    convolution = architecture["convolution"]

    if (
        convolution in {CONV_SAGE, CONV_GCN}
        and include_machine_edges
        and getattr(model, "_gnn_structured_first_layer", True)
    ):
        hidden_1, hidden_1_pre, hidden_1_bounds = (
            _add_structured_first_sage_layer(
                model,
                state_dict,
                "conv1",
                local_features,
                local_bounds,
                incoming,
                variables,
                instance,
                convolution=convolution,
                include_job_precedence_edges=include_job_edges,
            )
        )
    elif convolution == CONV_MPNN and not include_job_edges:
        hidden_1, hidden_1_pre, hidden_1_bounds = _add_mpnn_layer(
            model,
            state_dict,
            "conv1",
            local_features,
            local_bounds,
            incoming,
            variables=variables,
            instance=instance,
        )
    elif convolution == CONV_GINE and not include_job_edges:
        hidden_1, hidden_1_pre, hidden_1_bounds = _add_gine_layer(
            model,
            state_dict,
            "conv1",
            local_features,
            local_bounds,
            incoming,
            variables=variables,
            instance=instance,
        )
    else:
        hidden_1, hidden_1_pre, hidden_1_bounds = (
            _add_relational_edge_layer(
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
    if num_layers >= 2:
        hidden_2, hidden_2_pre, hidden_2_bounds = (
            _add_relational_edge_layer(
                model,
                state_dict,
                "conv2",
                convolution,
                hidden_1,
                hidden_1_bounds,
                incoming,
            )
        )
    else:
        hidden_2, hidden_2_pre, hidden_2_bounds = (
            hidden_1, hidden_1_pre, hidden_1_bounds
        )
    if num_layers == 3:
        hidden_3, hidden_3_pre, hidden_3_bounds = (
            _add_relational_edge_layer(
                model,
                state_dict,
                "conv3",
                convolution,
                hidden_2,
                hidden_2_bounds,
                incoming,
            )
        )
    else:
        hidden_3, hidden_3_pre, hidden_3_bounds = (
            hidden_2, hidden_2_pre, hidden_2_bounds
        )

    output_weight = state_dict["out.weight"][0]
    output_bias = state_dict["out.bias"][0]
    skip_weight = state_dict.get("out_input.weight")
    skip_bias = state_dict.get("out_input.bias")
    node_probabilities = []
    node_unclipped = []
    node_raw = []
    node_delays = []
    probability_products = {}
    for node_idx, operation in enumerate(operations):
        expression = _linear_expr(
            output_weight,
            hidden_3[node_idx],
            bias=output_bias,
        )
        lower, upper = _linear_bounds(
            output_weight,
            hidden_3_bounds[node_idx],
            bias=output_bias,
        )
        if skip_weight is not None:
            expression += _linear_expr(
                skip_weight[0],
                local_features[node_idx],
                bias=skip_bias[0],
            )
            skip_lower, skip_upper = _linear_bounds(
                skip_weight[0], local_bounds[node_idx], bias=skip_bias[0]
            )
            lower += skip_lower
            upper += skip_upper
        unclipped, raw = _add_relu(
            model,
            expression,
            name=f"relational_probability_unclipped[{operation}]",
            lower=lower,
            upper=upper,
        )
        probability = _add_upper_clip(
            model,
            unclipped,
            name=f"relational_probability[{operation}]",
            lower=max(0.0, lower),
            upper=max(0.0, upper),
            cap=1.0,
        )
        machine_products = []
        for machine in instance.eligible_machines[operation]:
            assignment = variables["Y"][operation, machine]
            product = model.addVar(
                lb=0.0,
                ub=1.0,
                vtype=GRB.CONTINUOUS,
                name=f"relational_probability_Y[{operation},{machine}]",
            )
            model.addConstr(product <= probability)
            model.addConstr(product <= assignment)
            model.addConstr(product >= probability - (1.0 - assignment))
            probability_products[operation, machine] = product
            machine_products.append(
                variables["repair_durations"][operation, machine] * product
            )
        delay = model.addVar(
            lb=0.0,
            ub=max(
                variables["repair_durations"][operation, machine]
                for machine in instance.eligible_machines[operation]
            ),
            vtype=GRB.CONTINUOUS,
            name=f"relational_predicted_delay[{operation}]",
        )
        model.addConstr(delay == gp.quicksum(machine_products))
        node_probabilities.append(probability)
        node_unclipped.append(unclipped)
        node_raw.append(raw)
        node_delays.append(delay)

    prediction = model.addVar(
        lb=0.0,
        ub=GRB.INFINITY,
        vtype=GRB.CONTINUOUS,
        name="predicted_total_failure_delay",
    )
    total_probability = model.addVar(
        lb=0.0,
        ub=float(len(operations)),
        vtype=GRB.CONTINUOUS,
        name="predicted_total_failure_probability",
    )
    raw_probability = model.addVar(
        lb=-GRB.INFINITY,
        vtype=GRB.CONTINUOUS,
        name="predicted_total_failure_probability_raw",
    )
    model.addConstr(prediction == gp.quicksum(node_delays))
    model.addConstr(
        total_probability == gp.quicksum(node_probabilities)
    )
    model.addConstr(raw_probability == gp.quicksum(node_raw))
    variables.update(
        {
            "predicted_total_failure_delay": prediction,
            "predicted_total_failure_probability": total_probability,
            "predicted_total_failure_probability_raw": raw_probability,
            "predicted_operation_failure_delays": node_delays,
            "predicted_operation_failure_probabilities": node_probabilities,
            "predicted_operation_failure_probabilities_unclipped": node_unclipped,
            "predicted_operation_failure_probabilities_raw": node_raw,
            "failure_probability_times_assignment": probability_products,
            "gnn_edges": edges,
            "gnn_edge_activations": {
                edge: gate for edge, gate in zip(edges, edge_gates)
            },
            "gnn_node_features": node_features,
            "gnn_effective_node_features": local_features,
            "gnn_hidden_1": hidden_1,
            "gnn_hidden_1_pre": hidden_1_pre,
            "gnn_hidden_2": hidden_2,
            "gnn_hidden_2_pre": hidden_2_pre,
            "gnn_hidden_3": hidden_3,
            "gnn_hidden_3_pre": hidden_3_pre,
            "gnn_feature_names": metadata["feature_names"],
            "gnn_graph_mode": GRAPH_MODE_FIXED_CANDIDATE,
        }
    )
    return prediction


def _add_gnn_prediction(
    model,
    variables,
    instance,
    state_dict,
    metadata,
    constraint_type,
):  
    return _add_relational_gnn_prediction(
        model,
        variables,
        instance,
        state_dict,
        metadata,
        constraint_type,
    )


def build_fjsp(
    fjsp,
    instance,
    model_path=None,
    metadata_path=None,
    model_seed=DEFAULT_MODEL_SEED,
    convolution=CONV_SAGE,
    aggregation="sum",
    pooling="global_add",
    layers=None,
    hidden_channels=None,
    add_machine_load_lb=True,
    add_schedule_upper_bounds=True,
    objective_weight=None,
    constraint_type=CONSTRAINT_WEIBULL,
    enforce_constraint=False,
    weibull_budget_per_operation=0.4,
    reliability_graph_config=None,
    bound_tightening=False,
    analytic_bounds=None,
    structured_first_layer=True,
    relu_formulation=RELU_BIG_M,
    edge_type=None,
    obbt_bounds=None,
    **_legacy_kwargs,
):
    """Build the fixed-graph probability-GNN makespan formulation.

    ``enforce_constraint`` and ``weibull_budget_per_operation`` remain in the
    public signature for compatibility with existing configurations.  The GNN
    formulation no longer imposes a total-failure-delay budget; predicted
    failure delay is minimized only through each operation's effective duration
    and therefore through ``C_max``.  ``analytic_bounds`` controls the safe
    instance-specific horizon, age, duration and job-time bounds.  The legacy
    ``bound_tightening`` value is used when ``analytic_bounds`` is omitted.
    ``structured_first_layer`` selects the exact compact SAGE/GCN formulation;
    disabling it retains the general edge-product embedding as a control.
    """
    model = fjsp
    # ``bound_tightening`` is retained as a backwards-compatible alias.  A
    # separately named option makes analytical-bound ablations unambiguous.
    if analytic_bounds is None:
        analytic_bounds = bool(bound_tightening)
    else:
        analytic_bounds = bool(analytic_bounds)
    structured_first_layer = bool(structured_first_layer)
    model._gnn_structured_first_layer = structured_first_layer
    relu_formulation = _validate_relu_formulation(relu_formulation)
    model._gnn_relu_formulation = relu_formulation
    model._gnn_obbt_bounds = dict(obbt_bounds or {})
    model._gnn_obbt_applied = 0
    constraint_type = validate_constraint_type(constraint_type)
    if not hasattr(instance, "mu_fail"):
        instance._set_default_nonlinear_parameters()
    graph_cfg = normalize_reliability_graph_config(reliability_graph_config)

    model_seed = int(model_seed)
    requested_architecture = validate_architecture(
        GRAPH_MODE_FIXED_CANDIDATE,
        convolution,
        aggregation,
        pooling,
    )
    configured_dir = architecture_model_dir(
        DEFAULT_MODEL_DIR,
        GRAPH_MODE_FIXED_CANDIDATE,
        requested_architecture["convolution"],
        requested_architecture["aggregation"],
        requested_architecture["pooling"],
        layers=layers,
        hidden_channels=hidden_channels,
    )
    configured_stem = (
        architecture_stem(
            GRAPH_MODE_FIXED_CANDIDATE,
            requested_architecture["convolution"],
            requested_architecture["aggregation"],
            requested_architecture["pooling"],
            target_column(constraint_type),
            model_seed,
            layers=layers,
            hidden_channels=hidden_channels,
        )
        if layers is not None and hidden_channels is not None
        else model_stem(constraint_type, model_seed)
    )
    configured_model = configured_dir / f"{configured_stem}.pt"
    configured_metadata = configured_dir / f"{configured_stem}_meta.json"
    if (
        not configured_model.exists()
        and requested_architecture == {
            "graph_mode": GRAPH_MODE_FIXED_CANDIDATE,
            "convolution": CONV_SAGE,
            "aggregation": "mean",
            "pooling": POOL_ADD,
        }
    ):
        configured_model = _default_model_path(model_seed, constraint_type)
        configured_metadata = _default_metadata_path(model_seed, constraint_type)
    model_path = _resolve_path(
        model_path, configured_model
    )
    metadata_path = _resolve_path(
        metadata_path, configured_metadata
    )
    metadata = _load_metadata(metadata_path)
    include_job_edges = bool(
        metadata.get("include_job_precedence_edges", False)
    )
    include_machine_edges = bool(
        metadata.get("include_machine_predecessor_edges", True)
    )
    if edge_type not in (None, ""):
        expected_edge_types = {
            "machine_only": (True, False),
            "job_only": (False, True),
            "both": (True, True),
        }
        edge_type = str(edge_type).strip().lower()
        if edge_type not in expected_edge_types:
            raise ValueError(
                "edge_type must be 'machine_only', 'job_only' or 'both'."
            )
        if (include_machine_edges, include_job_edges) != expected_edge_types[
            edge_type
        ]:
            raise ValueError(
                f"Configured edge_type={edge_type!r} does not match the "
                "selected GNN metadata."
            )
    edge_feature_names = reliability_edge_feature_names(
        include_job_edges, include_machine_edges
    )
    model._gnn_edge_feature_size = len(edge_feature_names)
    metadata_architecture = validate_architecture(
        metadata.get("graph_mode", GRAPH_MODE_FIXED_CANDIDATE),
        metadata.get("convolution", CONV_SAGE),
        metadata.get("aggregation", "mean"),
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
    metadata_graph_config = normalize_reliability_graph_config(
        metadata.get("reliability_graph_config")
    )
    if metadata_graph_config != graph_cfg:
        raise ValueError(
            "GNN reliability-graph parameters differ from the solver model: "
            f"metadata={reliability_graph_config_dict(metadata_graph_config)}, "
            f"solver={reliability_graph_config_dict(graph_cfg)}."
        )
    if metadata.get("edge_feature_names") != edge_feature_names:
        raise ValueError(
            "GNN edge feature order does not match the relational solver."
        )

    duration_upper = {
        operation: max(
            _repair_duration(instance, operation, machine)
            for machine in instance.eligible_machines[operation]
        )
        for operation in instance.real_operations
    }
    model, variables = build_base_fjsp(
        model,
        instance,
        extra_duration_upper_bounds=duration_upper,
        add_time_constraints=False,
        total_extra_duration_upper_bound=None,
    )
    variables["gnn_analytic_bounds"] = analytic_bounds
    # Serializing every operation is always feasible.  For the selected
    # machine k, its effective duration is at most p_ik + r_ik because the
    # clipped failure probability is at most one.  Taking the per-operation
    # maximum over a common k avoids combining incompatible processing and
    # repair maxima and is therefore a safe schedule upper bound.
    if analytic_bounds:
        variables["H"] = sum(
            max(
                float(instance.processing_times[operation, machine])
                + _repair_duration(instance, operation, machine)
                for machine in instance.eligible_machines[operation]
            )
            for operation in instance.real_operations
        )
    _add_reliability_graph_state(
        model,
        variables,
        instance,
        graph_cfg,
        bound_tightening=analytic_bounds,
    )
    if add_schedule_upper_bounds:
        _add_schedule_upper_bounds(variables)
        if analytic_bounds:
            _add_job_time_bounds(variables, instance)
    _add_gnn_prediction(
        model,
        variables,
        instance,
        state_dict,
        metadata,
        constraint_type,
    )
    _add_effective_duration_constraints(
        model,
        variables,
        instance,
    )
    if add_machine_load_lb:
        for machine in variables["machines"]:
            eligible = [
                operation
                for operation in variables["real_operations"]
                if machine in instance.eligible_machines[operation]
            ]
            if eligible:
                model.addConstr(
                    variables["C_max"]
                    >= gp.quicksum(
                        variables["Y"][operation, machine]
                        * float(
                            instance.processing_times[operation, machine]
                        )
                        + variables[
                            "repair_durations"
                        ][operation, machine]
                        * variables[
                            "failure_probability_times_assignment"
                        ][operation, machine]
                        for operation in eligible
                    ),
                    name=f"relational_gnn_machine_load_lb[{machine}]",
                )
    variables.update(
        {
            "gnn_model_path": str(model_path),
            "gnn_metadata_path": str(metadata_path),
            "gnn_metadata": metadata,
            "gnn_objective_weight": float(objective_weight or 0.0),
            "gnn_used_in_objective": True,
            "gnn_schedule_upper_bounds": bool(add_schedule_upper_bounds),
            "gnn_bound_tightening": analytic_bounds,
            "gnn_analytic_bounds": analytic_bounds,
            "gnn_structured_first_layer": structured_first_layer,
            "gnn_relu_formulation": relu_formulation,
            "gnn_obbt_bound_count": len(model._gnn_obbt_bounds),
            "gnn_obbt_applied_count": int(model._gnn_obbt_applied),
            "gnn_include_job_precedence_edges": include_job_edges,
            "gnn_include_machine_predecessor_edges": include_machine_edges,
            "gnn_edge_type": edge_type or (
                "both" if include_machine_edges and include_job_edges
                else "job_only" if include_job_edges
                else "machine_only"
            ),
            "constraint_type": constraint_type,
            "constraint_target_column": expected_target,
            "budget_constraint": None,
            "constraint_budget": None,
            "constraint_enforced": False,
            "weibull_budget_per_operation": float(
                weibull_budget_per_operation
            ),
            "formulation": "gnn_transition_hazard_weibull_unconstrained",
        }
    )
    model.setObjective(variables["C_max"], GRB.MINIMIZE)

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
