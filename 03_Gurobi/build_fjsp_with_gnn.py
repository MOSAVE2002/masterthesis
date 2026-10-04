"""Embed a trained repair-buffer GNN exactly into the FJSP MILP.

The module loads a trained PyTorch state dictionary and reconstructs its
forward pass with Gurobi variables and constraints. It creates the operation
features from scheduling decisions, propagates valid interval bounds, models
ReLU activations exactly and aggregates operation embeddings into one
nonnegative expected repair buffer per job. These buffers extend the shared
FJSP formulation through soft due dates and the common economic objective.
"""

import importlib
import json
import math
from pathlib import Path

from helper.local_buffer import JOB_TARGET, LABEL_METHOD, TARGET_COLUMN

import gurobipy as gp
from gurobipy import GRB
from helper.economic_objective import (
    add_economic_cost_objective,
    add_due_date_tardiness_constraints,
)
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
base_fjsp = importlib.import_module("03_Gurobi.build_fjsp")

from helper.sequence_setup import (
    RELIABILITY_GNN_GRAPH_SCHEMA,
    RELIABILITY_GNN_OUTPUT_HEAD,
    RELIABILITY_SERVICE_SCOPE,
    add_reliability_graph_variables,
    reliability_graph_config_dict,
    reliability_node_feature_names,
)
from helper.stochastic_fjsp import (
    normalize_machine_profile_config,
    stochastic_parameters,
)
gnn_architecture = importlib.import_module(
    "04_GraphNeuralNetworks.models.gnn_architecture"
)
CONV_LINEAR = gnn_architecture.CONV_LINEAR
CONV_SAGE = gnn_architecture.CONV_SAGE
CONV_JOB = gnn_architecture.CONV_JOB
VALID_LAYER_COUNTS = gnn_architecture.VALID_LAYER_COUNTS
validate_architecture = gnn_architecture.validate_architecture


def resolve_path(path_value):
    """Resolve a required artifact path relative to the project root.

    Args:
        path_value: Absolute path or project-relative path to a model artifact.

    Returns:
        A :class:`pathlib.Path` that is absolute or anchored at ``ROOT_DIR``.

    Raises:
        ValueError: If no path value was supplied.
    """
    if path_value in (None, ""):
        raise ValueError("Both model_path and metadata_path are required.")
    path = Path(path_value)
    if not path.is_absolute():
        path = ROOT_DIR / path
    return path


def load_metadata(metadata_path, instance):
    """Load and validate the metadata required for an exact GNN embedding.

    The checks ensure that architecture, output definition, graph schema,
    feature order, message-passing edges and machine profiles agree with the
    active solver implementation and the instance being optimized. A model is
    rejected before construction if any training assumption is incompatible.

    Args:
        metadata_path: JSON file stored next to the trained model weights.
        instance: FJSP instance whose machine profiles must match the training
            configuration recorded in the metadata.

    Returns:
        A pair containing the complete metadata dictionary and the normalized
        architecture dictionary.

    Raises:
        FileNotFoundError: If the metadata file does not exist.
        ValueError: If the saved model is incompatible with the active graph,
            features, labels, architecture or machine profiles.
    """
    if not metadata_path.exists():
        raise FileNotFoundError(f"GNN metadata file not found: {metadata_path}")

    with metadata_path.open(encoding="utf-8") as file:
        metadata = json.load(file)

    architecture = validate_architecture(metadata["convolution"])
    if any(
        metadata[key] != value
        for key, value in architecture.items()
    ):
        raise ValueError(
            "GNN metadata contains an incompatible graph architecture."
        )
    convolution = architecture["convolution"]
    if metadata["output_head"] != RELIABILITY_GNN_OUTPUT_HEAD:
        raise ValueError(
            "The embedded repair-buffer GNN requires output_head="
            f"{RELIABILITY_GNN_OUTPUT_HEAD!r}."
        )
    if metadata["target_column"] != TARGET_COLUMN:
        raise ValueError(
            "The embedded GNN requires local midpoint repair buffers."
        )
    if metadata["job_target"] != JOB_TARGET:
        raise ValueError(
            "The embedded GNN requires one local repair buffer per job."
        )
    if metadata["job_repair_buffer_label_method"] != LABEL_METHOD:
        raise ValueError(
            "Unknown GNN labels; regenerate local labels and retrain."
        )
    if (
        convolution in {CONV_SAGE, CONV_JOB}
        and not metadata["include_job_precedence_edges"]
    ):
        raise ValueError("The job-buffer GNN requires fixed job edges.")
    if (
        convolution == CONV_SAGE
        and not metadata["include_machine_predecessor_edges"]
    ):
        raise ValueError("The active GNN pipeline requires machine edges.")
    if (
        convolution != CONV_SAGE
        and metadata["include_machine_predecessor_edges"]
    ):
        raise ValueError(
            f"{convolution} metadata must not enable machine messages."
        )
    expected_machine_scope = "direct" if convolution == CONV_SAGE else "none"
    if metadata["machine_predecessor_edge_scope"] != expected_machine_scope:
        raise ValueError(
            f"The embedded {convolution} model requires "
            "machine_predecessor_edge_scope="
            f"{expected_machine_scope!r}."
        )
    if metadata["graph_schema"] != RELIABILITY_GNN_GRAPH_SCHEMA:
        raise ValueError(
            "The embedded GNN requires graph_schema="
            f"{RELIABILITY_GNN_GRAPH_SCHEMA!r}."
        )
    expected_message_passing = (
        "none" if convolution == CONV_LINEAR else "source_node_states_only"
    )
    if metadata["message_passing"] != expected_message_passing:
        raise ValueError(
            f"The embedded {convolution} model requires message_passing="
            f"{expected_message_passing!r}."
        )
    expected_features = reliability_node_feature_names()
    if metadata["feature_names"] != expected_features:
        raise ValueError(
            "GNN feature order does not match the embedded FJSP features: "
            f"expected {expected_features}, got {metadata['feature_names']}."
        )
    if int(metadata["input_size"]) != len(expected_features):
        raise ValueError(
            f"GNN input_size={metadata['input_size']} does not match "
            f"{len(expected_features)} embedded FJSP features."
        )
    num_layers = int(metadata["num_graphsage_layers"])
    if num_layers not in VALID_LAYER_COUNTS:
        choices = ", ".join(map(str, sorted(VALID_LAYER_COUNTS)))
        raise ValueError(f"GNN metadata must specify one of {choices} layers.")
    if int(metadata["hidden_channels"]) <= 0:
        raise ValueError("GNN hidden_channels must be positive.")

    expected_graph_config = reliability_graph_config_dict()
    metadata_graph_config = metadata["reliability_graph_config"]
    if metadata_graph_config != expected_graph_config:
        raise ValueError(
            "GNN reliability-graph parameters differ from the solver model: "
            f"metadata={metadata_graph_config}, "
            f"solver={expected_graph_config}."
        )
    metadata_profiles = metadata.get("machine_profile_config")
    if metadata_profiles is None:
        raise ValueError(
            "GNN metadata does not contain machine_profile_config. "
            "Regenerate the training data and retrain the GNN before solving."
        )
    expected_profiles = normalize_machine_profile_config(metadata_profiles)
    actual_profiles = normalize_machine_profile_config(
        instance.machine_profile_config
    )
    if expected_profiles != actual_profiles:
        raise ValueError(
            "GNN machine profiles differ from the solved instance. "
            "Regenerate the training data and retrain the GNN."
        )
    return metadata, architecture


def load_state_dict(model_path):
    """Load trained PyTorch parameters as CPU-backed NumPy arrays.

    Loading with ``weights_only=True`` restricts deserialization to the tensor
    state required for the mathematical embedding. NumPy arrays are returned
    because the coefficients become constants in Gurobi expressions.

    Args:
        model_path: Path to the serialized PyTorch state dictionary.

    Returns:
        Mapping from layer parameter names to NumPy arrays.

    Raises:
        FileNotFoundError: If the model file does not exist.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"GNN model file not found: {model_path}")

    return {
        key: value.detach().cpu().numpy()
        for key, value in torch.load(
            model_path, map_location="cpu", weights_only=True
        ).items()
    }


def add_stochastic_machine_state(variables, instance):
    """Attach stochastic machine parameters to the shared variable mapping.

    Args:
        variables: Mutable dictionary returned by the base FJSP builder.
        instance: FJSP instance providing Weibull and repair parameters.

    Side Effects:
        Adds ``weibull_alpha``, ``weibull_beta`` and ``repair_rate`` mappings
        to ``variables`` for later node-feature construction.
    """
    parameters = stochastic_parameters(instance)
    variables.update({
        "weibull_alpha": parameters["alpha"],
        "weibull_beta": parameters["beta"],
        "repair_rate": parameters["repair_rate"],
    })


def linear_expr(coefficients, values, bias=0.0):
    """Build an affine Gurobi expression with fixed trained coefficients.

    Args:
        coefficients: Numeric weights of one neural-network output channel.
        values: Gurobi expressions or numeric inputs in matching order.
        bias: Constant intercept added to the weighted sum.

    Returns:
        A :class:`gurobipy.LinExpr` representing the affine transformation.
    """
    expr = gp.LinExpr(float(bias))
    for coefficient, value in zip(coefficients, values):
        coefficient = float(coefficient)
        if abs(coefficient) > 1e-12:
            expr += coefficient * value
    return expr


def linear_bounds(coefficients, bounds, bias=0.0):
    """Propagate independent input intervals through an affine transformation.

    For every coefficient, the appropriate interval endpoint is selected from
    its sign. The resulting bounds are used as valid big-M constants for the
    exact ReLU formulation.

    Args:
        coefficients: Numeric weights of one affine output.
        bounds: ``(lower, upper)`` interval for every corresponding input.
        bias: Constant intercept of the affine transformation.

    Returns:
        The lower and upper bound of the affine output.
    """
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


def add_relu(model, expression, name, lower=None, upper=None):
    """Represent one bounded ReLU activation exactly in a Gurobi model.

    If the pre-activation interval lies entirely on one side of zero, the ReLU
    is fixed or kept linear. If the interval crosses zero, one binary phase
    variable and the ideal single-neuron mixed-integer formulation are added.

    Args:
        model: Gurobi model receiving the auxiliary variables and constraints.
        expression: Affine expression defining the neuron pre-activation.
        name: Base name for all generated variables and constraints.
        lower: Finite lower bound of the pre-activation.
        upper: Finite upper bound of the pre-activation.

    Returns:
        The continuous Gurobi variable representing the ReLU output.

    Raises:
        ValueError: If bounds are absent, nonfinite or inconsistent.
    """
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


def add_linear_layer(
    model,
    state_dict,
    layer_name,
    input_vectors,
    input_bounds,
):
    """Embed one node-wise linear layer followed by exact ReLU activations.

    This path implements the linear baseline architecture. Every operation
    node is transformed independently; no graph edges contribute messages.

    Args:
        model: Gurobi model receiving the embedded neural-network layer.
        state_dict: Trained weights and biases represented as NumPy arrays.
        layer_name: State-dictionary prefix of the layer to embed.
        input_vectors: Feature expressions grouped by operation node.
        input_bounds: Matching intervals for every input expression.

    Returns:
        A pair containing the ReLU output variables for all nodes and their
        propagated nonnegative bounds.
    """
    weight = state_dict[f"{layer_name}.weight"]
    bias = state_dict[f"{layer_name}.bias"]
    outputs, output_bounds = [], []
    for node_idx, values in enumerate(input_vectors):
        node_outputs, node_bounds = [], []
        for channel_idx in range(weight.shape[0]):
            lower, upper = linear_bounds(
                weight[channel_idx], input_bounds[node_idx], bias=bias[channel_idx]
            )
            activation = add_relu(
                model,
                linear_expr(weight[channel_idx], values, bias=bias[channel_idx]),
                name=f"{layer_name}_linear_node{node_idx}_h{channel_idx}",
                lower=lower,
                upper=upper,
            )
            node_outputs.append(activation)
            node_bounds.append((max(0.0, lower), max(0.0, upper)))
        outputs.append(node_outputs)
        output_bounds.append(node_bounds)
    return outputs, output_bounds


def build_node_feature_expressions(instance, variables):
    """Build the normalized operation features consumed by the trained GNN.

    The feature expressions combine the operation midpoint with the selected
    machine's Weibull scale, shape and repair rate. Existing exact products of
    midpoint and assignment variables ensure that each expression uses only
    the parameters of the selected machine. Valid feature intervals are built
    simultaneously for subsequent ReLU formulations.

    Args:
        instance: FJSP instance containing eligibility and processing times.
        variables: Solver-variable dictionary containing assignments,
            midpoint-assignment products, horizon and stochastic parameters.

    Returns:
        A pair containing one feature-expression vector per operation and the
        matching ``(lower, upper)`` interval vectors.
    """
    Y = variables["Y"]
    horizon = float(variables["H"])
    products = variables["midpoint_times_assignment"]
    features, bounds = [], []
    for operation in variables["real_operations"]:
        machines = instance.eligible_machines[operation]
        alpha = variables["weibull_alpha"]
        beta = variables["weibull_beta"]
        rate = variables["repair_rate"]
        constants = [
            {m: rate[m] * alpha[m] / 10.0 for m in machines},
            {m: beta[m] / 5.0 for m in machines},
            {m: 1.0 / (60.0 * rate[m]) for m in machines},
        ]
        features.append([
            gp.quicksum(products[operation, m] / alpha[m] for m in machines),
            *[gp.quicksum(Y[operation, m] * values[m] for m in machines)
              for values in constants],
        ])
        # S>=0 and C<=H imply p/2 <= T <= H-p/2 for the selected machine.
        bounds.append([
            (min(0.5 * instance.processing_times[operation, m] / alpha[m]
                 for m in machines),
             max((horizon - 0.5 * instance.processing_times[operation, m]) / alpha[m]
                 for m in machines)),
            *[(min(values.values()), max(values.values())) for values in constants],
        ])
    return features, bounds


def add_job_time_bounds(variables, instance):
    """Tighten completion-time bounds using technological job precedence.

    Earliest completions and latest feasible completions are derived from the
    minimum eligible-machine duration of each operation. Tighter bounds improve
    the feature intervals and therefore the embedded ReLU formulation.

    Args:
        variables: Solver-variable dictionary containing completion variables
            and the common scheduling horizon.
        instance: FJSP instance containing ordered job operations, processing
            times and eligible machines.

    Side Effects:
        Updates the lower and upper bounds of ``variables["C"]`` in place.
    """
    minimum_duration = {
        operation: min(
            float(instance.processing_times[operation, machine])
            for machine in instance.eligible_machines[operation]
        )
        for operation in variables["real_operations"]
    }
    horizon = float(variables["H"])
    for job_operations in instance.jobs.values():
        earliest_completion = 0.0
        remaining_duration = sum(
            minimum_duration[operation] for operation in job_operations
        )
        for operation in job_operations:
            duration = minimum_duration[operation]
            earliest_completion += duration
            remaining_duration -= duration
            completion = variables["C"][operation]
            completion.lb = max(
                float(completion.lb), earliest_completion
            )
            completion.ub = min(
                float(completion.ub), horizon - remaining_duration
            )


def relational_incoming_edges(
    instance,
    variables,
    convolution,
):
    """Collect the incoming edges used by the selected GNN architecture.

    Direct machine-predecessor edges are controlled by binary ``U`` variables
    and are included only for the SAGE architecture. Fixed technological job
    edges use a constant gate of one and are included for both SAGE and the
    job-precedence variant. The linear baseline receives no incoming edges.

    Args:
        instance: FJSP instance providing fixed job predecessors.
        variables: Solver-variable dictionary containing operation order and,
            for SAGE, direct machine-predecessor variables.
        convolution: Normalized architecture identifier.

    Returns:
        Dictionary indexed by target-node position. Each value contains tuples
        of source index, edge gate, operation IDs and machine ID. Fixed job
        edges use ``-1`` as their machine identifier.
    """
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


def gate_hidden_value(
    model,
    value,
    gate,
    lower,
    upper,
    name,
):
    """Model the exact product of a bounded hidden value and an edge gate.

    Fixed job-precedence edges return the original value directly. Variable
    machine edges use four linear constraints to represent ``gate * value``
    exactly for a binary gate and a bounded continuous value.

    Args:
        model: Gurobi model receiving the auxiliary product variable.
        value: Hidden-state expression transmitted by the source node.
        gate: Binary edge variable or the constant ``1.0``.
        lower: Lower bound of ``value``.
        upper: Upper bound of ``value``.
        name: Name assigned to the auxiliary variable.

    Returns:
        The original value for a fixed active edge or a gated Gurobi variable.
    """
    if isinstance(gate, (int, float)) and float(gate) == 1.0:
        return value
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


def add_relational_node_layer(
    model,
    state_dict,
    layer_name,
    convolution,
    inputs,
    input_bounds,
    incoming,
):
    """Embed one trained linear or relational GNN layer in the MILP.

    For relational architectures, bounded predecessor states are first gated
    by their active edges and summed feature-wise. Each output channel then
    combines a root-node transformation with a predecessor-message
    transformation and applies an exact ReLU. Interval arithmetic propagates
    valid bounds through both transformations. The linear architecture is
    delegated to :func:`add_linear_layer`.

    Args:
        model: Gurobi model receiving neural-network variables and constraints.
        state_dict: Trained weights and biases represented as NumPy arrays.
        layer_name: State-dictionary prefix of the hidden layer.
        convolution: Linear, direct-machine-edge or job-edge architecture.
        inputs: Input expressions grouped by node and feature.
        input_bounds: Matching input intervals grouped by node and feature.
        incoming: Incoming-edge tuples for every target node.

    Returns:
        A pair containing all activated node states and their propagated
        nonnegative intervals.
    """
    if convolution == CONV_LINEAR:
        return add_linear_layer(
            model, state_dict, layer_name, inputs, input_bounds
        )
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
                    gate_hidden_value(
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
            expression = linear_expr(
                root_weight[channel_idx],
                root_values,
                bias=bias[channel_idx],
            )
            expression += linear_expr(
                message_weight[channel_idx], aggregated
            )
            lower, upper = linear_bounds(
                root_weight[channel_idx],
                input_bounds[target_idx],
                bias=bias[channel_idx],
            )
            message_lower, message_upper = linear_bounds(
                message_weight[channel_idx], aggregated_bounds
            )
            lower += message_lower
            upper += message_upper
            activation = add_relu(
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


def add_relational_gnn_output(
    model,
    variables,
    instance,
    state_dict,
    metadata,
    architecture,
):
    """Embed the complete trained GNN and create one buffer per job.

    Node features are constructed from scheduling variables and propagated
    through the configured hidden layers. Hidden states and original inputs are
    summed over the operations belonging to each job, matching the training
    model's global-add pooling and skip connection. A final ReLU enforces the
    nonnegative job-specific expected repair-buffer output.

    Args:
        model: Gurobi model receiving the complete neural-network embedding.
        variables: Shared FJSP variable and metadata dictionary.
        instance: FJSP instance defining operations and job memberships.
        state_dict: Trained neural-network parameters as NumPy arrays.
        metadata: Validated model metadata including the number of layers.
        architecture: Validated normalized architecture dictionary.

    Returns:
        Dictionary mapping each job ID to its nonnegative Gurobi buffer
        variable.
    """
    operations = list(variables["real_operations"])
    node_features, node_bounds = build_node_feature_expressions(
        instance, variables
    )
    local_features, local_bounds = node_features, node_bounds
    convolution = architecture["convolution"]
    incoming = relational_incoming_edges(
        instance, variables, convolution
    )

    hidden, hidden_bounds = (
        add_relational_node_layer(
            model,
            state_dict,
            "conv1",
            convolution,
            local_features,
            local_bounds,
            incoming,
        )
    )
    num_layers = int(metadata["num_graphsage_layers"])
    for layer_number in range(2, num_layers + 1):
        hidden, hidden_bounds = (
            add_relational_node_layer(
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
    skip_weight = state_dict["out_input.weight"][0]
    skip_bias = state_dict["out_input.bias"][0]
    operation_to_index = {
        operation: index for index, operation in enumerate(operations)
    }
    output_expressions = {}
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
        expression = linear_expr(
            output_weight, pooled_hidden, bias=output_bias
        )
        expression += linear_expr(
            skip_weight, pooled_local, bias=skip_bias
        )
        raw = model.addVar(
            lb=-GRB.INFINITY, name=f"gnn_raw_job_repair_buffer[{job}]"
        )
        buffer = model.addVar(
            lb=0.0, name=f"gnn_job_repair_buffer[{job}]"
        )
        model.addConstr(raw == expression, name=f"gnn_raw_output_def[{job}]")
        model.addGenConstrMax(
            buffer, [raw], constant=0.0,
            name=f"gnn_repair_buffer_relu[{job}]",
        )
        output_expressions[job] = buffer
    return output_expressions


def add_midpoint_state(model, variables, instance):
    """Create operation starts, midpoints and exact assignment products.

    For each operation, nominal start and midpoint variables are linked to its
    completion time and assignment-dependent processing duration. Standard
    binary-continuous product constraints create ``midpoint * assignment`` for
    every eligible machine, which permits linear selected-machine features.

    Args:
        model: Gurobi model receiving variables and linking constraints.
        variables: Shared FJSP dictionary containing completion, assignment and
            horizon information.
        instance: FJSP instance containing machine eligibility and durations.

    Side Effects:
        Adds ``S``, ``T`` and ``midpoint_times_assignment`` to ``variables``.
    """
    horizon = float(variables["H"])
    starts, midpoints, midpoint_times_assignment = {}, {}, {}
    for operation in variables["real_operations"]:
        duration = gp.quicksum(
            float(instance.processing_times[operation, machine])
            * variables["Y"][operation, machine]
            for machine in instance.eligible_machines[operation]
        )
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
        "midpoint_times_assignment": midpoint_times_assignment,
    })


def build_fjsp(
    model,
    instance,
    model_path,
    metadata_path,
    facility_cost_per_time=1.0,
    tardiness_cost_per_time=1.0,
):
    """Build the complete FJSP MILP with embedded GNN repair buffers.

    The function validates and loads the trained model, constructs the shared
    FJSP formulation, adds midpoint-dependent features and embeds the selected
    neural architecture. For SAGE models, direct machine-predecessor graph
    variables are also created. Predicted job buffers enter the soft due-date
    constraints before the common economic objective is installed.

    Args:
        model: Empty or partially configured Gurobi model to extend.
        instance: FJSP instance to optimize.
        model_path: Absolute or project-relative PyTorch weight path.
        metadata_path: Absolute or project-relative model metadata path.
        facility_cost_per_time: Nonnegative cost of one makespan time unit.
        tardiness_cost_per_time: Nonnegative cost of one tardiness time unit.

    Returns:
        The updated Gurobi model and its variable/metadata dictionary.

    Raises:
        FileNotFoundError: If a required model artifact does not exist.
        ValueError: If the model metadata is incompatible with the instance or
            the active embedding implementation.
    """
    model_path = resolve_path(model_path)
    metadata_path = resolve_path(metadata_path)
    metadata, architecture = load_metadata(metadata_path, instance)
    state_dict = load_state_dict(model_path)
    model, variables = base_fjsp.build_core_fjsp(model, instance)
    add_midpoint_state(model, variables, instance)
    add_stochastic_machine_state(variables, instance)
    if architecture["convolution"] == CONV_SAGE:
        add_reliability_graph_variables(model, variables, instance)
    add_job_time_bounds(variables, instance)
    job_repair_buffers = add_relational_gnn_output(
        model,
        variables,
        instance,
        state_dict,
        metadata,
        architecture,
    )
    add_due_date_tardiness_constraints(
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
    variables.update({
        "service_scope": RELIABILITY_SERVICE_SCOPE,
        "job_repair_buffer_label_method": LABEL_METHOD,
        "gnn_metadata": metadata,
        "formulation": "gnn_unscaled_local_repair_buffer_v16",
    })

    model.update()
    return model, variables
