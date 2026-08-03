import importlib
import sys
import csv
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
from helper.sequence_setup import (
    RELIABILITY_EDGE_FEATURE_NAMES,
    reliability_graph_config_dict,
    reliability_node_feature_names,
    transition_edge_values,
)

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

_gnn_architecture = importlib.import_module(
    "04_GraphNeuralNetworks.models.gnn_architecture"
)
CONV_SAGE = _gnn_architecture.CONV_SAGE
architecture_slug = _gnn_architecture.architecture_slug
validate_architecture = _gnn_architecture.validate_architecture

CONFIG_PATH = ROOT_DIR / "config.json"
if CONFIG_PATH.exists():
    with CONFIG_PATH.open(encoding="utf-8") as file:
        CONFIG = json.load(file)
else:
    CONFIG = {}
GNN_CONFIG = CONFIG.get("training", {}).get("gnn", {})
GNN_ARCHITECTURE = (
    GNN_CONFIG.get("combinations", [None])[0]
    if GNN_CONFIG.get("combinations")
    else GNN_CONFIG.get("architecture", {})
)
import gurobipy as gp
from gurobipy import GRB

_instance_generator = importlib.import_module("01_generator.instance_generator")
FJSPData = _instance_generator.FJSPData
generated_instance_names = _instance_generator.generated_instance_names
load_generated_instance = _instance_generator.load_generated_instance

_gurobi = importlib.import_module("03_Gurobi.build_fjsp")
build_fjsp_gurobi = _gurobi.build_fjsp
write_solution_file_gurobi = _gurobi.write_solution_file

_gurobi_gnn = importlib.import_module("03_Gurobi.build_fjsp_with_gnn")
build_fjsp_gurobi_gnn = _gurobi_gnn.build_fjsp
write_solution_file_gurobi_gnn = _gurobi_gnn.write_solution_file

_gurobi_nonlinear = importlib.import_module("03_Gurobi.build_fjsp_with_nonlinear")
build_fjsp_gurobi_nonlinear = _gurobi_nonlinear.build_fjsp
build_fjsp_gurobi_linear_labeled = (
    _gurobi_nonlinear.build_linear_schedule_for_analytical_labeling
)
analytical_weibull_labels = _gurobi_nonlinear.analytical_weibull_labels
plot_candidate_graph_gurobi_nonlinear = _gurobi_nonlinear.plot_candidate_graph
plot_solution_schedule_gurobi_nonlinear = (
    _gurobi_nonlinear.plot_solution_schedule
)
plot_solution_graph_gurobi_nonlinear = _gurobi_nonlinear.plot_solution_graph_from_a
write_solution_file_gurobi_nonlinear = _gurobi_nonlinear.write_solution_file


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _solver_solution_dir(solver_name, architecture=None):
    path = ROOT_DIR / "02_data" / "fjsp_solutions" / solver_name
    if architecture is not None:
        requested_architecture = dict(architecture)
        architecture = validate_architecture(
            requested_architecture["graph_mode"],
            requested_architecture["convolution"],
            requested_architecture["aggregation"],
            requested_architecture["pooling"],
        )
        layers = requested_architecture.get("layers")
        hidden_channels = requested_architecture.get("hidden_channels")
        path = (
            path
            / architecture["graph_mode"]
            / architecture_slug(
                **architecture,
                layers=layers,
                hidden_channels=hidden_channels,
            )
        )
    path.mkdir(parents=True, exist_ok=True)
    return path


GNN_GRAPH_FIXED_CANDIDATE = "fixed_candidate"
_POOL_STRUCTURES_BY_OUTPUT = {}


def _validate_gnn_graph_mode(graph_mode):
    if graph_mode != GNN_GRAPH_FIXED_CANDIDATE:
        raise ValueError("gnn_graph_mode muss 'fixed_candidate' sein.")
    return graph_mode


def _default_gnn_graph_mode():
    return _validate_gnn_graph_mode(
        GNN_ARCHITECTURE.get("graph_mode", GNN_GRAPH_FIXED_CANDIDATE)
    )


def _sample_seed(instance_name: str, sample_idx: int, base_seed) -> int:
    seed_source = f"{instance_name}:{sample_idx}:{base_seed}" # zeichenkette
    digest = hashlib.sha256(seed_source.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") # Aus den ersten 8 Bytes des hashs wird eine Ganzzahl erzeugt


def _validate_fix_ratio_range(fix_ratio_min: float, fix_ratio_max: float):
    if not 0.0 <= fix_ratio_min <= fix_ratio_max <= 1.0:
        raise ValueError("fix_ratio_min und fix_ratio_max muessen zwischen 0 und 1 liegen.")


def _generate_random_fixed_y(
    instance,
    instance_name: str,
    sample_idx: int,
    base_seed,
):
    rng = random.Random(_sample_seed(instance_name, sample_idx, base_seed))
    return rng, {
        operation: rng.choice(instance.eligible_machines[operation])
        for operation in instance.real_operations
    }


def _random_precedence_feasible_order(instance, rng):
    """Return a randomized topological order of the job precedence graph."""
    operations = list(instance.real_operations)
    predecessors = {
        operation: set(instance.predecessors.get(operation, []))
        for operation in operations
    }
    successors = {operation: [] for operation in operations}
    for operation, operation_predecessors in predecessors.items():
        for predecessor in operation_predecessors:
            if predecessor in successors:
                successors[predecessor].append(operation)

    available = [
        operation for operation in operations if not predecessors[operation]
    ]
    result = []
    while available:
        operation = available.pop(rng.randrange(len(available)))
        result.append(operation)
        for successor in successors[operation]:
            predecessors[successor].discard(operation)
            if not predecessors[successor] and successor not in result:
                available.append(successor)
    if len(result) != len(operations):
        raise ValueError("The job precedence graph contains a cycle.")
    return result


def _random_direct_predecessor_edges(instance, fixed_y_assignment, rng):
    """Create an acyclic random machine sequence for a full Y assignment."""
    operation_order = _random_precedence_feasible_order(instance, rng)
    operations_by_machine = {
        machine: [] for machine in range(instance.num_machines)
    }
    for operation in operation_order:
        operations_by_machine[fixed_y_assignment[operation]].append(operation)
    return [
        (source, target, machine)
        for machine, machine_operations in operations_by_machine.items()
        for source, target in zip(
            machine_operations,
            machine_operations[1:],
        )
    ]


def _apply_fixed_u_values(
    model,
    variables,
    instance,
    fixed_y_assignment,
    rng,
    sequence_fix_ratio: float,
):
    """Fix a share of a precedence-feasible random direct-predecessor chain."""
    _validate_fix_ratio_range(sequence_fix_ratio, sequence_fix_ratio)
    if variables.get("U") is None:
        return {
            "requested_sequence_fix_ratio": float(sequence_fix_ratio),
            "sequence_fix_ratio": 0.0,
            "fixed_predecessor_edges": 0,
            "candidate_predecessor_edges": 0,
        }

    candidate_edges = [
        edge
        for edge in _random_direct_predecessor_edges(
            instance,
            fixed_y_assignment,
            rng,
        )
        if edge in variables["U"]
    ]
    fixed_count = round(len(candidate_edges) * sequence_fix_ratio)
    if sequence_fix_ratio > 0.0 and candidate_edges:
        fixed_count = max(1, fixed_count)
    fixed_count = min(fixed_count, len(candidate_edges))
    fixed_edges = (
        rng.sample(candidate_edges, fixed_count)
        if fixed_count else []
    )
    for edge in fixed_edges:
        variable = variables["U"][edge]
        variable.lb = 1.0
        variable.ub = 1.0
    model.update()
    return {
        "requested_sequence_fix_ratio": float(sequence_fix_ratio),
        "sequence_fix_ratio": (
            len(fixed_edges) / len(candidate_edges)
            if candidate_edges else 0.0
        ),
        "fixed_predecessor_edges": len(fixed_edges),
        "candidate_predecessor_edges": len(candidate_edges),
    }


def _apply_sampled_reliability_parameters(
    instance,
    instance_name: str,
    sample_idx: int,
    base_seed,
    ranges,
):
    """Use deployment parameters or draw broader Weibull inputs for a sample."""
    rng = random.Random(
        _sample_seed(instance_name, sample_idx, f"{base_seed}:reliability")
    )
    deployment_share = float(ranges.get("deployment_share", 0.0))
    if not 0.0 <= deployment_share <= 1.0:
        raise ValueError(
            "training.data_generation.reliability_ranges.deployment_share "
            "must lie in [0, 1]."
        )
    deployment = ranges.get("deployment", {})
    if not isinstance(deployment, dict):
        raise ValueError(
            "training.data_generation.reliability_ranges.deployment must "
            "be a JSON object."
        )

    def sample(key, default):
        low, high = ranges.get(key, default)
        low, high = float(low), float(high)
        if low > high:
            raise ValueError(
                f"training.data_generation.reliability_ranges.{key}: "
                "lower bound must not exceed upper bound."
            )
        return {
            machine: rng.uniform(low, high)
            for machine in range(instance.num_machines)
        }

    def deployment_values(key, fallback):
        value = float(deployment.get(key, fallback))
        return {
            machine: value
            for machine in range(instance.num_machines)
        }

    if rng.random() < deployment_share:
        instance.machine_initial_age = deployment_values(
            "initial_age", 10.0
        )
        instance.weibull_eta = deployment_values("eta", 80.0)
        instance.repair_duration = deployment_values(
            "repair_duration", 20.0
        )
        sampling_mode = "deployment"
    else:
        instance.machine_initial_age = sample("initial_age", [0.0, 60.0])
        instance.weibull_eta = sample("eta", [50.0, 120.0])
        instance.repair_duration = sample(
            "repair_duration",
            ranges.get("failure_cost", [10.0, 30.0]),
        )
        sampling_mode = "variation"
    instance.reliability_sampling_mode = sampling_mode
    return sampling_mode


def _apply_fixed_y_values(
    model,
    variables,
    instance,
    fixed_y_assignment,
    rng,
    fix_ratio=None,
    fix_ratio_min: float = 0.2,
    fix_ratio_max: float = 0.4,
):
    if fix_ratio is None:
        _validate_fix_ratio_range(fix_ratio_min, fix_ratio_max)
        fix_ratio = rng.uniform(fix_ratio_min, fix_ratio_max)

    Y = variables["Y"]
    candidate_ops = [
        operation
        for operation in fixed_y_assignment
        if len(instance.eligible_machines[operation]) > 1
    ]

    if candidate_ops:
        k = round(len(candidate_ops) * fix_ratio)
        if fix_ratio > 0.0:
            k = max(1, k)
        k = min(k, len(candidate_ops))
        to_fix = set(rng.sample(candidate_ops, k))
    else:
        to_fix = set()

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

    actual_fix_ratio = (len(to_fix) / len(candidate_ops)) if candidate_ops else 0.0
    return {
        "candidate_operations": len(candidate_ops),
        "fixed_operations": len(to_fix),
        "requested_fix_ratio": float(fix_ratio),
        "fix_ratio": actual_fix_ratio,
        "reliability_sampling_mode": getattr(
            instance, "reliability_sampling_mode", ""
        ),
    }

def _ensure_csv_schema(output_path: Path, fieldnames):
    if not output_path.exists() or output_path.stat().st_size == 0:
        return

    with open(output_path, newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        existing_fieldnames = next(reader, None)

    if existing_fieldnames != fieldnames:
        # A CSV from an older run must not block a new experiment. The first
        # row written with the current schema recreates the file; subsequent
        # rows from the same run are appended normally.
        output_path.unlink()


def _append_csv_row(output_path: Path, fieldnames, row, reset: bool = False):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if reset and output_path.exists():
        output_path.unlink()

    _ensure_csv_schema(output_path, fieldnames)
    write_header = not output_path.exists() or output_path.stat().st_size == 0
    with open(output_path, "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if write_header:
            writer.writerow(fieldnames)
        writer.writerow(row)


def _append_fixed_y_result_csv(
    instance,
    instance_name: str,
    sample_idx: int,
    total_failure_delay,
    merged_csv_path: str | None = None,
    reset_merged_csv: bool = False,
    gnn_data: dict | None = None,
    fixed_y_stats: dict | None = None,
    label_metadata: dict | None = None,
):
    # Only include total expected failure delay in the fixed-Y CSV by default.
    # Machine loads, machine-assigned operation counts and job processing
    # paths are intentionally omitted.
    fieldnames = [
        "instance_name",
        "total_failure_delay",
        "sample_idx",
        "requested_fix_ratio",
        "actual_fix_ratio",
        "fixed_operations",
        "candidate_operations",
        "requested_sequence_fix_ratio",
        "actual_sequence_fix_ratio",
        "fixed_predecessor_edges",
        "candidate_predecessor_edges",
        "reliability_sampling_mode",
    ]
    fixed_y_stats = fixed_y_stats or {}
    row = [
        instance_name,
        total_failure_delay,
        sample_idx,
        fixed_y_stats.get("requested_fix_ratio", ""),
        fixed_y_stats.get("fix_ratio", ""),
        fixed_y_stats.get("fixed_operations", ""),
        fixed_y_stats.get("candidate_operations", ""),
        fixed_y_stats.get("requested_sequence_fix_ratio", ""),
        fixed_y_stats.get("sequence_fix_ratio", ""),
        fixed_y_stats.get("fixed_predecessor_edges", ""),
        fixed_y_stats.get("candidate_predecessor_edges", ""),
        fixed_y_stats.get("reliability_sampling_mode", ""),
    ]
    if gnn_data is not None:
        gnn_fieldnames = [
            "gnn_feature_names",
            "gnn_edge_feature_names",
            "gnn_node_features",
            "gnn_active_edge_indices",
            "gnn_active_edge_features",
            "operation_failure_delays",
            "operation_failure_probabilities",
            "operation_repair_durations",
            "reliability_graph_parameters",
        ]
        fieldnames.extend(gnn_fieldnames)
        row.extend(
            [
                gnn_data["feature_names"],
                gnn_data["edge_feature_names"],
                gnn_data["node_features"],
                gnn_data["active_edge_indices"],
                gnn_data["active_edge_features"],
                gnn_data["operation_failure_delays"],
                gnn_data["operation_failure_probabilities"],
                gnn_data["operation_repair_durations"],
                gnn_data["reliability_graph_parameters"],
            ]
        )
    if label_metadata is not None:
        metadata_fieldnames = [
            "labeling_method",
            "solver_runtime_seconds",
            "solver_status",
            "mip_gap",
            "optimization_sample_idx",
            "pool_solution_number",
            "pool_objective",
            "effective_makespan",
            "normalized_nominal_makespan",
            "normalized_failure_delay",
            "normalized_effective_makespan",
            "pareto_rank",
            "pool_selection_category",
        ]
        fieldnames.extend(metadata_fieldnames)
        row.extend([
            label_metadata.get("labeling_method", ""),
            label_metadata.get("solver_runtime_seconds", ""),
            label_metadata.get("solver_status", ""),
            label_metadata.get("mip_gap", ""),
            label_metadata.get("optimization_sample_idx", ""),
            label_metadata.get("pool_solution_number", ""),
            label_metadata.get("pool_objective", ""),
            label_metadata.get("effective_makespan", ""),
            label_metadata.get("normalized_nominal_makespan", ""),
            label_metadata.get("normalized_failure_delay", ""),
            label_metadata.get("normalized_effective_makespan", ""),
            label_metadata.get("pareto_rank", ""),
            label_metadata.get("pool_selection_category", ""),
        ])

    merged_output_path = None
    if merged_csv_path:
        configured_path = Path(merged_csv_path)
        if configured_path.is_absolute():
            merged_output_path = configured_path
        else:
            merged_output_path = ROOT_DIR / configured_path
        _append_csv_row(
            merged_output_path,
            fieldnames,
            row,
            reset=reset_merged_csv and sample_idx == 0,
        )

    return merged_output_path


def _solution_value(variable, solution_number=None):
    return float(variable.X if solution_number is None else variable.Xn)


def _normalized_pool_value(value, values):
    lower = min(values)
    upper = max(values)
    if upper - lower <= 1e-12:
        return 0.0
    return (float(value) - lower) / (upper - lower)


def _pool_structure_tokens(structure_signature):
    selected_machines, predecessor_edges = structure_signature
    return frozenset(
        [("Y", *entry) for entry in selected_machines]
        + [("U", *entry) for entry in predecessor_edges]
    )


def _pool_structure_distance(left, right):
    left_tokens = left["structure_tokens"]
    right_tokens = right["structure_tokens"]
    union = left_tokens | right_tokens
    if not union:
        return 0.0
    return 1.0 - len(left_tokens & right_tokens) / len(union)


def _pareto_ranks(candidates):
    """Return deterministic non-dominated ranks for makespan and delay."""
    remaining = set(range(len(candidates)))
    ranks = [0] * len(candidates)
    rank = 0
    while remaining:
        front = []
        for candidate_idx in sorted(remaining):
            candidate = candidates[candidate_idx]
            dominated = False
            for other_idx in remaining:
                if other_idx == candidate_idx:
                    continue
                other = candidates[other_idx]
                no_worse = (
                    other["pool_objective"]
                    <= candidate["pool_objective"] + 1e-12
                    and other["total_failure_delay"]
                    <= candidate["total_failure_delay"] + 1e-12
                )
                strictly_better = (
                    other["pool_objective"]
                    < candidate["pool_objective"] - 1e-12
                    or other["total_failure_delay"]
                    < candidate["total_failure_delay"] - 1e-12
                )
                if no_worse and strictly_better:
                    dominated = True
                    break
            if not dominated:
                front.append(candidate_idx)
        for candidate_idx in front:
            ranks[candidate_idx] = rank
        remaining.difference_update(front)
        rank += 1
    return ranks


def _select_diverse_pool_candidates(
    candidates,
    output_limit,
    selection_ratios=None,
    quota_offset=0,
):
    """Select budget-free quality, label-range and structural coverage."""
    output_limit = min(int(output_limit), len(candidates))
    if output_limit <= 0:
        return []

    objective_values = [item["pool_objective"] for item in candidates]
    delay_values = [item["total_failure_delay"] for item in candidates]
    effective_values = [
        item["pool_objective"] + item["total_failure_delay"]
        for item in candidates
    ]
    pareto_ranks = _pareto_ranks(candidates)
    for item, effective_makespan, pareto_rank in zip(
        candidates,
        effective_values,
        pareto_ranks,
    ):
        item["effective_makespan"] = effective_makespan
        item["normalized_nominal_makespan"] = _normalized_pool_value(
            item["pool_objective"],
            objective_values,
        )
        item["normalized_failure_delay"] = _normalized_pool_value(
            item["total_failure_delay"],
            delay_values,
        )
        item["normalized_effective_makespan"] = _normalized_pool_value(
            effective_makespan,
            effective_values,
        )
        item["pareto_rank"] = pareto_rank

    category_names = [
        "very_good_effective",
        "good_nominal_makespan",
        "low_failure_delay",
        "high_failure_delay",
        "pareto_tradeoff",
        "structurally_diverse",
        "very_bad_effective",
    ]
    default_ratios = {
        "very_good_effective": 0.30,
        "good_nominal_makespan": 0.15,
        "low_failure_delay": 0.075,
        "high_failure_delay": 0.075,
        "pareto_tradeoff": 0.20,
        "structurally_diverse": 0.10,
        "very_bad_effective": 0.10,
    }
    ratios = default_ratios if selection_ratios is None else selection_ratios
    if not isinstance(ratios, dict):
        raise ValueError("pool_selection_ratios must be a JSON object.")
    unknown_categories = set(ratios) - set(category_names)
    if unknown_categories:
        raise ValueError(
            "Unknown pool selection categories: "
            + ", ".join(sorted(unknown_categories))
        )
    ratios = {
        category: float(ratios.get(category, 0.0))
        for category in category_names
    }
    if any(value < 0.0 for value in ratios.values()):
        raise ValueError("pool_selection_ratios must be non-negative.")
    ratio_sum = sum(ratios.values())
    if ratio_sum <= 0.0:
        raise ValueError("pool_selection_ratios must have a positive sum.")
    quota_offset = int(quota_offset)
    if quota_offset < 0:
        raise ValueError("quota_offset must be non-negative.")

    # Allocate a deterministic weighted round-robin sequence and take the
    # slice belonging to this pool. This distributes fractional per-pool
    # quotas across repeated optimization runs. With four pools of ten, the
    # configured 40-graph ratios are therefore met exactly per instance.
    cumulative_counts = {category: 0 for category in category_names}
    allocation_sequence = []
    for position in range(quota_offset + output_limit):
        category = max(
            category_names,
            key=lambda candidate_category: (
                (position + 1)
                * ratios[candidate_category]
                / ratio_sum
                - cumulative_counts[candidate_category],
                -category_names.index(candidate_category),
            ),
        )
        cumulative_counts[category] += 1
        allocation_sequence.append(category)
    quotas = {
        category: allocation_sequence[
            quota_offset:quota_offset + output_limit
        ].count(category)
        for category in category_names
    }

    selected = []
    selected_solution_numbers = set()

    def select_ranked(category, ranked_candidates):
        for item in ranked_candidates:
            if len([
                selected_item
                for selected_item in selected
                if selected_item["selection_category"] == category
            ]) >= quotas[category]:
                break
            solution_number = item["solution_number"]
            if solution_number in selected_solution_numbers:
                continue
            selected.append({
                **item,
                "selection_category": category,
            })
            selected_solution_numbers.add(solution_number)

    # Select both extremes first so that common overlap between good nominal
    # makespans and low failure delays cannot crowd out intentionally poor
    # examples.
    select_ranked(
        "very_good_effective",
        sorted(
            candidates,
            key=lambda item: (
                item["effective_makespan"],
                item["pool_objective"],
            ),
        ),
    )
    select_ranked(
        "very_bad_effective",
        sorted(
            candidates,
            key=lambda item: (
                -item["effective_makespan"],
                -item["total_failure_delay"],
            ),
        ),
    )
    select_ranked(
        "good_nominal_makespan",
        sorted(
            candidates,
            key=lambda item: (
                item["pool_objective"],
                item["total_failure_delay"],
            ),
        ),
    )
    select_ranked(
        "low_failure_delay",
        sorted(
            candidates,
            key=lambda item: (
                item["total_failure_delay"],
                item["pool_objective"],
            ),
        ),
    )
    select_ranked(
        "high_failure_delay",
        sorted(
            candidates,
            key=lambda item: (
                -item["total_failure_delay"],
                item["pool_objective"],
            ),
        ),
    )
    select_ranked(
        "pareto_tradeoff",
        sorted(
            candidates,
            key=lambda item: (
                item["pareto_rank"],
                max(
                    item["normalized_nominal_makespan"],
                    item["normalized_failure_delay"],
                ),
                item["normalized_effective_makespan"],
            ),
        ),
    )

    # Fill the explicit diversity quota with candidates that are maximally
    # different from all graphs selected so far.
    while len(selected) < output_limit:
        remaining = [
            item for item in candidates
            if item["solution_number"] not in selected_solution_numbers
        ]
        if not remaining:
            break
        if selected:
            next_item = max(
                remaining,
                key=lambda item: (
                    min(
                        _pool_structure_distance(item, chosen)
                        for chosen in selected
                    ),
                    abs(item["normalized_failure_delay"] - 0.5),
                    abs(item["normalized_effective_makespan"] - 0.5),
                ),
            )
        else:
            next_item = min(
                remaining,
                key=lambda item: item["effective_makespan"],
            )
        selected.append({
            **next_item,
            "selection_category": "structurally_diverse",
        })
        selected_solution_numbers.add(next_item["solution_number"])

    return selected


def _selected_machine_from_y(operation, variables, solution_number=None):
    for machine in variables["eligible_machines"][operation]:
        if int(round(_solution_value(
            variables["Y"][operation, machine],
            solution_number,
        ))):
            return machine
    raise ValueError(f"No selected machine found for operation {operation}.")


def _build_gnn_node_features(operations, variables, solution_number=None):
    if variables.get("constraint_type") != "weibull":
        raise ValueError("GNN node features are only defined for Weibull constraints.")

    feature_names = reliability_node_feature_names()
    node_features = np.zeros((len(operations), len(feature_names)), dtype=np.float32)
    for row_idx, operation in enumerate(operations):
        selected_machine = _selected_machine_from_y(
            operation,
            variables,
            solution_number,
        )
        assigned_processing_time = float(
            variables["processing_times"][operation, selected_machine]
        )
        eta = float(variables["weibull_eta"][selected_machine])
        beta = float(variables["weibull_beta"][selected_machine])
        machine_age = _solution_value(
            variables["R"][operation, selected_machine],
            solution_number,
        )
        node_features[row_idx] = [
            assigned_processing_time / eta,
            machine_age / eta,
            beta / 5.0,
        ]

    return node_features, feature_names


def _json_dumps_compact(value):
    return json.dumps(value, separators=(",", ":"))


def _compute_total_failure_delay(variables):
    if not variables.get("pi_fail"):
        return None
    return sum(
        float(variables["repair_durations"][operation, machine])
        * float(variables["pi_fail"][operation, machine].X)
        for operation, machine in variables["Y_index"]
    )


def _build_gnn_csv_payload(
    model,
    variables,
    graph_mode,
    analytical_labels=None,
    solution_number=None,
):
    graph_mode = _validate_gnn_graph_mode(graph_mode)
    operations = list(variables["real_operations"])
    node_features, feature_names = _build_gnn_node_features(
        operations,
        variables,
        solution_number,
    )
    if graph_mode != GNN_GRAPH_FIXED_CANDIDATE:
        raise ValueError(
            "Machine-age graph training supports only fixed_candidate."
        )

    active_edge_indices = []
    active_edge_features = []
    edge_idx = 0
    for source in operations:
        for target in operations:
            if source == target:
                continue
            common_machines = (
                set(variables["eligible_machines"][source])
                & set(variables["eligible_machines"][target])
            )
            for machine in sorted(common_machines):
                gate = variables["U"][source, target, machine]
                if _solution_value(gate, solution_number) > 0.5:
                    active_edge_indices.append(edge_idx)
                    active_edge_features.append(list(transition_edge_values(
                        variables["instance"],
                        source,
                        target,
                        machine,
                        variables["weibull_eta"],
                        variables["machine_processing_max"],
                    )))
                edge_idx += 1

    if analytical_labels is not None:
        operation_failure_delays = [
            float(analytical_labels["delays"][operation])
            for operation in operations
        ]
        operation_failure_probabilities = [
            float(analytical_labels["probabilities"][operation])
            for operation in operations
        ]
    elif variables.get("pi_fail"):
        operation_failure_delays = [
            sum(
                float(variables["repair_durations"][operation, machine])
                * _solution_value(
                    variables["pi_fail"][operation, machine],
                    solution_number,
                )
                for machine in variables["eligible_machines"][operation]
            )
            for operation in operations
        ]
        operation_failure_probabilities = [
            sum(
                _solution_value(
                    variables["pi_fail"][operation, machine],
                    solution_number,
                )
                for machine in variables["eligible_machines"][operation]
            )
            for operation in operations
        ]
    else:
        operation_failure_delays = None
        operation_failure_probabilities = None

    return {
        "feature_names": _json_dumps_compact(feature_names),
        "edge_feature_names": _json_dumps_compact(
            RELIABILITY_EDGE_FEATURE_NAMES
        ),
        "node_features": _json_dumps_compact(node_features.tolist()),
        "active_edge_indices": _json_dumps_compact(active_edge_indices),
        "active_edge_features": _json_dumps_compact(active_edge_features),
        "operation_failure_delays": (
            _json_dumps_compact(operation_failure_delays)
            if operation_failure_delays is not None else ""
        ),
        "operation_failure_probabilities": (
            _json_dumps_compact(operation_failure_probabilities)
            if operation_failure_probabilities is not None else ""
        ),
        "operation_repair_durations": _json_dumps_compact([
            sum(
                float(variables["repair_durations"][operation, machine])
                * _solution_value(
                    variables["Y"][operation, machine],
                    solution_number,
                )
                for machine in variables["eligible_machines"][operation]
            )
            for operation in operations
        ]),
        "reliability_graph_parameters": _json_dumps_compact(
            reliability_graph_config_dict(
                variables["reliability_graph_config"]
            )
        ),
    }


_GUROBI_NONLINEAR_BUILD_KEY_MAP = {
    "add_machine_load_lb": "add_machine_load_lb",
    "type": "constraint_type",
    "constraint_type": "constraint_type",
    "enforce_constraint": "enforce_constraint",
    "weibull_budget_per_operation": "weibull_budget_per_operation",
    "reliability_graph_config": "reliability_graph_config",
}

_GUROBI_GNN_BUILD_KEY_MAP = {
    "add_machine_load_lb": "add_machine_load_lb",
    "add_schedule_upper_bounds": "add_schedule_upper_bounds",
    "model_path": "model_path",
    "metadata_path": "metadata_path",
    "model_seed": "model_seed",
    "convolution": "convolution",
    "aggregation": "aggregation",
    "pooling": "pooling",
    "layers": "layers",
    "hidden_channels": "hidden_channels",
    "bound_tightening": "bound_tightening",
    "analytic_bounds": "analytic_bounds",
    "structured_first_layer": "structured_first_layer",
    "relu_formulation": "relu_formulation",
    "objective_weight": "objective_weight",
    "type": "constraint_type",
    "constraint_type": "constraint_type",
    "enforce_constraint": "enforce_constraint",
    "weibull_budget_per_operation": "weibull_budget_per_operation",
    "reliability_graph_config": "reliability_graph_config",
    "edge_type": "edge_type",
}

def _split_gurobi_nonlinear_kwargs(kwargs):
    gurobi_params = {}
    build_kwargs = {}
    write_solution = True
    write_gnn_inputs = False
    gnn_graph_mode = _default_gnn_graph_mode()
    plot_solution_graph = False
    plot_solution_schedule = False
    plot_candidate_graph = False
    plot_solution_graph_style = "disjunctive"

    for key, value in kwargs.items():
        normalized_key = key.lower()
        if normalized_key == "write_solution":
            write_solution = _as_bool(value)
            continue
        if normalized_key == "write_gnn_inputs":
            write_gnn_inputs = _as_bool(value)
            continue
        if normalized_key in {"gnn_graph_mode", "graph_mode"}:
            gnn_graph_mode = _validate_gnn_graph_mode(value)
            continue
        if normalized_key == "plot_solution_graph":
            plot_solution_graph = _as_bool(value)
            continue
        if normalized_key in {"plot_solution_schedule", "plot_schedule"}:
            plot_solution_schedule = _as_bool(value)
            continue
        if normalized_key == "plot_candidate_graph":
            plot_candidate_graph = _as_bool(value)
            continue
        if normalized_key == "plot_solution_graph_style":
            plot_solution_graph_style = str(value)
            continue
        if normalized_key == "gnn_input_dir":
            continue

        build_key = _GUROBI_NONLINEAR_BUILD_KEY_MAP.get(normalized_key)
        if build_key is None:
            gurobi_params[key] = value
            continue

        if build_key in {"add_machine_load_lb", "enforce_constraint"}:
            value = _as_bool(value)
        elif build_key == "weibull_budget_per_operation":
            value = float(value)

        build_kwargs[build_key] = value

    return (
        gurobi_params,
        build_kwargs,
        write_solution,
        write_gnn_inputs,
        gnn_graph_mode,
        plot_solution_graph,
        plot_solution_schedule,
        plot_candidate_graph,
        plot_solution_graph_style,
    )


def _split_gurobi_gnn_kwargs(kwargs):
    gurobi_params = {}
    build_kwargs = {}
    write_solution = True

    for key, value in kwargs.items():
        normalized_key = key.lower()
        if normalized_key == "write_solution":
            write_solution = _as_bool(value)
            continue

        build_key = _GUROBI_GNN_BUILD_KEY_MAP.get(normalized_key)
        if build_key is None:
            gurobi_params[key] = value
            continue

        if build_key in {
            "add_machine_load_lb",
            "add_schedule_upper_bounds",
            "enforce_constraint",
            "analytic_bounds",
            "structured_first_layer",
        }:
            value = _as_bool(value)
        elif build_key == "model_seed":
            value = int(value)
        elif build_key in {
            "objective_weight", "weibull_budget_per_operation"
        } and value not in (None, ""):
            value = float(value)

        build_kwargs[build_key] = value

    return gurobi_params, build_kwargs, write_solution



def _solve_gurobi_model(
    fjsp_instance,
    instance_name: str,
    create_fixed_y: bool,
    sample_idx: int,
    random_seed,
    save_assignments: bool,
    fix_ratio: float | None,
    fix_ratio_min: float,
    fix_ratio_max: float,
    merged_csv_path: str | None,
    reset_merged_csv: bool,
    solver_kwargs,
):
    model = gp.Model("FJSP Model")

    for key, value in solver_kwargs.items():
        if hasattr(model.Params, key):
            setattr(model.Params, key, value)
            print(f"  Set Gurobi parameter {key} = {value}")
        else:
            print(f"Warning: Unknown keyword argument '{key}' provided. It will be ignored.")

    model, variables = build_fjsp_gurobi(model, fjsp_instance)

    fixed_y_assignment = None
    fixed_y_stats = None
    if create_fixed_y:
        rng, fixed_y_assignment = _generate_random_fixed_y(
            fjsp_instance,
            instance_name=instance_name,
            sample_idx=sample_idx,
            base_seed=random_seed,
        )
        fixed_y_stats = _apply_fixed_y_values(
            model,
            variables,
            fjsp_instance,
            fixed_y_assignment,
            rng,
            fix_ratio=fix_ratio,
            fix_ratio_min=fix_ratio_min,
            fix_ratio_max=fix_ratio_max,
        )
        print(
            "  Using fixed random Y assignment "
            f"(sample {sample_idx + 1}, base seed {random_seed}, "
            f"fixed {fixed_y_stats['fixed_operations']}/{fixed_y_stats['candidate_operations']} "
            f"flexible operations, ratio {fixed_y_stats['fix_ratio']:.2f})"
        )

        if save_assignments:
            print("Warning: save_assignments is ignored to avoid creating additional files.")

    print("\nStarting optimization...")
    model.optimize()

    solution_dir = _solver_solution_dir("gurobi")
    solution_suffix = f"__yfix_{sample_idx:03d}" if create_fixed_y else ""
    solution_path = solution_dir / f"solution_{instance_name}_gurobi{solution_suffix}.txt"

    if model.Status == GRB.OPTIMAL:
        print(f"\nOptimal solution found! Objective value: {model.ObjVal:.2f}")
    elif model.Status == GRB.TIME_LIMIT and model.SolCount > 0:
        print(f"\nTime limit reached. Best solution found: {model.ObjVal:.2f}")
    elif model.Status == GRB.INFEASIBLE:
        print("\nError: Model is infeasible. No solution exists.")
    elif model.Status == GRB.UNBOUNDED:
        print("\nError: Model is unbounded.")
    else:
        print(f"\nOptimization ended with status {model.Status}.")

    if not create_fixed_y:
        write_solution_file_gurobi(model, variables, fjsp_instance, solution_path)
        print(f"Solution written to: {solution_path}")

    if create_fixed_y and fixed_y_assignment is not None:
        if model.SolCount == 0:
            print("No feasible solution found. CSV row skipped.")
            model.dispose()
            return

        print(
            "Fixed-Y CSV skipped for solver 'gurobi' because "
            "total_failure_delay is only available in 'gurobi_nonlinear'."
        )
    model.dispose()



def _solve_gurobi_gnn_model(
    fjsp_instance,
    instance_name: str,
    create_fixed_y: bool,
    sample_idx: int,
    random_seed,
    save_assignments: bool,
    solver_kwargs,
):
    model = gp.Model("FJSP Gurobi GNN Model")
    gurobi_params, build_kwargs, write_solution = _split_gurobi_gnn_kwargs(
        solver_kwargs
    )

    for key, value in gurobi_params.items():
        if hasattr(model.Params, key):
            setattr(model.Params, key, value)
            print(f"  Set Gurobi parameter {key} = {value}")
        else:
            print(f"Warning: Unknown keyword argument '{key}' provided. It will be ignored.")

    if build_kwargs:
        for key, value in build_kwargs.items():
            print(f"  Set GNN model parameter {key} = {value}")

    model, variables = build_fjsp_gurobi_gnn(
        model,
        fjsp_instance,
        **build_kwargs,
    )

    if create_fixed_y:
        print("Warning: create_fixed_y is not supported for solver 'gurobi_gnn'. Ignoring it.")
    if save_assignments:
        print("Warning: save_assignments is ignored for solver 'gurobi_gnn'.")

    print("\nStarting optimization...")
    model.optimize()

    gnn_metadata = variables["gnn_metadata"]
    gnn_layers = int(gnn_metadata.get("num_graphsage_layers", 2))
    gnn_hidden_channels = int(gnn_metadata.get("hidden_channels", 16))
    gnn_seed = int(gnn_metadata.get("seed", 42))
    solution_dir = _solver_solution_dir(
        "gurobi_gnn",
        {
            "graph_mode": variables["gnn_graph_mode"],
            "convolution": gnn_metadata.get(
                "convolution", CONV_SAGE
            ),
            "aggregation": gnn_metadata.get("aggregation", "mean"),
            "pooling": gnn_metadata.get("pooling", "global_add"),
            "layers": gnn_layers,
            "hidden_channels": gnn_hidden_channels,
        },
    )
    solution_path = solution_dir / (
        f"solution_{instance_name}_gurobi_gnn_"
        f"layers{gnn_layers}_hidden{gnn_hidden_channels}_"
        f"relu{variables['gnn_relu_formulation']}_"
        f"seed{gnn_seed}.txt"
    )

    if model.Status == GRB.OPTIMAL:
        print(f"\nOptimal solution found! Objective value: {model.ObjVal:.2f}")
    elif model.Status == GRB.TIME_LIMIT and model.SolCount > 0:
        print(f"\nTime limit reached. Best solution found: {model.ObjVal:.2f}")
    elif model.Status == GRB.INFEASIBLE:
        print("\nError: Model is infeasible. No solution exists.")
    elif model.Status == GRB.UNBOUNDED:
        print("\nError: Model is unbounded.")
    else:
        print(f"\nOptimization ended with status {model.Status}.")

    if model.SolCount > 0:
        predicted_delay = float(
            variables["predicted_total_failure_delay"].X
        )
        print(
            "Predicted total failure delay: "
            f"{predicted_delay:.4f}"
        )
        if variables.get("constraint_enforced"):
            budget = float(variables["constraint_budget"])
            print(
                f"Reliability budget: {budget:.4f} | "
                f"Slack: {budget - predicted_delay:.4f}"
            )

    if write_solution:
        write_solution_file_gurobi_gnn(model, variables, fjsp_instance, solution_path)
        print(f"Solution written to: {solution_path}")
    model.dispose()



def _solve_gurobi_linear_labeled_model(
    fjsp_instance,
    instance_name: str,
    create_fixed_y: bool,
    sample_idx: int,
    random_seed,
    save_assignments: bool,
    fix_ratio: float | None,
    fix_ratio_min: float,
    fix_ratio_max: float,
    sequence_fix_ratio: float,
    pool_output_limit: int,
    pool_selection_ratios,
    merged_csv_path: str | None,
    reset_merged_csv: bool,
    solver_kwargs,
):
    """Optimize a linear schedule and label it analytically afterwards."""
    model = gp.Model("FJSP Gurobi Linear Analytical Label Model")
    (
        gurobi_params,
        build_kwargs,
        write_solution,
        write_gnn_inputs,
        gnn_graph_mode,
        _plot_solution_graph,
        _plot_solution_schedule,
        _plot_candidate_graph,
        _plot_solution_graph_style,
    ) = _split_gurobi_nonlinear_kwargs(solver_kwargs)

    for key, value in gurobi_params.items():
        if hasattr(model.Params, key):
            setattr(model.Params, key, value)
            print(f"  Set Gurobi parameter {key} = {value}")
        else:
            print(
                f"Warning: Unknown keyword argument '{key}' provided. "
                "It will be ignored."
            )
    if not any(str(key).lower() == "seed" for key in gurobi_params):
        model.Params.Seed = int(
            _sample_seed(instance_name, sample_idx, random_seed)
            % 2_000_000_000
        )

    # Legacy callers may still pass enforce_constraint. The two-stage data
    # generator is intentionally budget-free; exact delay is only a label and
    # a component of the effective-makespan sampling score.
    requested_enforcement = _as_bool(
        build_kwargs.pop("enforce_constraint", False)
    )
    if requested_enforcement:
        print(
            "  Ignoring legacy enforce_constraint for budget-free "
            "two-stage data generation."
        )
    model, variables = build_fjsp_gurobi_linear_labeled(
        model,
        fjsp_instance,
        **build_kwargs,
    )

    fixed_y_assignment = None
    fixed_y_stats = None
    if create_fixed_y:
        rng, fixed_y_assignment = _generate_random_fixed_y(
            fjsp_instance,
            instance_name=instance_name,
            sample_idx=sample_idx,
            base_seed=random_seed,
        )
        fixed_y_stats = _apply_fixed_y_values(
            model,
            variables,
            fjsp_instance,
            fixed_y_assignment,
            rng,
            fix_ratio=fix_ratio,
            fix_ratio_min=fix_ratio_min,
            fix_ratio_max=fix_ratio_max,
        )
        fixed_y_stats.update(
            _apply_fixed_u_values(
                model,
                variables,
                fjsp_instance,
                fixed_y_assignment,
                rng,
                sequence_fix_ratio=sequence_fix_ratio,
            )
        )
        print(
            "  Using fixed random Y assignment "
            f"(sample {sample_idx + 1}, base seed {random_seed}, "
            f"fixed {fixed_y_stats['fixed_operations']}/"
            f"{fixed_y_stats['candidate_operations']} flexible operations, "
            f"Y ratio {fixed_y_stats['fix_ratio']:.2f}; "
            f"fixed {fixed_y_stats['fixed_predecessor_edges']}/"
            f"{fixed_y_stats['candidate_predecessor_edges']} sequence edges, "
            f"U ratio {fixed_y_stats['sequence_fix_ratio']:.2f}; "
            "reliability "
            f"{fixed_y_stats['reliability_sampling_mode']})"
        )

    if save_assignments:
        print(
            "Warning: save_assignments is ignored to avoid creating "
            "additional files."
        )
    if write_solution and not create_fixed_y:
        print(
            "Warning: write_solution is not implemented for the two-stage "
            "data generator."
        )

    if int(model.Params.PoolSolutions) > 1:
        for variable in model.getVars():
            variable.setAttr(GRB.Attr.PoolIgnore, 1)
        for variable in variables["Y"].values():
            variable.setAttr(GRB.Attr.PoolIgnore, 0)
        for variable in variables["U"].values():
            variable.setAttr(GRB.Attr.PoolIgnore, 0)
        model.update()

    print("\nStarting linear candidate optimization...")
    model.optimize()
    if model.SolCount == 0:
        print(
            f"No candidate solution found (Gurobi status {model.Status}). "
            "CSV row skipped."
        )
        model.dispose()
        return

    available_pool_solutions = int(model.SolCount)
    seen_structures = set()
    output_structure_key = str(merged_csv_path or "")
    if reset_merged_csv and sample_idx == 0:
        _POOL_STRUCTURES_BY_OUTPUT[output_structure_key] = set()
    output_structures = _POOL_STRUCTURES_BY_OUTPUT.setdefault(
        output_structure_key,
        set(),
    )
    pool_candidates = []
    for solution_number in range(available_pool_solutions):
        model.Params.SolutionNumber = solution_number
        structure_signature = (
            tuple(
                (operation, machine)
                for operation, machine in variables["Y_index"]
                if _solution_value(
                    variables["Y"][operation, machine],
                    solution_number,
                ) > 0.5
            ),
            tuple(
                edge
                for edge in variables["U_index"]
                if _solution_value(
                    variables["U"][edge],
                    solution_number,
                ) > 0.5
            ),
        )
        dataset_structure_signature = (
            instance_name,
            structure_signature,
        )
        if (
            structure_signature in seen_structures
            or dataset_structure_signature in output_structures
        ):
            continue
        seen_structures.add(structure_signature)

        labels = analytical_weibull_labels(
            variables,
            solution_number=solution_number,
        )
        total_failure_delay = float(labels["total_failure_delay"])
        pool_objective = float(model.PoolObjVal)
        pool_candidates.append({
            "solution_number": solution_number,
            "structure_signature": structure_signature,
            "structure_tokens": _pool_structure_tokens(
                structure_signature
            ),
            "labels": labels,
            "total_failure_delay": total_failure_delay,
            "pool_objective": pool_objective,
        })

    if not pool_candidates:
        print("  No new unique structural pool candidate available.")
        model.dispose()
        return

    selected_candidates = _select_diverse_pool_candidates(
        pool_candidates,
        pool_output_limit,
        selection_ratios=pool_selection_ratios,
        quota_offset=sample_idx * int(pool_output_limit),
    )
    effective_values = [
        item["effective_makespan"] for item in pool_candidates
    ]
    print(
        f"  Unique structural candidates: {len(pool_candidates)} | "
        "effective makespan range: "
        f"{min(effective_values):.4f}..{max(effective_values):.4f}"
    )
    selected_category_counts = {
        category: sum(
            item["selection_category"] == category
            for item in selected_candidates
        )
        for category in (
            "very_good_effective",
            "good_nominal_makespan",
            "low_failure_delay",
            "high_failure_delay",
            "pareto_tradeoff",
            "structurally_diverse",
            "very_bad_effective",
        )
    }
    print(
        "  Selected pool mix: "
        + ", ".join(
            f"{category}={count}"
            for category, count in selected_category_counts.items()
        )
    )

    saved_solutions = 0
    for candidate in selected_candidates:
        solution_number = candidate["solution_number"]
        model.Params.SolutionNumber = solution_number
        labels = candidate["labels"]
        total_failure_delay = candidate["total_failure_delay"]
        pool_objective = candidate["pool_objective"]
        effective_makespan = candidate["effective_makespan"]
        selection_category = candidate["selection_category"]
        print(
            f"  Selected [{selection_category}] pool solution "
            f"{solution_number + 1}/{available_pool_solutions} | "
            f"nominal makespan: {pool_objective:.2f} | "
            f"exact Weibull delay: {total_failure_delay:.4f} | "
            f"effective makespan: {effective_makespan:.4f}"
        )

        if not (create_fixed_y and fixed_y_assignment is not None):
            continue
        gnn_data = (
            _build_gnn_csv_payload(
                model,
                variables,
                graph_mode=gnn_graph_mode,
                analytical_labels=labels,
                solution_number=solution_number,
            )
            if write_gnn_inputs
            else None
        )
        mip_gap = ""
        try:
            mip_gap = float(model.MIPGap)
        except (AttributeError, gp.GurobiError):
            pass
        csv_path = _append_fixed_y_result_csv(
            instance=fjsp_instance,
            instance_name=instance_name,
            sample_idx=(
                sample_idx * int(pool_output_limit) + saved_solutions
            ),
            total_failure_delay=round(total_failure_delay, 6),
            merged_csv_path=merged_csv_path,
            reset_merged_csv=reset_merged_csv,
            gnn_data=gnn_data,
            fixed_y_stats=fixed_y_stats,
            label_metadata={
                "labeling_method": (
                    "linear_fix_and_optimize_exact_weibull_posthoc"
                ),
                "solver_runtime_seconds": round(float(model.Runtime), 6),
                "solver_status": int(model.Status),
                "mip_gap": mip_gap,
                "optimization_sample_idx": sample_idx,
                "pool_solution_number": solution_number,
                "pool_objective": round(pool_objective, 6),
                "effective_makespan": round(effective_makespan, 6),
                "normalized_nominal_makespan": round(
                    candidate["normalized_nominal_makespan"], 6
                ),
                "normalized_failure_delay": round(
                    candidate["normalized_failure_delay"], 6
                ),
                "normalized_effective_makespan": round(
                    candidate["normalized_effective_makespan"], 6
                ),
                "pareto_rank": candidate["pareto_rank"],
                "pool_selection_category": selection_category,
            },
        )
        output_structures.add((
            instance_name,
            candidate["structure_signature"],
        ))
        saved_solutions += 1
        if csv_path is not None:
            print(f"Two-stage labeled result appended to: {csv_path}")
    print(
        "  Unique structural pool solutions saved: "
        f"{saved_solutions}/{int(pool_output_limit)} requested "
        f"from {available_pool_solutions} pool candidates"
    )

    model.dispose()


def _solve_gurobi_nonlinear_model(
    fjsp_instance,
    instance_name: str,
    create_fixed_y: bool,
    sample_idx: int,
    random_seed,
    save_assignments: bool,
    fix_ratio: float | None,
    fix_ratio_min: float,
    fix_ratio_max: float,
    sequence_fix_ratio: float,
    merged_csv_path: str | None,
    reset_merged_csv: bool,
    solver_kwargs,
):
    model = gp.Model("FJSP Gurobi Nonlinear Model")
    (
        gurobi_params,
        build_kwargs,
        write_solution,
        write_gnn_inputs,
        gnn_graph_mode,
        plot_solution_graph,
        plot_solution_schedule,
        plot_candidate_graph,
        plot_solution_graph_style,
    ) = _split_gurobi_nonlinear_kwargs(solver_kwargs)

    for key, value in gurobi_params.items():
        if hasattr(model.Params, key):
            setattr(model.Params, key, value)
            print(f"  Set Gurobi parameter {key} = {value}")
        else:
            print(
                f"Warning: Unknown keyword argument '{key}' provided. "
                "It will be ignored."
            )

    if build_kwargs:
        for key, value in build_kwargs.items():
            print(f"  Set nonlinear model parameter {key} = {value}")
    if write_gnn_inputs:
        print(f"  Set GNN graph mode = {gnn_graph_mode}")

    model, variables = build_fjsp_gurobi_nonlinear(
        model,
        fjsp_instance,
        **build_kwargs,
    )

    fixed_y_assignment = None
    fixed_y_stats = None
    if create_fixed_y:
        rng, fixed_y_assignment = _generate_random_fixed_y(
            fjsp_instance,
            instance_name=instance_name,
            sample_idx=sample_idx,
            base_seed=random_seed,
        )
        fixed_y_stats = _apply_fixed_y_values(
            model,
            variables,
            fjsp_instance,
            fixed_y_assignment,
            rng,
            fix_ratio=fix_ratio,
            fix_ratio_min=fix_ratio_min,
            fix_ratio_max=fix_ratio_max,
        )
        fixed_y_stats.update(
            _apply_fixed_u_values(
                model,
                variables,
                fjsp_instance,
                fixed_y_assignment,
                rng,
                sequence_fix_ratio=sequence_fix_ratio,
            )
        )
        print(
            "  Using fixed random Y assignment "
            f"(sample {sample_idx + 1}, base seed {random_seed}, "
            f"fixed {fixed_y_stats['fixed_operations']}/"
            f"{fixed_y_stats['candidate_operations']} flexible operations, "
            f"Y ratio {fixed_y_stats['fix_ratio']:.2f}; "
            f"fixed {fixed_y_stats['fixed_predecessor_edges']}/"
            f"{fixed_y_stats['candidate_predecessor_edges']} sequence edges, "
            f"U ratio {fixed_y_stats['sequence_fix_ratio']:.2f})"
        )

    if save_assignments:
        print(
            "Warning: save_assignments is ignored to avoid creating additional files."
        )

    print("\nStarting optimization...")
    model.optimize()

    solution_dir = _solver_solution_dir("gurobi_nonlinear")
    solution_suffix = f"__yfix_{sample_idx:03d}" if create_fixed_y else ""
    solution_path = (
        solution_dir
        / f"solution_{instance_name}_gurobi_nonlinear{solution_suffix}.txt"
    )

    if model.Status == GRB.OPTIMAL:
        print(f"\nOptimal solution found! Objective value: {model.ObjVal:.2f}")
    elif model.Status == GRB.TIME_LIMIT and model.SolCount > 0:
        print(f"\nTime limit reached. Best solution found: {model.ObjVal:.2f}")
    elif model.Status == GRB.INFEASIBLE:
        print("\nError: Model is infeasible. No solution exists.")
    elif model.Status == GRB.UNBOUNDED:
        print("\nError: Model is unbounded.")
    else:
        print(f"\nOptimization ended with status {model.Status}.")

    if model.SolCount > 0:
        makespan = float(variables["C_max"].X)
        total_failure_delay = _compute_total_failure_delay(variables)
        print(f"Makespan: {makespan:.2f}")
        if total_failure_delay is not None:
            print(f"Expected failure delay: {total_failure_delay:.2f}")
            if variables.get("constraint_enforced"):
                budget = float(variables["constraint_budget"])
                print(
                    f"Reliability budget: {budget:.2f} | "
                    f"Slack: {budget - total_failure_delay:.2f}"
                )

    if write_solution and not create_fixed_y:
        write_solution_file_gurobi_nonlinear(
            model,
            variables,
            fjsp_instance,
            solution_path,
        )
        print(f"Solution written to: {solution_path}")

    if (
        plot_solution_graph
        or plot_solution_schedule
        or plot_candidate_graph
    ) and not create_fixed_y:
        graph_dir = ROOT_DIR / "plots" / "fjsp_solution_plots"
        graph_dir.mkdir(parents=True, exist_ok=True)

        if plot_solution_schedule and model.Status == GRB.OPTIMAL:
            schedule_path = (
                graph_dir / f"schedule_{instance_name}_gurobi_nonlinear.png"
            )
            try:
                plot_solution_schedule_gurobi_nonlinear(
                    variables,
                    fjsp_instance,
                    schedule_path,
                )
                print(f"Schedule plot written to: {schedule_path}")
            except Exception as exc:
                print(f"Warning: schedule plot could not be created: {exc}")
        elif plot_solution_schedule and model.SolCount > 0:
            print("Schedule plot skipped because the solution is not proven optimal.")

        if plot_solution_graph and model.Status == GRB.OPTIMAL:
            graph_path = graph_dir / f"graph_{instance_name}_gurobi_nonlinear.png"
            try:
                normalized_plot_style = plot_solution_graph_style.strip().lower()
                graph_title = (
                    f"{instance_name} optimal disjunctive solution graph"
                    if normalized_plot_style in {"disjunctive", "disjunctive_solution"}
                    else f"{instance_name} optimal machine-operation graph"
                )
                plot_solution_graph_gurobi_nonlinear(
                    model,
                    variables,
                    fjsp_instance,
                    graph_path,
                    title=graph_title,
                    style=plot_solution_graph_style,
                )
                print(f"Graph plot written to: {graph_path}")
            except Exception as exc:
                print(f"Warning: graph plot could not be created: {exc}")
        elif plot_solution_graph and model.SolCount > 0:
            print("Graph plot skipped because the solution is not proven optimal.")

        # A requested solution graph always includes both views:
        # 1. the candidate graph after the optimized Y assignment and
        # 2. the optimal graph above with only the selected machine-order arcs.
        # ``plot_candidate_graph`` remains available for requesting only the
        # candidate view without the optimal solution graph.
        create_candidate_graph = plot_solution_graph or plot_candidate_graph
        if create_candidate_graph and model.SolCount > 0:
            candidate_graph_path = (
                graph_dir / f"graph_{instance_name}_gurobi_nonlinear_candidates.png"
            )
            try:
                plot_candidate_graph_gurobi_nonlinear(
                    variables,
                    fjsp_instance,
                    candidate_graph_path,
                    title=f"{instance_name} possible machine-order graph after assignment",
                )
                print(f"Candidate graph plot written to: {candidate_graph_path}")
            except Exception as exc:
                print(f"Warning: candidate graph plot could not be created: {exc}")
        elif create_candidate_graph:
            print("Candidate graph plot skipped because no solution is available.")

    if create_fixed_y and fixed_y_assignment is not None:
        if model.SolCount == 0:
            print("No feasible solution found. CSV row skipped.")
            model.dispose()
            return

        gnn_data = (
            _build_gnn_csv_payload(model, variables, graph_mode=gnn_graph_mode)
            if write_gnn_inputs
            else None
        )
        total_failure_delay = _compute_total_failure_delay(variables)
        csv_path = _append_fixed_y_result_csv(
            instance=fjsp_instance,
            instance_name=instance_name,
            sample_idx=sample_idx,
            total_failure_delay=(
                round(total_failure_delay, 6)
                if total_failure_delay is not None else ""
            ),
            merged_csv_path=merged_csv_path,
            reset_merged_csv=reset_merged_csv,
            gnn_data=gnn_data,
            fixed_y_stats=fixed_y_stats,
        )
        if csv_path is not None:
            print(f"Merged Fixed-Y result appended to: {csv_path}")

    model.dispose()





def solveModel(**kwargs):
    """Load a pickled FJSP instance, solve it, and write a solution file."""
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
    raw_fix_ratio = kwargs.pop("fix_ratio", None)
    fix_ratio = (
        None if raw_fix_ratio in (None, "") else float(raw_fix_ratio)
    )
    if fix_ratio is not None:
        _validate_fix_ratio_range(fix_ratio, fix_ratio)
    fix_ratio_min = float(kwargs.pop("fix_ratio_min", 0.2))
    fix_ratio_max = float(kwargs.pop("fix_ratio_max", 0.4))
    sequence_fix_ratio = float(kwargs.pop("sequence_fix_ratio", 0.0))
    _validate_fix_ratio_range(sequence_fix_ratio, sequence_fix_ratio)
    pool_output_limit = int(kwargs.pop("pool_output_limit", 1))
    if pool_output_limit <= 0:
        raise ValueError("pool_output_limit muss groesser als 0 sein.")
    pool_selection_ratios = kwargs.pop("pool_selection_ratios", None)
    merged_csv_path = kwargs.pop("merged_csv_path", None)
    reset_merged_csv = _as_bool(kwargs.pop("reset_merged_csv", False))
    reliability_ranges = kwargs.pop("reliability_ranges", None)
    instance_directory = kwargs.pop("instance_directory", None)
    try:
        fjsp_instance: FJSPData = load_generated_instance(
            instance_name,
            instance_directory=instance_directory,
        )
    except FileNotFoundError:
        print(f"Error: Instance '{instance_name}' not found")
        print("Available instances:")
        names = generated_instance_names(instance_directory)
        legacy_dir = ROOT_DIR / "02_data" / "fjsp_instances"
        if legacy_dir.exists():
            names.extend(file.stem for file in legacy_dir.glob("*.fjsp"))
        for name in sorted(set(names)):
            print(f"  - {name}")
        return

    if create_fixed_y and reliability_ranges:
        _apply_sampled_reliability_parameters(
            fjsp_instance,
            instance_name=instance_name,
            sample_idx=sample_idx,
            base_seed=random_seed,
            ranges=reliability_ranges,
        )

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
            fix_ratio=fix_ratio,
            fix_ratio_min=fix_ratio_min,
            fix_ratio_max=fix_ratio_max,
            merged_csv_path=merged_csv_path,
            reset_merged_csv=reset_merged_csv,
            solver_kwargs=kwargs,
        )

    elif solver == "gurobi_gnn":
        _solve_gurobi_gnn_model(
            fjsp_instance,
            instance_name=instance_name,
            create_fixed_y=create_fixed_y,
            sample_idx=sample_idx,
            random_seed=random_seed,
            save_assignments=save_assignments,
            solver_kwargs=kwargs,
        )
    elif solver == "gurobi_nonlinear":
        _solve_gurobi_nonlinear_model(
            fjsp_instance,
            instance_name=instance_name,
            create_fixed_y=create_fixed_y,
            sample_idx=sample_idx,
            random_seed=random_seed,
            save_assignments=save_assignments,
            fix_ratio=fix_ratio,
            fix_ratio_min=fix_ratio_min,
            fix_ratio_max=fix_ratio_max,
            sequence_fix_ratio=sequence_fix_ratio,
            merged_csv_path=merged_csv_path,
            reset_merged_csv=reset_merged_csv,
            solver_kwargs=kwargs,
        )
    elif solver == "gurobi_linear_labeled":
        _solve_gurobi_linear_labeled_model(
            fjsp_instance,
            instance_name=instance_name,
            create_fixed_y=create_fixed_y,
            sample_idx=sample_idx,
            random_seed=random_seed,
            save_assignments=save_assignments,
            fix_ratio=fix_ratio,
            fix_ratio_min=fix_ratio_min,
            fix_ratio_max=fix_ratio_max,
            sequence_fix_ratio=sequence_fix_ratio,
            pool_output_limit=pool_output_limit,
            pool_selection_ratios=pool_selection_ratios,
            merged_csv_path=merged_csv_path,
            reset_merged_csv=reset_merged_csv,
            solver_kwargs=kwargs,
        )

    else:
        print(
            f"Error: Unknown solver '{solver}'. "
            "Use 'gurobi', 'gurobi_ml', 'gurobi_gnn', "
            "'gurobi_nonlinear' or 'gurobi_linear_labeled'."
        )

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
    raw_fix_ratios = solver_kwargs.pop("fix_ratios", None)
    fix_ratios = None
    if raw_fix_ratios not in (None, ""):
        if not isinstance(raw_fix_ratios, (list, tuple)):
            raise ValueError("fix_ratios muss eine Liste von Anteilen sein.")
        fix_ratios = [float(value) for value in raw_fix_ratios]
        if not fix_ratios:
            raise ValueError("fix_ratios darf nicht leer sein.")
        for value in fix_ratios:
            _validate_fix_ratio_range(value, value)
    raw_sequence_fix_ratios = solver_kwargs.pop(
        "sequence_fix_ratios", None
    )
    sequence_fix_ratios = [0.0]
    if raw_sequence_fix_ratios not in (None, ""):
        if not isinstance(raw_sequence_fix_ratios, (list, tuple)):
            raise ValueError(
                "sequence_fix_ratios muss eine Liste von Anteilen sein."
            )
        sequence_fix_ratios = [
            float(value) for value in raw_sequence_fix_ratios
        ]
        if not sequence_fix_ratios:
            raise ValueError("sequence_fix_ratios darf nicht leer sein.")
        for value in sequence_fix_ratios:
            _validate_fix_ratio_range(value, value)

    if instance_name is None:
        operations_per_job = CONFIG["instances"]["generation"]["operations_per_job"]
        instance_name = (
            f"i{num_jobs}_k{num_machines}_"
            f"o{min(operations_per_job)}-{max(operations_per_job)}_{instance_nb}"
        )

    if solver in {
        "gurobi",
        "gurobi_gnn",
        "gurobi_nonlinear",
        "gurobi_linear_labeled",
    }:
        if create_fixed_y:
            for sample_idx in range(amount_of_samples_per_instance):
                sample_kwargs = dict(solver_kwargs)
                if fix_ratios is not None:
                    sample_kwargs["fix_ratio"] = fix_ratios[
                        sample_idx % len(fix_ratios)
                    ]
                sample_kwargs["sequence_fix_ratio"] = sequence_fix_ratios[
                    sample_idx % len(sequence_fix_ratios)
                ]
                solveModel(
                    instance_name=instance_name,
                    solver=solver,
                    create_fixed_y=True,
                    sample_idx=sample_idx,
                    random_seed=random_seed,
                    save_assignments=save_assignments,
                    **sample_kwargs,
                )
        else:
            solveModel(instance_name=instance_name, solver=solver, **solver_kwargs)


    else:
        print(
            f"Error: Unknown solver '{solver}'. "
            "Use 'gurobi', 'gurobi_gnn', "
            "'gurobi_nonlinear' or 'gurobi_linear_labeled'."

        )
