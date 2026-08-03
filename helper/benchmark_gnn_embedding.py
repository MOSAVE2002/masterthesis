"""Small reproducible benchmark for compact embedded reliability GNNs."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import re
import sys
import time
from types import SimpleNamespace
from pathlib import Path

import gurobipy as gp
from gurobipy import GRB
import torch


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))
MODEL_ROOT = ROOT_DIR / "04_GraphNeuralNetworks" / "trained_gnn_models"

load_generated_instance = importlib.import_module(
    "01_generator.instance_generator"
).load_generated_instance
build_gnn = importlib.import_module(
    "03_Gurobi.build_fjsp_with_gnn"
).build_fjsp
build_nonlinear = importlib.import_module(
    "03_Gurobi.build_fjsp_with_nonlinear"
).build_fjsp
build_linear = importlib.import_module("03_Gurobi.build_fjsp").build_fjsp
gnn_builder_module = importlib.import_module("03_Gurobi.build_fjsp_with_gnn")
FJSPGraphSAGE = importlib.import_module(
    "04_GraphNeuralNetworks.models.model_training_FJSP_GNN"
).FJSPGraphSAGE
transition_edge_values = importlib.import_module(
    "helper.sequence_setup"
).transition_edge_values
_exact_fixed_schedule_makespan = importlib.import_module(
    "helper.gurobi_solution_writer"
)._exact_fixed_schedule_makespan


def _model_files(directory, stem):
    return {
        "model_path": str(MODEL_ROOT / directory / f"{stem}.pt"),
        "metadata_path": str(MODEL_ROOT / directory / f"{stem}_meta.json"),
    }


VARIANTS = {
    "nonlinear": {"kind": "nonlinear"},
    "sage_h8_loose": {
        "kind": "gnn",
        "hidden_channels": 8,
        "bound_tightening": False,
        **_model_files(
            "sage_sum_global_add/hidden8_layers1",
            "fjsp_gnn_fixed_candidate_sage_sum_global_add_layers1_hidden8_"
            "total_failure_delay_seed42",
        ),
    },
    "sage_h8_tight": {
        "kind": "gnn",
        "hidden_channels": 8,
        "bound_tightening": True,
        "relu_formulation": "big_m",
        **_model_files(
            "sage_sum_global_add/hidden8_layers1",
            "fjsp_gnn_fixed_candidate_sage_sum_global_add_layers1_hidden8_"
            "total_failure_delay_seed42",
        ),
    },
    "sage_h8_tight_sos1": {
        "kind": "gnn",
        "hidden_channels": 8,
        "bound_tightening": True,
        "relu_formulation": "sos1",
        **_model_files(
            "sage_sum_global_add/hidden8_layers1",
            "fjsp_gnn_fixed_candidate_sage_sum_global_add_layers1_hidden8_"
            "total_failure_delay_seed42",
        ),
    },
    "sage_h6_tight": {
        "kind": "gnn",
        "hidden_channels": 6,
        "bound_tightening": True,
        **_model_files(
            "sage_sum_global_add/hidden6_layers1",
            "fjsp_gnn_fixed_candidate_sage_sum_global_add_layers1_hidden6_"
            "total_failure_delay_seed42",
        ),
    },
    "sage_h4_tight": {
        "kind": "gnn",
        "hidden_channels": 4,
        "bound_tightening": True,
        **_model_files(
            "sage_sum_global_add/hidden4_layers1",
            "fjsp_gnn_fixed_candidate_sage_sum_global_add_layers1_hidden4_"
            "total_failure_delay_seed42",
        ),
    },
    "solveraware_h4_tight": {
        "kind": "gnn",
        "hidden_channels": 4,
        "bound_tightening": True,
        **_model_files(
            "solveraware_sage_sum_global_add/hidden4_layers1",
            "fjsp_gnn_solveraware_fixed_candidate_sage_sum_global_add_"
            "layers1_hidden4_total_failure_delay_seed42",
        ),
    },
}

JOB_EDGE_PILOT_ROOT = ROOT_DIR / "04_GraphNeuralNetworks" / "models" / "job_edge_pilot"
for suffix in ("machine_only", "job_only", "machine_plus_job"):
    stem = f"sage_h4_{suffix}"
    VARIANTS[f"pilot_{stem}"] = {
        "kind": "gnn",
        "convolution": "sage",
        "aggregation": "sum",
        "hidden_channels": 4,
        "bound_tightening": True,
        "relu_formulation": "big_m",
        "model_path": str(JOB_EDGE_PILOT_ROOT / f"{stem}.pt"),
        "metadata_path": str(JOB_EDGE_PILOT_ROOT / f"{stem}_meta.json"),
        "edge_type": "both" if suffix == "machine_plus_job" else suffix,
    }

for convolution, aggregation in (
    ("linear", "none"),
    ("gcn", "sum"),
    ("sage", "sum"),
    ("mpnn", "sum"),
):
    directory = (
        f"{convolution}_{aggregation}_global_add/hidden4_layers1"
    )
    stem = (
        f"fjsp_gnn_fixed_candidate_{convolution}_{aggregation}_global_add_"
        "layers1_hidden4_total_failure_delay_seed42"
    )
    for relu_formulation in ("big_m", "sos1"):
        VARIANTS[f"{convolution}_h4_{relu_formulation}"] = {
            "kind": "gnn",
            "convolution": convolution,
            "aggregation": aggregation,
            "hidden_channels": 4,
            "bound_tightening": True,
            "relu_formulation": relu_formulation,
            **_model_files(directory, stem),
        }


def _status_name(status):
    return {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
    }.get(status, str(status))


RELIABILITY_CONFIG = {
    "enabled": True,
    "beta": 3.0,
    "transition_gamma": 0.02,
}


def _set_solver_parameters(model, time_limit, extra_parameters=None):
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = float(time_limit)
    model.Params.MIPGap = 0.0
    model.Params.Threads = 1
    model.Params.Seed = 42
    for name, value in (extra_parameters or {}).items():
        model.setParam(name, value)


def _metadata(specification):
    return json.loads(Path(specification["metadata_path"]).read_text())


def _build_model(model, instance, specification, obbt_bounds=None):
    if specification["kind"] == "nonlinear":
        return build_nonlinear(
            model,
            instance,
            reliability_graph_config=RELIABILITY_CONFIG,
        )
    metadata = _metadata(specification)
    return build_gnn(
        model,
        instance,
        convolution=specification.get(
            "convolution", metadata.get("convolution", "sage")
        ),
        aggregation=specification.get(
            "aggregation", metadata.get("aggregation", "sum")
        ),
        pooling="global_add",
        layers=int(metadata.get("num_graphsage_layers", 1)),
        hidden_channels=specification["hidden_channels"],
        model_path=specification["model_path"],
        metadata_path=specification["metadata_path"],
        bound_tightening=specification["bound_tightening"],
        analytic_bounds=specification.get("analytic_bounds"),
        structured_first_layer=specification.get(
            "structured_first_layer", True
        ),
        relu_formulation=specification.get("relu_formulation", "big_m"),
        reliability_graph_config=RELIABILITY_CONFIG,
        edge_type=specification.get("edge_type"),
        obbt_bounds=obbt_bounds,
    )


def _relu_layer(name):
    if name.startswith("conv1"):
        return "conv1"
    if name.startswith("conv2"):
        return "conv2"
    if name.startswith("conv3"):
        return "conv3"
    if "probability_unclipped" in name:
        return "output"
    return "other"


def _relu_records(model):
    records = []
    for variable in model.getVars():
        if not variable.VarName.endswith("_pre"):
            continue
        name = variable.VarName[:-4]
        lower, upper = float(variable.LB), float(variable.UB)
        state = (
            "inactive" if upper <= 0.0
            else "active" if lower >= 0.0
            else "unstable"
        )
        records.append({
            "name": name,
            "variable_name": variable.VarName,
            "lower": lower,
            "upper": upper,
            "width": upper - lower,
            "layer": _relu_layer(name),
            "state": state,
        })
    return records


def _relu_summary(records):
    result = {
        "total": len(records),
        "active": sum(item["state"] == "active" for item in records),
        "inactive": sum(item["state"] == "inactive" for item in records),
        "unstable": sum(item["state"] == "unstable" for item in records),
        "by_layer": {},
    }
    for layer in sorted({item["layer"] for item in records}):
        selected = [item for item in records if item["layer"] == layer]
        result["by_layer"][layer] = {
            "total": len(selected),
            "active": sum(item["state"] == "active" for item in selected),
            "inactive": sum(item["state"] == "inactive" for item in selected),
            "unstable": sum(item["state"] == "unstable" for item in selected),
        }
    return result


def _compute_obbt_bounds(instance, specification, mode, lp_time_limit):
    started = time.perf_counter()
    probe = gp.Model("gnn_obbt_probe")
    _set_solver_parameters(probe, max(float(lp_time_limit), 1e-3))
    # A Big-M probe gives a purely linear relaxation even when the final
    # encoding uses SOS1. Every integer-feasible schedule is contained in it.
    probe_specification = {
        **specification,
        "relu_formulation": "big_m",
    }
    build_started = time.perf_counter()
    probe, _probe_variables = _build_model(
        probe, instance, probe_specification
    )
    probe.update()
    probe_build_seconds = time.perf_counter() - build_started
    initial_records = _relu_records(probe)
    unstable = [item for item in initial_records if item["state"] == "unstable"]
    if mode == "first_layer":
        selected = [item for item in unstable if item["layer"] == "conv1"]
    elif mode == "all":
        selected = unstable
    else:
        raise ValueError("obbt_mode must be none, first_layer, or all.")

    relaxed = probe.relax()
    relaxed.Params.OutputFlag = 0
    relaxed.Params.Threads = 1
    relaxed.Params.Method = 1
    relaxed.Params.TimeLimit = max(float(lp_time_limit), 1e-3)
    bounds = {}
    successful_lp_solves = 0
    lp_started = time.perf_counter()
    for item in selected:
        target = relaxed.getVarByName(item["variable_name"])
        new_lower, new_upper = item["lower"], item["upper"]
        relaxed.setObjective(target, GRB.MINIMIZE)
        relaxed.optimize()
        if relaxed.Status == GRB.OPTIMAL:
            new_lower = max(new_lower, float(relaxed.ObjVal))
            successful_lp_solves += 1
        relaxed.setObjective(target, GRB.MAXIMIZE)
        relaxed.optimize()
        if relaxed.Status == GRB.OPTIMAL:
            new_upper = min(new_upper, float(relaxed.ObjVal))
            successful_lp_solves += 1
        # Outward padding protects against accepting a bound solely because of
        # LP feasibility/optimality tolerances.
        padding = 1e-7 * max(1.0, abs(new_lower), abs(new_upper))
        new_lower = max(item["lower"], new_lower - padding)
        new_upper = min(item["upper"], new_upper + padding)
        if new_lower <= new_upper:
            bounds[item["name"]] = (new_lower, new_upper)
    lp_seconds = time.perf_counter() - lp_started
    relaxed.dispose()
    probe.dispose()
    stable_after = sum(
        upper <= 0.0 or lower >= 0.0 for lower, upper in bounds.values()
    )
    return bounds, {
        "obbt_mode": mode,
        "obbt_seconds": time.perf_counter() - started,
        "obbt_probe_build_seconds": probe_build_seconds,
        "obbt_lp_seconds": lp_seconds,
        "obbt_candidates": len(unstable),
        "obbt_selected": len(selected),
        "obbt_lp_solves": 2 * len(selected),
        "obbt_successful_lp_solves": successful_lp_solves,
        "obbt_selected_stable_after": stable_after,
        "relu_before_obbt": _relu_summary(initial_records),
    }


def _apply_nominal_warm_start(model, variables, instance, time_limit):
    started = time.perf_counter()
    nominal = gp.Model("nominal_warm_start")
    _set_solver_parameters(nominal, time_limit)
    nominal, nominal_variables = build_linear(nominal, instance)
    nominal.optimize()
    if nominal.SolCount == 0:
        status = _status_name(nominal.Status)
        nominal.dispose()
        return {"warm_start_seconds": time.perf_counter() - started,
                "warm_start_status": status, "warm_start_values": 0}

    starts = 0
    for key in variables["Y_index"]:
        variables["Y"][key].Start = round(nominal_variables["Y"][key].X)
        starts += 1
    for key in variables["X_index"]:
        variables["X"][key].Start = round(nominal_variables["X"][key].X)
        starts += 1

    if variables.get("U") is not None:
        selected_machine = {
            operation: max(
                instance.eligible_machines[operation],
                key=lambda machine: nominal_variables["Y"][operation, machine].X,
            )
            for operation in instance.real_operations
        }
        chains = {machine: [] for machine in variables["machines"]}
        for operation, machine in selected_machine.items():
            processing = float(instance.processing_times[operation, machine])
            start = float(nominal_variables["C"][operation].X) - processing
            chains[machine].append((start, operation))
        successor_edges = set()
        for machine, chain in chains.items():
            ordered = [operation for _start, operation in sorted(chain)]
            successor_edges.update(
                (source, target, machine)
                for source, target in zip(ordered, ordered[1:])
            )
            if variables.get("machine_used") is not None:
                variables["machine_used"][machine].Start = int(bool(ordered))
                starts += 1
            for operation in ordered:
                variables["machine_first"][operation, machine].Start = int(
                    operation == ordered[0]
                )
                variables["machine_last"][operation, machine].Start = int(
                    operation == ordered[-1]
                )
                starts += 2
        for key in variables["U_index"]:
            variables["U"][key].Start = int(key in successor_edges)
            starts += 1
        for operation, machine in variables["Y_index"]:
            if selected_machine[operation] != machine:
                variables["machine_first"][operation, machine].Start = 0
                variables["machine_last"][operation, machine].Start = 0
        if variables.get("A_plus") is not None:
            for i, j, machine in variables["X_index"]:
                yi = int(selected_machine[i] == machine)
                yj = int(selected_machine[j] == machine)
                x = round(nominal_variables["X"][i, j, machine].X)
                variables["A_plus"][i, j, machine].Start = yi * yj * x
                variables["A_minus"][i, j, machine].Start = yi * yj * (1 - x)
                starts += 2
    status = _status_name(nominal.Status)
    nominal.dispose()
    return {
        "warm_start_seconds": time.perf_counter() - started,
        "warm_start_status": status,
        "warm_start_values": starts,
    }


def _structure(model, relu_records):
    variables = model.getVars()
    binaries = [variable for variable in variables if variable.VType == GRB.BINARY]
    return {
        "variables": len(variables),
        "binary_variables": len(binaries),
        "continuous_variables": sum(
            variable.VType == GRB.CONTINUOUS for variable in variables
        ),
        "linear_constraints": int(model.NumConstrs),
        "quadratic_constraints": int(model.NumQConstrs),
        "general_constraints": int(model.NumGenConstrs),
        "constraints": model.NumConstrs + model.NumQConstrs + model.NumGenConstrs,
        "sos_constraints": int(model.NumSOS),
        "nonzeros": int(model.NumNZs),
        "relu_phase_binaries": sum(
            variable.VarName.endswith("_relu_phase") for variable in binaries
        ),
        "clip_binaries": sum(
            variable.VarName.endswith("_clip_above_cap") for variable in binaries
        ),
        "relu": _relu_summary(relu_records),
    }


def _forward_error(variables, instance, specification):
    metadata = _metadata(specification)
    node_features = torch.tensor(
        [
            [float(value.getValue()) for value in row]
            for row in variables["gnn_node_features"]
        ],
        dtype=torch.float32,
    )
    operations = list(variables["real_operations"])
    active_edges, edge_attributes = [], []
    include_job = bool(metadata.get("include_job_precedence_edges", False))
    for edge in variables["gnn_edges"]:
        gate = variables["gnn_edge_activations"][edge]
        gate_value = float(gate.X) if hasattr(gate, "X") else float(gate)
        if gate_value < 0.5:
            continue
        source_idx, target_idx, machine = edge
        if machine == "job":
            attributes = [0.0, 0.0, 1.0] if bool(
                metadata.get("include_machine_predecessor_edges", True)
            ) else [1.0]
        else:
            attributes = list(transition_edge_values(
                instance,
                operations[source_idx],
                operations[target_idx],
                machine,
                variables["weibull_eta"],
                variables["machine_processing_max"],
            ))
            if include_job:
                attributes.append(0.0)
        active_edges.append((source_idx, target_idx))
        edge_attributes.append(attributes)
    edge_index = torch.tensor(active_edges, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attributes, dtype=torch.float32)
    repair_duration = torch.tensor([
        variables["repair_durations"][
            operation,
            max(
                instance.eligible_machines[operation],
                key=lambda machine: variables["Y"][operation, machine].X,
            ),
        ]
        for operation in operations
    ], dtype=torch.float32)
    graph = SimpleNamespace(
        x=node_features,
        edge_index=edge_index,
        edge_attr=edge_attr,
        batch=torch.zeros(len(operations), dtype=torch.long),
        repair_duration=repair_duration,
    )
    network = FJSPGraphSAGE(
        input_size=int(metadata["input_size"]),
        hidden_channels=int(metadata["hidden_channels"]),
        aggregation=metadata["aggregation"],
        output_mode="node_sum",
        num_graphsage_layers=int(metadata["num_graphsage_layers"]),
        convolution=metadata["convolution"],
        pooling=metadata["pooling"],
        edge_feature_size=len(metadata["edge_feature_names"]),
    )
    network.load_state_dict(torch.load(
        specification["model_path"], map_location="cpu", weights_only=True
    ))
    network.eval()
    with torch.no_grad():
        torch_total, torch_nodes = network(graph, return_nodes=True)
    milp_nodes = [
        float(value.X)
        for value in variables["predicted_operation_failure_probabilities"]
    ]
    node_error = max(
        abs(milp - pytorch)
        for milp, pytorch in zip(milp_nodes, torch_nodes.tolist())
    )
    total_error = abs(
        float(variables["predicted_total_failure_delay"].X)
        - float(torch_total.item())
    )
    return node_error, total_error


def _telemetry_callback(storage):
    def callback(model, where):
        if where == GRB.Callback.MIP:
            node_count = float(model.cbGet(GRB.Callback.MIP_NODCNT))
            if node_count < 0.5:
                bound = float(model.cbGet(GRB.Callback.MIP_OBJBND))
                if math.isfinite(bound):
                    storage["root_node_bound_with_cuts"] = bound
        elif where == GRB.Callback.MIPSOL:
            runtime = float(model.cbGet(GRB.Callback.RUNTIME))
            objective = float(model.cbGet(GRB.Callback.MIPSOL_OBJ))
            if storage["time_to_first_solution"] is None:
                storage["time_to_first_solution"] = runtime
            if objective < storage["best_callback_objective"] - 1e-9:
                storage["best_callback_objective"] = objective
                storage["time_to_best_solution"] = runtime
    return callback


def _plain_root_lp_bound(model):
    started = time.perf_counter()
    relaxation = model.relax()
    relaxation.Params.OutputFlag = 0
    relaxation.Params.Threads = 1
    relaxation.optimize()
    bound = (
        float(relaxation.ObjVal)
        if relaxation.Status == GRB.OPTIMAL else None
    )
    relaxation.dispose()
    return bound, time.perf_counter() - started


def _run(
    instance_name,
    variant_name,
    time_limit,
    obbt_mode="none",
    obbt_lp_time_limit=10.0,
    warm_start=False,
    warm_start_time_limit=0.25,
    solver_parameters=None,
):
    instance = load_generated_instance(instance_name)
    specification = VARIANTS[variant_name]
    if specification["kind"] != "gnn" and obbt_mode != "none":
        raise ValueError("OBBT is available only for GNN variants.")

    obbt_bounds = {}
    obbt_statistics = {
        "obbt_mode": "none",
        "obbt_seconds": 0.0,
        "obbt_probe_build_seconds": 0.0,
        "obbt_lp_seconds": 0.0,
        "obbt_candidates": 0,
        "obbt_selected": 0,
        "obbt_lp_solves": 0,
        "obbt_successful_lp_solves": 0,
        "obbt_selected_stable_after": 0,
        "relu_before_obbt": None,
    }
    if obbt_mode != "none":
        obbt_bounds, obbt_statistics = _compute_obbt_bounds(
            instance, specification, obbt_mode, obbt_lp_time_limit
        )

    model = gp.Model(f"benchmark_{instance_name}_{variant_name}_{obbt_mode}")
    _set_solver_parameters(model, time_limit, solver_parameters)
    build_started = time.perf_counter()
    model, variables = _build_model(
        model, instance, specification, obbt_bounds=obbt_bounds
    )
    model.update()
    build_seconds = time.perf_counter() - build_started
    relu_records = _relu_records(model) if specification["kind"] == "gnn" else []
    structure = _structure(model, relu_records)

    warm_statistics = {
        "warm_start_seconds": 0.0,
        "warm_start_status": None,
        "warm_start_values": 0,
    }
    if warm_start:
        warm_statistics = _apply_nominal_warm_start(
            model, variables, instance, warm_start_time_limit
        )

    root_lp_bound, root_lp_measurement_seconds = _plain_root_lp_bound(model)

    telemetry = {
        "root_node_bound_with_cuts": None,
        "time_to_first_solution": None,
        "time_to_best_solution": None,
        "best_callback_objective": math.inf,
    }
    model.optimize(_telemetry_callback(telemetry))
    has_solution = model.SolCount > 0
    result = {
        "instance": instance_name,
        "variant": variant_name,
        "status": _status_name(model.Status),
        "build_seconds": build_seconds,
        "preparation_seconds": (
            obbt_statistics["obbt_seconds"]
            + warm_statistics["warm_start_seconds"]
        ),
        "total_wall_seconds": (
            build_seconds
            + obbt_statistics["obbt_seconds"]
            + warm_statistics["warm_start_seconds"]
            + float(model.Runtime)
        ),
        "runtime_seconds": float(model.Runtime),
        "objective": float(model.ObjVal) if has_solution else None,
        "best_bound": float(model.ObjBound),
        "mip_gap": float(model.MIPGap) if has_solution else None,
        "nodes": float(model.NodeCount),
        "root_lp_bound": root_lp_bound,
        "root_lp_measurement_seconds": root_lp_measurement_seconds,
        "root_node_bound_with_cuts": telemetry[
            "root_node_bound_with_cuts"
        ],
        "time_to_first_solution": telemetry["time_to_first_solution"],
        "time_to_best_solution": telemetry["time_to_best_solution"],
        "convolution": specification.get("convolution"),
        "layers": (
            int(_metadata(specification).get("num_graphsage_layers", 1))
            if specification["kind"] == "gnn" else None
        ),
        "hidden_channels": specification.get("hidden_channels"),
        "edge_type": specification.get("edge_type", "machine_only"),
        "relu_formulation": specification.get("relu_formulation"),
        "bound_tightening": specification.get("bound_tightening"),
        "analytic_bounds": specification.get(
            "analytic_bounds", specification.get("bound_tightening")
        ),
        "structured_first_layer": specification.get(
            "structured_first_layer", True
        ),
        "model_path": specification.get("model_path"),
        "metadata_path": specification.get("metadata_path"),
        "solver_parameters": dict(solver_parameters or {}),
        **obbt_statistics,
        **warm_statistics,
        **structure,
    }
    if has_solution:
        exact_makespan = _exact_fixed_schedule_makespan(variables, instance)
        result["exact_fixed_makespan"] = exact_makespan
        result["surrogate_error"] = (
            float(model.ObjVal) - exact_makespan
            if exact_makespan is not None else None
        )
        if specification["kind"] == "gnn":
            node_error, total_error = _forward_error(
                variables, instance, specification
            )
            result["forward_max_node_probability_error"] = node_error
            result["forward_total_delay_error"] = total_error
    model.dispose()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--instances",
        nargs="+",
        default=["i3_k3_o3-5_15", "i3_k5_o3-5_10", "i5_k5_o3-5_17"],
    )
    parser.add_argument("--time-limit", type=float, default=20.0)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument(
        "--obbt-mode",
        choices=("none", "first_layer", "all"),
        default="none",
    )
    parser.add_argument("--obbt-lp-time-limit", type=float, default=10.0)
    parser.add_argument("--warm-start", action="store_true")
    parser.add_argument("--warm-start-time-limit", type=float, default=0.25)
    parser.add_argument(
        "--gurobi-param",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Controlled one-factor solver ablations such as MIPFocus=1.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    solver_parameters = {}
    for assignment in args.gurobi_param:
        if "=" not in assignment:
            parser.error("--gurobi-param requires NAME=VALUE.")
        name, raw_value = assignment.split("=", 1)
        try:
            value = float(raw_value)
            value = int(value) if value.is_integer() else value
        except ValueError:
            value = raw_value
        solver_parameters[name] = value
    if args.output is None:
        operation_ranges = sorted({
            match.group(1)
            for instance in args.instances
            if (match := re.search(r"_o([^_]+)_", instance))
        })
        operation_range = "-".join(operation_ranges) or "unknown"
        args.output = ROOT_DIR / "02_data" / (
            "gnn_benchmark_edges-mixed_arch-mixed_layers-mixed_hidden-mixed_"
            f"relu-mixed_obbt-{args.obbt_mode}_t{args.time_limit:g}_"
            f"operations-{operation_range}.json"
        )
    if args.output.exists() and not args.overwrite:
        parser.error(
            f"Refusing to overwrite existing raw data: {args.output}. "
            "Choose a new --output or pass --overwrite explicitly."
        )
    results = []
    for instance_name in args.instances:
        for variant_name in args.variants:
            result = _run(
                instance_name,
                variant_name,
                args.time_limit,
                obbt_mode=args.obbt_mode,
                obbt_lp_time_limit=args.obbt_lp_time_limit,
                warm_start=args.warm_start,
                warm_start_time_limit=args.warm_start_time_limit,
                solver_parameters=solver_parameters,
            )
            results.append(result)
            print(json.dumps(result, sort_keys=True), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
