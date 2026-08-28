"""Compare PyTorch and Gurobi predictions for the same solved schedule.

This validator is deliberately independent of the main pipeline. Run it from
the repository root with::

    python3 tests/validate_gnn_embedding.py

The command solves one small instance with every GNN configured in
``config.json``, reconstructs the resulting graph independently for PyTorch,
and fails when a per-job prediction differs from the embedded MILP output by
more than the configured tolerance.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
from pathlib import Path

import torch
from torch_geometric.data import Data


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import main as pipeline
from helper.start_solve_ins import solve_instances_with_solver


_training = importlib.import_module(
    "04_GraphNeuralNetworks.models.model_training_FJSP_GNN"
)
FJSPGraphSAGE = _training.FJSPGraphSAGE


def _value(item) -> float:
    if hasattr(item, "X"):
        return float(item.X)
    if hasattr(item, "getValue"):
        return float(item.getValue())
    return float(item)


def _selected_machine(variables, operation):
    return max(
        variables["eligible_machines"][operation],
        key=lambda machine: _value(variables["Y"][operation, machine]),
    )


def _pytorch_graph(variables, instance):
    """Reconstruct the solved embedded graph without using training CSVs."""
    operations = list(variables["real_operations"])
    operation_to_index = {
        operation: index for index, operation in enumerate(operations)
    }
    jobs = tuple(sorted(instance.jobs))
    job_to_index = {job: index for index, job in enumerate(jobs)}
    operation_job = {
        operation: job
        for job, job_operations in instance.jobs.items()
        for operation in job_operations
    }
    horizon = max(float(value) for value in instance.due_dates.values())

    node_features = []
    for operation in operations:
        machine = _selected_machine(variables, operation)
        start = _value(variables["S"][operation])
        completion = _value(variables["C"][operation])
        alpha = float(variables["weibull_alpha"][machine])
        beta = float(variables["weibull_beta"][machine])
        repair_rate = float(variables["repair_rate"][machine])
        processing_time = float(
            instance.processing_times[operation, machine]
        )
        node_features.append([
            start / horizon,
            completion / horizon,
            processing_time / alpha,
            repair_rate * alpha / 30.0,
            beta / 5.0,
        ])

    machine_edges = []
    for source, target, machine in variables.get("U_index", []):
        if _value(variables["U"][source, target, machine]) <= 0.5:
            continue
        machine_edges.append((
            operation_to_index[source], operation_to_index[target]
        ))
    job_edges = []
    for target in operations:
        for source in instance.predecessors.get(target, []):
            if source not in operation_to_index:
                continue
            job_edges.append((
                operation_to_index[source], operation_to_index[target]
            ))

    edges = machine_edges + job_edges

    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    if job_edges:
        job_edge_index = torch.tensor(
            job_edges, dtype=torch.long
        ).t().contiguous()
    else:
        job_edge_index = torch.empty((2, 0), dtype=torch.long)

    graph = Data(
        x=torch.tensor(node_features, dtype=torch.float32),
        edge_index=edge_index,
        job_edge_index=job_edge_index,
        job_membership=torch.tensor(
            [job_to_index[operation_job[operation]] for operation in operations],
            dtype=torch.long,
        ),
        num_jobs_tensor=torch.tensor([len(jobs)], dtype=torch.long),
        batch=torch.zeros(len(operations), dtype=torch.long),
    )
    return graph, jobs


def _pytorch_prediction(model_path, metadata, graph):
    network = FJSPGraphSAGE(
        input_size=int(metadata["input_size"]),
        hidden_channels=int(metadata["hidden_channels"]),
        num_graphsage_layers=int(metadata["num_graphsage_layers"]),
        convolution=metadata["convolution"],
    )
    network.load_state_dict(torch.load(model_path, map_location="cpu"))
    network.eval()
    with torch.no_grad():
        return network(graph).detach().cpu().tolist()


def validate_result(result, *, tolerance=1e-5):
    """Return detailed parity records or raise on an embedding mismatch."""
    if result["solution_count"] <= 0:
        raise RuntimeError("Gurobi produced no incumbent to validate.")
    variables = result["variables"]
    metadata = variables["gnn_metadata"]
    graph, jobs = _pytorch_graph(variables, result["instance"])
    pytorch_values = _pytorch_prediction(
        variables["gnn_model_path"], metadata, graph
    )
    # The current target is an expected repair-time buffer, not a probability.
    # The embedded output already contains the same ReLU as the PyTorch model,
    # so values greater than one are valid and must not be clipped.
    gurobi_values = [
        _value(variables["gnn_job_output_expressions"][job])
        for job in jobs
    ]
    if len(pytorch_values) != len(gurobi_values):
        raise AssertionError(
            "PyTorch and Gurobi returned different numbers of jobs: "
            f"{len(pytorch_values)} != {len(gurobi_values)}"
        )

    records = []
    for job, pytorch_value, gurobi_value in zip(
        jobs, pytorch_values, gurobi_values
    ):
        error = abs(float(pytorch_value) - float(gurobi_value))
        if not all(math.isfinite(value) for value in (
            pytorch_value, gurobi_value, error
        )):
            raise AssertionError(f"Non-finite prediction for job {job}.")
        records.append({
            "job": job,
            "pytorch": float(pytorch_value),
            "gurobi": float(gurobi_value),
            "absolute_error": error,
        })

    maximum_error = max(record["absolute_error"] for record in records)
    if maximum_error > float(tolerance):
        raise AssertionError(
            "GNN embedding mismatch: "
            f"maximum error {maximum_error:.12g} exceeds "
            f"tolerance {float(tolerance):.12g}."
        )
    return records


def _default_instance(config):
    specs = pipeline._instance_specs(config["instances"]["generation"])
    splits = pipeline._configured_instance_splits(config, specs)
    plan = pipeline._in_distribution_plan(config, splits)
    if not plan:
        raise ValueError(
            "No in-distribution evaluation instance is configured; pass "
            "--instance explicitly."
        )
    return plan[0]["instance_name"], None


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Validate PyTorch/Gurobi GNN prediction parity."
    )
    parser.add_argument("--instance", help="Generated instance name.")
    parser.add_argument(
        "--instance-directory",
        help="Optional directory containing the instance pickle.",
    )
    parser.add_argument(
        "--convolution",
        choices=("linear", "sage", "job"),
        help="Validate only this configured GNN architecture.",
    )
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument("--time-limit", type=float, default=20.0)
    return parser.parse_args()


def main():
    args = _parse_args()
    if args.tolerance <= 0.0:
        raise ValueError("--tolerance must be positive.")
    if args.time_limit <= 0.0:
        raise ValueError("--time-limit must be positive.")

    with pipeline.CONFIG_PATH.open(encoding="utf-8") as file:
        config = json.load(file)
    if args.instance:
        instance_name = args.instance
        instance_directory = args.instance_directory
    else:
        instance_name, instance_directory = _default_instance(config)

    models = pipeline._trained_models(config)
    if args.convolution:
        models = [
            model for model in models
            if model["convolution"] == args.convolution
        ]
    if not models:
        raise ValueError("No configured GNN model matches the selection.")

    graph_config = config["constraint"]["weibull"]["reliability_graph"]
    common = dict(config["solvers"]["gurobi"].get("common", {}))
    common.update({
        "OutputFlag": 0,
        "TimeLimit": float(args.time_limit),
        "write_solution": False,
    })
    gnn_config = dict(config["solvers"]["gurobi"].get("gnn", {}))
    gnn_config["write_solution"] = False

    summaries = []
    for model_specification in models:
        parameters = {
            **common,
            **gnn_config,
            **model_specification,
            "solver": "gurobi_gnn",
            "instance_name": instance_name,
            "constraint_type": config["constraint"]["type"],
            "reliability_graph_config": graph_config,
        }
        if instance_directory:
            parameters["instance_directory"] = instance_directory
        result = solve_instances_with_solver(**parameters)
        records = validate_result(result, tolerance=args.tolerance)
        maximum_error = max(
            record["absolute_error"] for record in records
        )
        summary = {
            "instance": instance_name,
            "convolution": model_specification["convolution"],
            "layers": model_specification["layers"],
            "hidden_channels": model_specification["hidden_channels"],
            "jobs": len(records),
            "maximum_absolute_error": maximum_error,
            "tolerance": float(args.tolerance),
        }
        summaries.append(summary)
        print(
            "EMBEDDING_OK | "
            f"instance={instance_name} | "
            f"model={model_specification['convolution']} | "
            f"jobs={len(records)} | max_error={maximum_error:.3e} | "
            f"tolerance={args.tolerance:.3e}",
            flush=True,
        )

    print(json.dumps({"embedding_validation": summaries}, indent=2))


if __name__ == "__main__":
    main()
