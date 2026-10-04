"""Execute configured solver plans and persist a durable run manifest.

Each benchmark or extrapolation instance is dispatched to the requested
nominal, nonlinear and GNN formulations. Result metadata is appended after
every run and the Gurobi model is disposed immediately afterward.
"""

import json
from pathlib import Path

from helper.start_solve_ins import solve_instance


ROOT_DIR = Path(__file__).resolve().parents[1]
RESULT_KEYS = (
    "status",
    "solution_count",
    "objective",
    "best_bound",
    "mip_gap",
    "branch_and_bound_nodes",
    "linear_matrix_nonzeros",
    "runtime",
    "model_build_runtime",
    "optimizer_wall_runtime",
    "build_plus_optimizer_runtime",
    "solution_path",
)


def _absolute(path):
    """Resolve a configured path relative to the repository root.

    Absolute paths are preserved for externally selected output locations.
    """
    path = Path(path)
    return path if path.is_absolute() else ROOT_DIR / path


def solve_plan(
    config,
    plan,
    requested,
    models,
    common,
    records=None,
):
    """Run all requested formulations for one tier plan.

    Args:
        config: Complete project configuration.
        plan: Ordered solve cases containing instance names and directories.
        requested: Solver identifiers to execute for every case.
        models: Trained GNN artifact pairs used by the GNN solver.
        common: Shared Gurobi and economic parameters.
        records: Optional existing result list to extend across tiers.

    Returns:
        The accumulated compact solver-result records.
    """
    records = [] if records is None else records
    manifest_directory = _absolute(config["solve"]["manifest_directory"])
    manifest_directory.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_directory / "solve_manifest.json"
    solver_config = config["solvers"]["gurobi"]

    def run(**parameters):
        """Execute one solver/model case and immediately persist its result.

        The live Gurobi model is disposed after compact metrics are extracted.
        """
        model_label = (
            Path(parameters["model_path"]).stem
            if parameters.get("model_path") else "-"
        )
        print(
            f"[Solve] starting instance={parameters['instance_name']} | "
            f"solver={parameters['solver']} | model={model_label}",
            flush=True,
        )
        full_result = solve_instance(**parameters)
        try:
            result = {
                **{key: full_result.get(key) for key in RESULT_KEYS},
                "solver": parameters["solver"],
                "instance_name": parameters["instance_name"],
                "model_path": parameters.get("model_path"),
            }
        finally:
            full_result["model"].dispose()
        records.append(result)
        manifest_path.write_text(json.dumps({"runs": records}, indent=2))
        print(
            f"[Solve] finished instance={parameters['instance_name']} | "
            f"solver={parameters['solver']} | model={model_label} | "
            f"status={result.get('status')} | runtime={result.get('runtime')}",
            flush=True,
        )

    for instance_index, item in enumerate(plan, start=1):
        print(
            f"[Solve] benchmark case {instance_index}/{len(plan)} | "
            f"instance={item['instance_name']}",
            flush=True,
        )
        instance_args = {
            "instance_name": item["instance_name"],
            "instance_directory": item["instance_directory"],
        }
        for solver in requested:
            if solver == "gurobi":
                run(solver=solver, **instance_args, **common)
            elif solver == "gurobi_nonlinear":
                run(
                    solver=solver,
                    **instance_args,
                    **common,
                    **solver_config.get("nonlinear", {}),
                )
            elif solver == "gurobi_gnn":
                for trained in models:
                    run(
                        solver=solver,
                        **instance_args,
                        **common,
                        **solver_config.get("gnn", {}),
                        **trained,
                    )
            else:
                raise ValueError(f"Unknown solver: {solver}")
        print(
            f"[Solve] completed benchmark case {instance_index}/{len(plan)} | "
            f"instance={item['instance_name']}",
            flush=True,
        )
    return records
