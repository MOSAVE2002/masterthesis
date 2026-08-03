
import csv
import importlib
import json
from pathlib import Path
from pprint import pprint

from helper.start_solve_ins import (
    solve_instances_with_solver,
)
from helper.surrogate_constraint import target_column
from helper.sequence_setup import (
    RELIABILITY_GNN_GRAPH_SCHEMA,
    RELIABILITY_GNN_GRAPH_SCHEMA_JOB_ONLY,
    RELIABILITY_GNN_GRAPH_SCHEMA_WITH_JOB_EDGES,
    RELIABILITY_GNN_OUTPUT_HEAD,
    normalize_reliability_graph_config,
    reliability_node_feature_names,
)

_instance_generator = importlib.import_module("01_generator.instance_generator")
SPLIT_DIRECTORIES = _instance_generator.SPLIT_DIRECTORIES
SPLIT_CSV_FILENAMES = _instance_generator.SPLIT_CSV_FILENAMES
FIXED_SOLUTION_CSV_FILENAMES = (
    _instance_generator.FIXED_SOLUTION_CSV_FILENAMES
)
generate_instance_specs = _instance_generator.generate_instance_specs
generate_evaluation_instance_specs = (
    _instance_generator.generate_evaluation_instance_specs
)
generated_instance_names_by_split = _instance_generator.generated_instance_names_by_split
configured_instance_names_by_split = (
    _instance_generator.configured_instance_names_by_split
)
selected_instance_splits = _instance_generator.selected_instance_splits
load_generated_instance = _instance_generator.load_generated_instance

_gnn_architecture = importlib.import_module(
    "04_GraphNeuralNetworks.models.gnn_architecture"
)
validate_architecture = _gnn_architecture.validate_architecture
expand_architecture_variants = (
    _gnn_architecture.expand_architecture_variants
)

ROOT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = ROOT_DIR / "config.json" #Pfad bitte auf Config Path einstellen
MODEL_DIR = ROOT_DIR / "04_GraphNeuralNetworks" / "trained_gnn_models"
# Bitte alles in die Config Datei eintragen

GNN_EDGE_TYPES = {
    "machine_only": (True, False, RELIABILITY_GNN_GRAPH_SCHEMA),
    "job_only": (False, True, RELIABILITY_GNN_GRAPH_SCHEMA_JOB_ONLY),
    "both": (True, True, RELIABILITY_GNN_GRAPH_SCHEMA_WITH_JOB_EDGES),
}


def main():
    with CONFIG_PATH.open(encoding="utf-8") as file:
        config = json.load(file)

    # load data from config file
    workflow_cfg = config.get("workflow", {})
    solve_cfg = config.get("solve", {})
    print(
        "FJSP pipeline started | "
        f"create_instances={workflow_cfg.get('create_instances', False)} | "
        "generate_training_data="
        f"{workflow_cfg.get('generate_training_data', False)} | "
        f"train_gnn={workflow_cfg.get('train_gnn', False)} | "
        f"solve={workflow_cfg.get('solve', False)}",
        flush=True,
    )

    # Training data
    training_cfg = config.get("training", {})
    gnn_cfg = training_cfg.get("gnn", {})
    configured_model_dir = Path(
        gnn_cfg.get("model_directory", MODEL_DIR)
    )
    if not configured_model_dir.is_absolute():
        configured_model_dir = ROOT_DIR / configured_model_dir

    # Instance data
    instances_cfg = config.get("instances")
    generation_cfg = instances_cfg.get("generation")
    num_jobs = generation_cfg["num_jobs"]
    num_machines = generation_cfg["num_machines"]
    operations_per_job = generation_cfg["operations_per_job"]
    instances_per_size = generation_cfg["instances_per_size"]

    instance_specification = [
        {
            "num_jobs": jobs,
            "num_machines": machines,
            "operations_per_job": operations_per_job,
            "count": instances_per_size,
        }
        for jobs in num_jobs
        for machines in num_machines
    ]
    instance_names = [
        f"i{spec['num_jobs']}_k{spec['num_machines']}_"
        f"o{min(spec['operations_per_job'])}-{max(spec['operations_per_job'])}_"
        f"{instance_number}"
        for spec in instance_specification
        for instance_number in range(1, spec["count"] + 1)
    ]

    # Solver data
    solvers = solve_cfg.get("solvers", [])
    mip_gap = solve_cfg.get("mip_gap")

    # GNN Architecture
    raw_architectures = gnn_cfg.get("combinations")
    if not raw_architectures:
        raise ValueError(
            "training.gnn.combinations muss mindestens eine GNN-Architektur "
            "enthalten."
        )

    # Check if GNN Architecture is valid
    architecture_combinations = []
    seen_architectures = set()
    for raw in raw_architectures:
        for architecture in expand_architecture_variants(raw):
            if architecture["graph_mode"] != "fixed_candidate":
                raise ValueError(
                    "Die neue Expected-Delay-Pipeline unterstützt zunächst "
                    "nur graph_mode='fixed_candidate'."
                )
            key = (
                architecture["graph_mode"],
                architecture["convolution"],
                architecture["aggregation"],
                architecture["pooling"],
                architecture["layers"],
                architecture["hidden_channels"],
            )
            if key not in seen_architectures:
                architecture_combinations.append(architecture)
                seen_architectures.add(key)

    # Get Solver Data
    solver_cfg = config.get("solvers", {})
    gurobi_solver_cfg = solver_cfg.get("gurobi", {})
    gurobi_cfg = dict(gurobi_solver_cfg.get("common", {}))
    gurobi_gnn_cfg = dict(gurobi_solver_cfg.get("gnn", {}))
    gurobi_nonlinear_cfg = dict(gurobi_solver_cfg.get("nonlinear", {}))

    # Get Constraint Data
    constraint = config.get("constraint", {})
    weibull = constraint.get("weibull", {})
    reliability_graph = dict(weibull.get("reliability_graph", {}))
    constraint_cfg = {
        "type": constraint.get("type", "weibull"),
        "enforce_constraint": constraint.get("enforce", False),
        "reliability_graph_config": reliability_graph,
    }

    # Get Training Data
    data_generation = training_cfg.get("data_generation", {})
    reliability_ranges = data_generation.get("reliability_ranges", {})
    fixed_y = data_generation.get("fixed_y", {})
    data_generation_method = str(
        data_generation.get("method", "random_feasible")
    ).strip().lower()
    samples_per_instance = int(data_generation.get("samples_per_instance", 250))
    samples_per_split = data_generation.get("samples_per_split", {})
    if not isinstance(samples_per_split, dict):
        raise ValueError(
            "training.data_generation.samples_per_split muss ein "
            "JSON-Objekt sein."
        )
    data_cfg = {
        "method": data_generation_method,
        "generate_splits": data_generation.get("generate_splits"),
        "instance_names": data_generation.get("instance_names"),
        "output_directory": data_generation.get(
            "output_directory",
            "02_data/gnn_dataset",
        ),
        "random_seed": data_generation.get("random_seed", 42),
        "samples_per_instance": samples_per_instance,
        "samples_per_split": samples_per_split,
        "initial_age_range": reliability_ranges.get("initial_age", [0.0, 60.0]),
        "eta_range": reliability_ranges.get("eta", [50.0, 120.0]),
        "repair_duration_range": reliability_ranges.get(
            "repair_duration",
            reliability_ranges.get("failure_cost", [10.0, 30.0]),
        ),
        "deployment_share": float(
            reliability_ranges.get("deployment_share", 0.0)
        ),
        "deployment_reliability": reliability_ranges.get("deployment", {}),
        "fix_ratio_min": fixed_y.get("fix_ratio_min", 0.2),
        "fix_ratio_max": fixed_y.get("fix_ratio_max", 0.4),
        "fix_ratios": fixed_y.get("fix_ratios"),
        "sequence_fix_ratios": fixed_y.get(
            "sequence_fix_ratios", [0.0]
        ),
        "minimum_ratio_by_max_dimension": fixed_y.get(
            "minimum_ratio_by_max_dimension", {}
        ),
        "fixed_time_limit_seconds": fixed_y.get(
            "time_limit_seconds", 5
        ),
        "fixed_enforce_constraint": fixed_y.get(
            "enforce_constraint", False
        ),
        "fixed_output_flag": int(fixed_y.get("output_flag", 0)),
        "pool_solutions": int(fixed_y.get("pool_solutions", 1)),
        "pool_candidates": int(fixed_y.get("pool_candidates", 1)),
        "pool_search_mode": int(fixed_y.get("pool_search_mode", 0)),
        "pool_gap": float(fixed_y.get("pool_gap", 1.0)),
        "fixed_mip_gap": float(fixed_y.get("mip_gap", 0.2)),
        "fixed_mip_focus": int(fixed_y.get("mip_focus", 1)),
        "pool_selection_ratios": fixed_y.get(
            "pool_selection_ratios",
            {
                "very_good_effective": 0.30,
                "good_nominal_makespan": 0.15,
                "low_failure_delay": 0.075,
                "high_failure_delay": 0.075,
                "pareto_tradeoff": 0.20,
                "structurally_diverse": 0.10,
                "very_bad_effective": 0.10,
            },
        ),
        "split_ratios": data_generation.get(
            "split_ratios", {"train": 0.8, "valid": 0.1, "test": 0.1}
        ),
        "reliability_graph": reliability_graph,
    }

    # Generate Instances
    if _as_bool(workflow_cfg.get("create_instances", False)):
        generate_instance_specs(
            instance_specification,
            split_ratios=generation_cfg.get("split_ratios"),
            random_seed=generation_cfg.get("random_seed", 42),
        )

    train_gnn_requested = _as_bool(workflow_cfg.get("train_gnn", False))
    generate_training_data_requested = _as_bool(
        workflow_cfg.get("generate_training_data", False)
    )

    if generate_training_data_requested:
        data_cfg = dict(data_cfg)
        if _as_bool(workflow_cfg.get("create_instances", False)):
            data_cfg["instance_splits"] = configured_instance_names_by_split(
                instance_specification,
                split_ratios=generation_cfg.get("split_ratios"),
                random_seed=generation_cfg.get("random_seed", 42),
            )
        else:
            data_cfg["instance_splits"] = generated_instance_names_by_split()
        data_cfg["instance_splits"] = _select_training_data_instances(
            data_cfg["instance_splits"],
            data_cfg.get("instance_names"),
        )
        _generate_training_data(
            data_cfg,
            architecture_combinations[0]["graph_mode"],
            gurobi_cfg=gurobi_cfg,
            gurobi_nonlinear_cfg=gurobi_nonlinear_cfg,
            constraint_cfg=constraint_cfg,
        )

    if train_gnn_requested:
        train_from_config = importlib.import_module(
            "04_GraphNeuralNetworks.models.model_training_FJSP_GNN"
        ).train_from_config
        train_from_config()

    if _as_bool(workflow_cfg.get("solve", False)):
        solve_training_data_path = _configured_gnn_training_csv_path(
            data_cfg["output_directory"],
            data_generation_method=data_generation_method,
        )
        allowed_gnn_architectures = {
            (
                architecture["convolution"],
                architecture["aggregation"],
                architecture["pooling"],
                int(architecture["layers"]),
                int(architecture["hidden_channels"]),
            )
            for architecture in architecture_combinations
        }
        evaluation_cfg = solve_cfg.get("evaluation")
        evaluation_plan = (
            _prepare_solve_evaluation_plan(
                evaluation_cfg,
                instance_specification=instance_specification,
                generation_cfg=generation_cfg,
            )
            if evaluation_cfg is not None
            else [
                {
                    "tier": "configured_instances",
                    "instance_name": instance_name,
                    "instance_directory": None,
                }
                for instance_name in instance_names
            ]
        )
        solver_runs = []
        for solver in solvers:
            solver_name = solver.lower()
            fixed_y_params = _fixed_y_params_for_solver(
                solver_name, {}, gurobi_nonlinear_cfg
            )
            solver_specific_params = _solver_specific_params(
                solver_name,
                gurobi_cfg,
                gurobi_gnn_cfg,
                gurobi_nonlinear_cfg,
                constraint_cfg,
            )
            solver_specific_params = _with_mip_gap(
                solver_name, solver_specific_params, mip_gap
            )
            architectures = _solver_architecture_overrides(
                solver_name,
                model_seed=solver_specific_params.get("model_seed", 42),
                constraint_type=constraint_cfg["type"],
                reliability_graph_config=constraint_cfg[
                    "reliability_graph_config"
                ],
                expected_training_data_path=solve_training_data_path,
                allowed_architectures=allowed_gnn_architectures,
                edge_type=solver_specific_params.get(
                    "edge_type", "machine_only"
                ),
                model_dir=configured_model_dir,
            )
            if solver_name == "gurobi_gnn":
                found_architectures = {
                    (
                        architecture["convolution"],
                        architecture["aggregation"],
                        architecture["pooling"],
                        int(architecture["layers"]),
                        int(architecture["hidden_channels"]),
                    )
                    for architecture in architectures
                }
                missing_architectures = sorted(
                    allowed_gnn_architectures - found_architectures
                )
                pprint(
                    {
                        "GNN-Modelle werden gelöst": [
                            {
                                "architecture": (
                                    f"{architecture['convolution']}/"
                                    f"{architecture['aggregation']}"
                                ),
                                "layers": architecture["layers"],
                                "hidden": architecture["hidden_channels"],
                                "model_path": architecture["model_path"],
                            }
                            for architecture in architectures
                        ],
                        "Übersprungen (noch nicht trainiert)": [
                            {
                                "architecture": f"{conv}/{agg}",
                                "layers": layers,
                                "hidden": hidden,
                            }
                            for conv, agg, _pool, layers, hidden
                            in missing_architectures
                        ],
                    },
                    sort_dicts=False,
                )
            solver_runs.append((
                solver,
                solver_specific_params,
                fixed_y_params,
                architectures,
            ))

        solver_configurations = sum(
            len(architectures)
            for _, _, _, architectures in solver_runs
        )
        print(
            "Solve evaluation workload | "
            f"instances={len(evaluation_plan)} | "
            f"solver configurations={solver_configurations} | "
            f"total runs={len(evaluation_plan) * solver_configurations}",
            flush=True,
        )

        active_tier = None
        for evaluation_item in evaluation_plan:
            if evaluation_item["tier"] != active_tier:
                active_tier = evaluation_item["tier"]
                print(
                    "\n=== Solve evaluation tier: "
                    f"{active_tier} ===",
                    flush=True,
                )
            instance_name = evaluation_item["instance_name"]
            print(f"\nEvaluation instance: {instance_name}", flush=True)
            for (
                solver,
                solver_specific_params,
                fixed_y_params,
                architectures,
            ) in solver_runs:
                for architecture in architectures:
                    run_params = dict(solver_specific_params)
                    if architecture:
                        run_params.update(architecture)
                    if solver.lower() == "gurobi_gnn":
                        run_params["TimeLimit"] = solver_specific_params["TimeLimit"]
                    instance_directory = evaluation_item[
                        "instance_directory"
                    ]
                    if instance_directory is not None:
                        run_params["instance_directory"] = str(
                            instance_directory
                        )
                    solve_instances_with_solver(
                        instance_name=instance_name,
                        solver=solver,
                        **run_params,
                        **fixed_y_params,
                    )

def _as_bool(value):
    return value if isinstance(value, bool) else str(value).lower() == "true"


def _configured_gnn_training_csv_path(
    output_directory,
    data_generation_method,
):
    if not isinstance(output_directory, (str, Path)):
        raise ValueError(
            "training.data_generation.output_directory muss ein Pfad sein."
        )
    dataset_directory = Path(output_directory)
    if not dataset_directory.is_absolute():
        dataset_directory = ROOT_DIR / dataset_directory
    split_filenames = (
        FIXED_SOLUTION_CSV_FILENAMES
        if data_generation_method in {
            "gurobi_fixed",
            "gurobi_linear_labeled",
        }
        else SPLIT_CSV_FILENAMES
    )
    return (
        dataset_directory
        / SPLIT_DIRECTORIES["train"]
        / split_filenames["train"]
    ).resolve()


def _positive_integer(value, config_path):
    if isinstance(value, bool):
        raise ValueError(f"{config_path} muss eine positive ganze Zahl sein.")
    normalized = int(value)
    if normalized <= 0 or normalized != float(value):
        raise ValueError(f"{config_path} muss eine positive ganze Zahl sein.")
    return normalized


def _positive_integer_list(value, config_path):
    if not isinstance(value, list) or not value:
        raise ValueError(f"{config_path} muss eine nichtleere Liste sein.")
    normalized = [
        _positive_integer(item, config_path)
        for item in value
    ]
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{config_path} darf keine Duplikate enthalten.")
    return normalized


def _evaluation_tier_specs(
    tier_name,
    tier_config,
    operations_per_job,
):
    if not _as_bool(tier_config.get("enabled", True)):
        return []
    jobs = _positive_integer_list(
        tier_config.get("num_jobs"),
        f"solve.evaluation.{tier_name}.num_jobs",
    )
    machines = _positive_integer_list(
        tier_config.get("num_machines"),
        f"solve.evaluation.{tier_name}.num_machines",
    )
    count = _positive_integer(
        tier_config.get("instances_per_size", 1),
        f"solve.evaluation.{tier_name}.instances_per_size",
    )
    return [
        {
            "tier": tier_name,
            "num_jobs": jobs_count,
            "num_machines": machines_count,
            "operations_per_job": operations_per_job,
            "count": count,
        }
        for jobs_count in jobs
        for machines_count in machines
    ]


def _prepare_solve_evaluation_plan(
    evaluation_config,
    instance_specification,
    generation_cfg,
):
    if not isinstance(evaluation_config, dict):
        raise ValueError("solve.evaluation muss ein JSON-Objekt sein.")
    seed = int(evaluation_config.get("random_seed", 2026))
    operations_per_job = evaluation_config.get(
        "operations_per_job",
        generation_cfg["operations_per_job"],
    )
    plan = []

    in_distribution = evaluation_config.get("in_distribution", {})
    if not isinstance(in_distribution, dict):
        raise ValueError(
            "solve.evaluation.in_distribution muss ein JSON-Objekt sein."
        )
    if _as_bool(in_distribution.get("enabled", True)):
        count = _positive_integer(
            in_distribution.get("instances_per_size", 3),
            "solve.evaluation.in_distribution.instances_per_size",
        )
        configured_splits = configured_instance_names_by_split(
            instance_specification,
            split_ratios=generation_cfg.get("split_ratios"),
            random_seed=generation_cfg.get("random_seed", 42),
        )
        test_names = configured_splits["test"]
        for spec_index, spec in enumerate(instance_specification):
            prefix = (
                f"i{spec['num_jobs']}_k{spec['num_machines']}_"
                f"o{min(spec['operations_per_job'])}-"
                f"{max(spec['operations_per_job'])}_"
            )
            candidates = sorted(
                name for name in test_names if name.startswith(prefix)
            )
            if len(candidates) < count:
                raise ValueError(
                    f"Für {prefix[:-1]} gibt es nur {len(candidates)} "
                    f"Testinstanzen, benötigt werden {count}."
                )
            tier_rng = _instance_generator.random.Random(
                seed + spec_index
            )
            for instance_name in sorted(
                tier_rng.sample(candidates, count)
            ):
                plan.append({
                    "tier": "in_distribution",
                    "instance_name": instance_name,
                    "instance_directory": None,
                })

    tier_specs = []
    for tier_name in ("extrapolation", "stress"):
        tier_config = evaluation_config.get(tier_name, {})
        if not isinstance(tier_config, dict):
            raise ValueError(
                f"solve.evaluation.{tier_name} muss ein JSON-Objekt sein."
            )
        tier_specs.extend(_evaluation_tier_specs(
            tier_name,
            tier_config,
            operations_per_job,
        ))

    size_keys = [
        (spec["num_jobs"], spec["num_machines"])
        for spec in tier_specs
    ]
    if len(size_keys) != len(set(size_keys)):
        raise ValueError(
            "Extrapolation und Stress dürfen keine identischen "
            "Instanzgrößen enthalten."
        )

    if tier_specs:
        evaluation_root = Path(
            evaluation_config.get(
                "instance_directory",
                "02_data/fjsp_instances/evaluation",
            )
        )
        if not evaluation_root.is_absolute():
            evaluation_root = ROOT_DIR / evaluation_root
        for tier_name in ("extrapolation", "stress"):
            selected_specs = [
                spec for spec in tier_specs if spec["tier"] == tier_name
            ]
            if not selected_specs:
                continue
            tier_directory = evaluation_root / tier_name
            generator_specs = [
                {
                    key: spec[key]
                    for key in (
                        "num_jobs",
                        "num_machines",
                        "operations_per_job",
                        "count",
                    )
                }
                for spec in selected_specs
            ]
            generate_evaluation_instance_specs(
                generator_specs,
                random_seed=seed,
                output_directory=tier_directory,
            )

            configured_instance_names_by_split(
                generator_specs,
                split_ratios={
                    "train": 0.0,
                    "valid": 0.0,
                    "test": 1.0,
                },
                random_seed=seed,
                instance_directory=tier_directory,
            )
            for spec in selected_specs:
                for instance_number in range(1, spec["count"] + 1):
                    plan.append({
                        "tier": tier_name,
                        "instance_name": (
                            f"i{spec['num_jobs']}_k{spec['num_machines']}_"
                            f"o{min(spec['operations_per_job'])}-"
                            f"{max(spec['operations_per_job'])}_"
                            f"{instance_number}"
                        ),
                        "instance_directory": tier_directory,
                    })

    if not plan:
        raise ValueError(
            "Mindestens eine Solve-Evaluationsstufe muss aktiviert sein."
        )
    tier_counts = {
        tier: sum(item["tier"] == tier for item in plan)
        for tier in ("in_distribution", "extrapolation", "stress")
    }
    print(
        "Solve evaluation plan | "
        + " | ".join(
            f"{tier}={count}"
            for tier, count in tier_counts.items()
        ),
        flush=True,
    )
    return plan


def _discover_trained_gnn_overrides(
    graph_mode,
    model_seed,
    constraint_type,
    reliability_graph_config=None,
    expected_training_data_path=None,
    allowed_architectures=None,
    edge_type="machine_only",
    model_dir=MODEL_DIR,
):
    edge_type = str(edge_type).strip().lower()
    if edge_type not in GNN_EDGE_TYPES:
        raise ValueError(
            "solvers.gurobi.gnn.edge_type muss 'machine_only', "
            "'job_only' oder 'both' sein."
        )
    expected_machine, expected_job, expected_schema = GNN_EDGE_TYPES[
        edge_type
    ]
    expected_target = target_column(constraint_type)
    expected_graph_config = normalize_reliability_graph_config(
        reliability_graph_config
    )
    expected_features = reliability_node_feature_names(
        expected_graph_config
    )
    expected_training_data_path = (
        Path(expected_training_data_path).resolve()
        if expected_training_data_path is not None
        else None
    )
    allowed_architectures = (
        set(allowed_architectures)
        if allowed_architectures is not None
        else None
    )
    discovered = {}
    for metadata_path in Path(model_dir).rglob("*_meta.json"):
        with metadata_path.open(encoding="utf-8") as file:
            metadata = json.load(file)
        if metadata.get("graph_mode") != graph_mode:
            continue
        if metadata.get("target_column") != expected_target:
            continue
        if int(metadata.get("seed", -1)) != int(model_seed):
            continue
        if metadata.get("feature_names") != expected_features:
            continue
        if int(metadata.get("input_size", -1)) != len(expected_features):
            continue
        if expected_training_data_path is not None:
            metadata_training_path = metadata.get("training_data_path")
            if not metadata_training_path:
                continue
            if (
                Path(metadata_training_path).resolve()
                != expected_training_data_path
            ):
                continue
        try:
            metadata_graph_config = normalize_reliability_graph_config(
                metadata.get("reliability_graph_config")
            )
        except (TypeError, ValueError):
            continue
        if metadata_graph_config != expected_graph_config:
            continue
        if metadata.get("graph_schema") != expected_schema:
            continue
        if bool(
            metadata.get("include_machine_predecessor_edges", True)
        ) != expected_machine:
            continue
        if bool(
            metadata.get("include_job_precedence_edges", False)
        ) != expected_job:
            continue
        if metadata.get("node_target") != "operation_failure_probability":
            continue
        if metadata.get("output_head") != RELIABILITY_GNN_OUTPUT_HEAD:
            continue

        try:
            architecture = validate_architecture(
                metadata["graph_mode"],
                metadata["convolution"],
                metadata["aggregation"],
                metadata["pooling"],
            )
            layers = int(metadata["num_graphsage_layers"])
            hidden_channels = int(metadata["hidden_channels"])
        except (KeyError, TypeError, ValueError):
            continue
        if layers not in {1, 2, 3} or hidden_channels <= 0:
            continue

        model_path = metadata_path.with_name(
            metadata_path.name.removesuffix("_meta.json") + ".pt"
        )
        if not model_path.exists():
            continue
        key = (
            architecture["convolution"],
            architecture["aggregation"],
            architecture["pooling"],
            layers,
            hidden_channels,
        )
        if (
            allowed_architectures is not None
            and key not in allowed_architectures
        ):
            continue
        selection_rank = (
            model_path.stat().st_mtime_ns,
            metadata_path.stat().st_mtime_ns,
            -len(metadata_path.parts),
            str(metadata_path),
        )
        candidate = {
            "convolution": architecture["convolution"],
            "aggregation": architecture["aggregation"],
            "pooling": architecture["pooling"],
            "layers": layers,
            "hidden_channels": hidden_channels,
            "model_path": str(model_path),
            "metadata_path": str(metadata_path),
            "edge_type": edge_type,
            "_selection_rank": selection_rank,
        }
        current = discovered.get(key)
        if (
            current is None
            or candidate["_selection_rank"] > current["_selection_rank"]
        ):
            discovered[key] = candidate

    return [
        {
            field: value
            for field, value in discovered[key].items()
            if field != "_selection_rank"
        }
        for key in sorted(discovered)
    ]


def _solver_architecture_overrides(
    solver_name,
    model_seed=42,
    constraint_type="weibull",
    reliability_graph_config=None,
    expected_training_data_path=None,
    allowed_architectures=None,
    edge_type="machine_only",
    model_dir=MODEL_DIR,
):
    graph_mode = (
        "fixed_candidate" if solver_name == "gurobi_gnn" else None
    )
    if graph_mode is None:
        return [None]
    overrides = _discover_trained_gnn_overrides(
        graph_mode,
        model_seed,
        constraint_type,
        reliability_graph_config=reliability_graph_config,
        expected_training_data_path=expected_training_data_path,
        allowed_architectures=allowed_architectures,
        edge_type=edge_type,
        model_dir=model_dir,
    )
    return overrides


def _generate_training_data(
    data_cfg,
    gnn_graph_mode,
    gurobi_cfg,
    gurobi_nonlinear_cfg,
    constraint_cfg,
):
    method = str(data_cfg.get("method", "random_feasible")).strip().lower()
    instance_splits = (
        data_cfg.get("instance_splits")
        or generated_instance_names_by_split()
    )
    instance_splits = selected_instance_splits(
        instance_splits,
        data_cfg.get("generate_splits"),
    )
    empty_splits = [
        SPLIT_DIRECTORIES[split_name]
        for split_name, names in instance_splits.items()
        if not names
    ]
    if empty_splits:
        raise ValueError(
            "Keine Instanzen für folgende GNN-Datensplits gefunden: "
            f"{', '.join(empty_splits)}."
        )
    data_cfg = dict(data_cfg)
    data_cfg["instance_splits"] = instance_splits

    if method == "random_feasible":
        generate_from_config = importlib.import_module(
            "04_GraphNeuralNetworks.models.generate_weibull_training_data"
        ).generate_from_config
        generate_from_config(data_cfg)
        return

    if method not in {"gurobi_fixed", "gurobi_linear_labeled"}:
        raise ValueError(
            "training.data_generation.method must be "
            "'random_feasible', 'gurobi_fixed' or "
            "'gurobi_linear_labeled'."
        )

    solver_params = dict(gurobi_cfg)
    solver_params.update(constraint_cfg)
    solver_params.update(
        _without_keys(
            gurobi_nonlinear_cfg,
            "use_fixed_y",
            "create_fixed_y",
        )
    )
    solver_params["write_solution"] = False
    solver_params["write_gnn_inputs"] = True
    solver_params["gnn_graph_mode"] = gnn_graph_mode
    solver_params["enforce_constraint"] = _as_bool(
        data_cfg.get("fixed_enforce_constraint", False)
    )
    fixed_time_limit = float(
        data_cfg.get("fixed_time_limit_seconds", 5)
    )
    if fixed_time_limit <= 0:
        raise ValueError(
            "training.data_generation.fixed_y.time_limit_seconds "
            "muss groesser als 0 sein."
        )
    solver_params["TimeLimit"] = fixed_time_limit
    solver_params["OutputFlag"] = int(
        data_cfg.get("fixed_output_flag", 0)
    )
    pool_solutions = int(data_cfg.get("pool_solutions", 1))
    if pool_solutions <= 0:
        raise ValueError("fixed_y.pool_solutions muss groesser als 0 sein.")
    solver_params.update({
        "PoolSolutions": int(data_cfg.get("pool_candidates", pool_solutions)),
        "PoolSearchMode": int(data_cfg.get("pool_search_mode", 0)),
        "PoolGap": float(data_cfg.get("pool_gap", 1.0)),
        "MIPGap": float(data_cfg.get("fixed_mip_gap", 0.2)),
        "MIPFocus": int(data_cfg.get("fixed_mip_focus", 1)),
        "pool_output_limit": pool_solutions,
        "pool_selection_ratios": data_cfg["pool_selection_ratios"],
    })

    reliability_ranges = {
        "deployment_share": data_cfg["deployment_share"],
        "deployment": data_cfg["deployment_reliability"],
        "initial_age": data_cfg["initial_age_range"],
        "eta": data_cfg["eta_range"],
        "repair_duration": data_cfg["repair_duration_range"],
    }
    output_directory = Path(data_cfg["output_directory"])
    if not output_directory.is_absolute():
        output_directory = ROOT_DIR / output_directory
    for split_name, instance_names_for_split in instance_splits.items():
        split_samples = int(
            data_cfg.get("samples_per_split", {}).get(
                SPLIT_DIRECTORIES[split_name],
                data_cfg["samples_per_instance"],
            )
        )
        if split_samples <= 0:
            raise ValueError(
                "Die Samplezahl pro Datensplit muss groesser als 0 sein."
            )
        if split_samples % pool_solutions != 0:
            raise ValueError(
                "Die Samplezahl pro Split und Instanz muss durch "
                "fixed_y.pool_solutions teilbar sein."
            )
        optimization_runs_per_instance = split_samples // pool_solutions
        output_path = (
            output_directory
            / SPLIT_DIRECTORIES[split_name]
            / FIXED_SOLUTION_CSV_FILENAMES[split_name]
        )
        reset_output = True
        for instance_name in instance_names_for_split:
            fixed_ratio_params = _fixed_ratios_for_instance(
                data_cfg,
                instance_name,
            )
            fixed_ratio_params["sequence_fix_ratios"] = data_cfg[
                "sequence_fix_ratios"
            ]
            solve_instances_with_solver(
                instance_name=instance_name,
                solver=(
                    "gurobi_linear_labeled"
                    if method == "gurobi_linear_labeled"
                    else "gurobi_nonlinear"
                ),
                create_fixed_y=True,
                amount_of_samples_per_instance=(
                    optimization_runs_per_instance
                ),
                random_seed=data_cfg["random_seed"],
                merged_csv_path=str(output_path),
                reset_merged_csv=reset_output,
                reliability_ranges=reliability_ranges,
                **fixed_ratio_params,
                **solver_params,
            )
            reset_output = False

    if method == "gurobi_linear_labeled":
        total_rows = 0
        category_counts = {}
        effective_makespans = []
        for split_name in instance_splits:
            output_path = (
                output_directory
                / SPLIT_DIRECTORIES[split_name]
                / FIXED_SOLUTION_CSV_FILENAMES[split_name]
            )
            if not output_path.exists():
                continue
            with output_path.open(newline="", encoding="utf-8") as file:
                for row in csv.DictReader(file):
                    total_rows += 1
                    category = row.get("pool_selection_category", "")
                    if category:
                        category_counts[category] = (
                            category_counts.get(category, 0) + 1
                        )
                    effective_makespan = row.get("effective_makespan", "")
                    if effective_makespan not in (None, ""):
                        effective_makespans.append(float(effective_makespan))
        if total_rows:
            print(
                "Two-stage budget-free selection summary | "
                f"rows={total_rows} | categories="
                + ", ".join(
                    f"{category}:{count}"
                    for category, count in sorted(category_counts.items())
                )
            )
            if effective_makespans:
                print(
                    "  Effective makespan range: "
                    f"{min(effective_makespans):.4f}.."
                    f"{max(effective_makespans):.4f}"
                )


def _select_training_data_instances(
    instance_splits,
    configured_names,
):
    """Optionally restrict the already configured split-safe instances."""
    if configured_names in (None, ""):
        return {
            split_name: list(names)
            for split_name, names in instance_splits.items()
        }
    if not isinstance(configured_names, dict):
        raise ValueError(
            "training.data_generation.instance_names muss ein JSON-Objekt sein."
        )

    config_to_split = {
        "training": "train",
        "train": "train",
        "valid": "valid",
        "validation": "valid",
        "test": "test",
    }
    unknown = set(configured_names) - set(config_to_split)
    if unknown:
        raise ValueError(
            "Unbekannte Einträge in training.data_generation.instance_names: "
            f"{sorted(unknown)}."
        )

    selected = {}
    for config_key, raw_names in configured_names.items():
        split_name = config_to_split[config_key]
        if split_name in selected:
            raise ValueError(
                "training.data_generation.instance_names enthält mehrere "
                f"Bezeichnungen für den Split '{split_name}'."
            )
        if not isinstance(raw_names, list) or not raw_names:
            raise ValueError(
                "Jeder ausgewählte instance_names-Split muss eine nichtleere "
                "Liste enthalten."
            )
        names = [str(name) for name in raw_names]
        duplicates = sorted({
            name for name in names if names.count(name) > 1
        })
        if duplicates:
            raise ValueError(
                f"Doppelte Instanznamen im Split '{split_name}': {duplicates}."
            )
        available = set(instance_splits.get(split_name, []))
        invalid = sorted(set(names) - available)
        if invalid:
            raise ValueError(
                "Die folgenden Instanzen gehören nicht zum konfigurierten "
                f"Split '{split_name}' oder fehlen: {invalid}."
            )
        selected[split_name] = names

    return selected


def _fixed_ratios_for_instance(data_cfg, instance_name):
    fix_ratio_min = float(data_cfg["fix_ratio_min"])
    fix_ratio_max = float(data_cfg["fix_ratio_max"])
    if not 0.0 <= fix_ratio_min <= fix_ratio_max <= 1.0:
        raise ValueError(
            "fix_ratio_min und fix_ratio_max müssen zwischen 0 und 1 "
            "liegen und fix_ratio_min darf nicht größer sein."
        )

    configured_ratios = data_cfg.get("fix_ratios")
    fix_ratios = (
        None
        if configured_ratios in (None, "")
        else [float(value) for value in configured_ratios]
    )
    if fix_ratios is not None and (
        not fix_ratios
        or any(not 0.0 <= value <= 1.0 for value in fix_ratios)
    ):
        raise ValueError(
            "fix_ratios muss mindestens einen Wert zwischen 0 und 1 enthalten."
        )

    thresholds = data_cfg.get("minimum_ratio_by_max_dimension") or {}
    if not isinstance(thresholds, dict):
        raise ValueError(
            "minimum_ratio_by_max_dimension muss ein JSON-Objekt sein."
        )
    if not thresholds:
        return {
            "fix_ratio_min": fix_ratio_min,
            "fix_ratio_max": fix_ratio_max,
            "fix_ratios": fix_ratios,
        }

    instance = load_generated_instance(instance_name)
    max_dimension = max(int(instance.num_jobs), int(instance.num_machines))
    adaptive_minimum = 0.0
    for raw_dimension, raw_ratio in thresholds.items():
        dimension = int(raw_dimension)
        ratio = float(raw_ratio)
        if dimension <= 0 or not 0.0 <= ratio <= 1.0:
            raise ValueError(
                "Die Dimensionsgrenzen müssen positiv und ihre "
                "Mindest-Ratios zwischen 0 und 1 liegen."
            )
        if max_dimension >= dimension:
            adaptive_minimum = max(adaptive_minimum, ratio)

    if adaptive_minimum <= 0.0:
        return {
            "fix_ratio_min": fix_ratio_min,
            "fix_ratio_max": fix_ratio_max,
            "fix_ratios": fix_ratios,
        }

    fix_ratio_min = max(fix_ratio_min, adaptive_minimum)
    fix_ratio_max = max(fix_ratio_max, fix_ratio_min)
    if fix_ratios is not None:
        fix_ratios = [
            ratio for ratio in fix_ratios
            if ratio >= adaptive_minimum
        ]
        if not fix_ratios:
            fix_ratios = [adaptive_minimum]

    print(
        f"  Fixed-Y ratio for {instance_name}: max dimension "
        f"{max_dimension}, minimum {adaptive_minimum:.2f}"
    )
    return {
        "fix_ratio_min": fix_ratio_min,
        "fix_ratio_max": fix_ratio_max,
        "fix_ratios": fix_ratios,
    }


def _supports_fixed_y(solver_name):
    return solver_name in {"gurobi", "gurobi_nonlinear"}


def _fixed_y_params_for_solver(solver_name, fixed_y_cfg, gurobi_nonlinear_cfg):
    if not _supports_fixed_y(solver_name):
        return {}

    params = dict(fixed_y_cfg)
    if solver_name == "gurobi_nonlinear":
        use_fixed_y = _config_value(
            gurobi_nonlinear_cfg,
            "use_fixed_y",
            "create_fixed_y",
        )
        if use_fixed_y is not None:
            params["create_fixed_y"] = _as_bool(use_fixed_y)

    return params


def _config_value(config, *keys):
    for key in keys:
        if key in config:
            return config[key]
    return None


def _without_keys(config, *keys):
    normalized_keys = {key.lower() for key in keys}
    return {
        key: value
        for key, value in config.items()
        if key.lower() not in normalized_keys
    }


def _solver_specific_params(
    solver_name,
    gurobi_cfg,
    gurobi_gnn_cfg,
    gurobi_nonlinear_cfg,
    constraint_cfg,
):
    if solver_name in {"gurobi", "gurobi_ml"}:
        return dict(gurobi_cfg)

    if solver_name == "gurobi_gnn":
        merged_cfg = dict(gurobi_cfg)
        merged_cfg.update(constraint_cfg)
        merged_cfg.update(gurobi_gnn_cfg)
        return merged_cfg

    if solver_name == "gurobi_nonlinear":
        merged_cfg = dict(gurobi_cfg)
        merged_cfg.update(constraint_cfg)
        merged_cfg.update(
            _without_keys(
                gurobi_nonlinear_cfg,
                "use_fixed_y",
                "create_fixed_y",
            )
        )
        return merged_cfg


def _with_mip_gap(solver_name, solver_params, mip_gap):
    solver_params = dict(solver_params)
    if mip_gap in (None, ""):
        return solver_params

    mip_gap = float(mip_gap)
    if mip_gap < 0:
        raise ValueError("mip_gap muss groesser oder gleich 0 sein.")

    if solver_name in {
        "gurobi", "gurobi_gnn", "gurobi_nonlinear",
    }:
        solver_params.setdefault("MIPGap", mip_gap)
    else:
        if "limits/gap" not in solver_params and "limits_gap" not in solver_params:
            solver_params["limits/gap"] = mip_gap

    return solver_params


if __name__ == "__main__":
    main()
