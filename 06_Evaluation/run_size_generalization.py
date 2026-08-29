"""Pilot study: train on small FJSPs and solve progressively larger FJSPs.

All artifacts are isolated below ``06_Evaluation/results/size_generalization``.
The script deliberately does not modify the project's configured dataset,
trained models, evaluation instances, or solution directories.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


CONFIG_PATH = ROOT_DIR / "config.json"
DEFAULT_OUTPUT = ROOT_DIR / "06_Evaluation" / "results" / "size_generalization"
SIZE_GRID = ((3, 3), (5, 5), (7, 4), (8, 5), (10, 5))


def _load_config():
    with CONFIG_PATH.open(encoding="utf-8") as file:
        return json.load(file)


def _write_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_scaling_plot(output, rows):
    """Plot censored solve times and independently simulated robustness."""
    import matplotlib.pyplot as plt

    colors = {
        "in_distribution": "#1f77b4",
        "extrapolation": "#ff7f0e",
        "stress": "#d62728",
    }
    labels = {
        "in_distribution": "in distribution",
        "extrapolation": "extrapolation",
        "stress": "stress",
    }
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    seen = set()
    for row in rows:
        tier = row["evaluation_tier"]
        incumbent = row["postsolve_evaluation_status"] == "evaluated"
        axes[0].scatter(
            row["number_of_operations"],
            row["runtime_seconds"],
            color=colors[tier],
            marker="o" if incumbent else "x",
            s=55,
            label=labels[tier] if tier not in seen else None,
        )
        seen.add(tier)
        if incumbent:
            axes[1].scatter(
                row["number_of_operations"],
                row["minimum_mc_ontime_probability"],
                color=colors[tier],
                s=55,
            )
    axes[0].set(
        xlabel="number of operations",
        ylabel="Gurobi runtime [s]",
        title="Embedded GNN-MILP runtime",
    )
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.25)
    axes[1].set(
        xlabel="number of operations",
        ylabel="minimum MC on-time probability",
        title="Independent solution robustness",
        ylim=(0.80, 1.005),
    )
    axes[1].grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output / "scaling_results.png", dpi=180)
    figure.savefig(output / "scaling_results.pdf")
    plt.close(figure)


def _small_instance_splits(per_train=3, per_valid=1, per_test=1):
    instances = importlib.import_module("01_generator.instance_generator")
    available = instances.generated_instance_names_by_split()
    requested = {"train": per_train, "valid": per_valid, "test": per_test}
    selected = {}
    for split, count in requested.items():
        by_size = defaultdict(list)
        for name in available[split]:
            instance = instances.load_generated_instance(name)
            size = (int(instance.num_jobs), int(instance.num_machines))
            if size in {(3, 3), (3, 5), (5, 3), (5, 5)}:
                by_size[size].append(name)
        selected[split] = []
        for size in ((3, 3), (3, 5), (5, 3), (5, 5)):
            names = sorted(by_size[size])
            if len(names) < count:
                raise ValueError(
                    f"Need {count} {split} instances for size {size}, "
                    f"found {len(names)}."
                )
            selected[split].extend(names[:count])
    return selected


def generate_data(args):
    config = _load_config()
    source = config["training"]["data_generation"]
    fixed = dict(source["fixed_y"])
    fixed.update({
        "time_limit_seconds": float(args.candidate_time_limit),
        "pool_candidates": int(args.pool_candidates),
        "minimum_candidate_pool_runs": 2,
    })
    hybrid = dict(fixed.get("hybrid_selection") or {})
    hybrid["maximum_candidate_pool_runs"] = 4
    hybrid.pop("final_job_probability_coverage", None)
    fixed["hybrid_selection"] = hybrid
    dataset_dir = args.output / "dataset"
    generation = {
        "method": "fix_and_optimize",
        "instance_splits": _small_instance_splits(
            args.train_instances_per_size,
            args.valid_instances_per_size,
            args.test_instances_per_size,
        ),
        "generate_splits": {"training": True, "valid": True, "test": True},
        "output_directory": str(dataset_dir),
        "random_seed": int(args.seed),
        "samples_per_instance": int(args.samples_per_instance),
        "instance_failure_handling": source.get("instance_failure_handling"),
        "reliability_graph": config["constraint"]["weibull"]["reliability_graph"],
        "fixed_y": fixed,
    }
    settings_path = args.output / "experiment_settings.json"
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(
        json.dumps({"generation": generation, "size_grid": SIZE_GRID}, indent=2),
        encoding="utf-8",
    )
    generator = importlib.import_module(
        "04_GraphNeuralNetworks.models.generate_weibull_training_data"
    )
    return generator.generate_from_config(generation)


def train_model(args):
    config = _load_config()
    trainer = importlib.import_module(
        "04_GraphNeuralNetworks.models.model_training_FJSP_GNN"
    )
    dataset = args.output / "dataset"
    model_dir = args.output / "model"
    return trainer.train_from_file(
        dataset / "training" / "graphs_training.csv",
        dataset / "valid" / "graphs_valid.csv",
        dataset / "test" / "graphs_test.csv",
        seed=args.seed,
        epochs=args.epochs,
        hidden_channels=4,
        batch_size=32,
        learning_rate=0.001,
        graph_mode="fixed_candidate",
        convolution="sage",
        aggregation="sum",
        pooling="global_add",
        num_graphsage_layers=1,
        validation_interval=25,
        early_stopping_patience=150,
        enforce_graph_influence=False,
        expected_service_level=float(
            config["training"]["data_generation"]["fixed_y"].get(
                "label_distribution_center", 0.50
            )
        ),
        loss_name="mse",
        boundary_width=float(
            config["training"]["data_generation"]["fixed_y"][
                "label_distribution_half_width"
            ]
        ),
        output_stem="small_to_large_sage_l1_h4",
        model_dir=model_dir,
    )


def generate_evaluation_instances(args):
    config = _load_config()
    generator = importlib.import_module("01_generator.instance_generator")
    generation = config["instances"]["generation"]
    due_dates = dict(generation["due_dates"])
    tiers = {
        "in_distribution": ((3, 3), (5, 5)),
        "extrapolation": ((7, 4),),
        "stress": ((8, 5), (10, 5)),
    }
    for tier_index, (tier, sizes) in enumerate(tiers.items()):
        specs = [
            {
                "num_jobs": jobs,
                "num_machines": machines,
                "operations_per_job": generation["operations_per_job"],
                "count": args.evaluation_instances_per_size,
            }
            for jobs, machines in sizes
        ]
        generator.generate_evaluation_instance_specs(
            specs,
            random_seed=args.seed + 1000 + tier_index,
            output_directory=args.output / "instances" / tier,
            processing_time_range=generation["processing_times"]["base_range"],
            machine_profile_config=generation["machine_profiles"],
            due_date_config=due_dates,
            instance_name_suffix="sizegen",
        )


def _model_paths(args):
    base = args.output / "model" / "small_to_large_sage_l1_h4"
    return base.with_suffix(".pt"), base.parent / f"{base.name}_meta.json"


def solve_and_evaluate(args):
    config = _load_config()
    solve_model = importlib.import_module("helper.start_solve_ins").solveModel
    gnn = importlib.import_module("03_Gurobi.build_fjsp_with_gnn")
    evaluator = importlib.import_module("06_Evaluation.evaluate_solutions")
    simulation = config["evaluation"].get("simulation")
    model_path, metadata_path = _model_paths(args)
    instances_root = args.output / "instances"
    solutions_dir = args.output / "solutions"
    solutions_dir.mkdir(parents=True, exist_ok=True)
    schedule_rows, job_rows = [], []
    for instance_path in sorted(instances_root.rglob("*.pkl")):
        tier = instance_path.parent.name
        name = instance_path.stem
        result = solve_model(
            instance_name=name,
            instance_directory=str(instance_path.parent),
            solver="gurobi_gnn",
            model_path=str(model_path),
            metadata_path=str(metadata_path),
            convolution="sage",
            aggregation="sum",
            pooling="global_add",
            layers=1,
            hidden_channels=4,
            add_schedule_upper_bounds=True,
            analytic_bounds=True,
            constraint_type="weibull",
            reliability_graph_config=config["constraint"]["weibull"][
                "reliability_graph"
            ],
            OutputFlag=0,
            TimeLimit=float(args.solve_time_limit),
            MIPGap=float(args.solve_mip_gap),
            Threads=0,
            Seed=int(args.seed),
            write_solution=False,
        )
        solution_path = solutions_dir / (
            f"solution_{name}_gurobi_gnn_layers1_hidden4_seed{args.seed}.txt"
        )
        gnn.write_solution_file(
            result["model"], result["variables"], result["instance"], solution_path
        )
        schedule, jobs = evaluator.evaluate_solution(
            solution_path,
            instances_root=instances_root,
            replications=args.evaluation_replications,
            base_seed=args.seed + 900000,
            confidence=0.95,
            simulation_config=simulation,
        )
        instance = result["instance"]
        schedule.update({
            "evaluation_tier": tier,
            "number_of_machines": int(instance.num_machines),
            "number_of_operations": len(instance.real_operations),
            "build_plus_optimizer_seconds": result[
                "build_plus_optimizer_runtime"
            ],
            "model_variables": int(result["model"].NumVars),
            "model_binary_variables": int(result["model"].NumBinVars),
            "model_constraints": int(result["model"].NumConstrs),
        })
        schedule_rows.append(schedule)
        job_rows.extend(jobs)
        _write_csv(args.output / "schedule_results.csv", schedule_rows)
        _write_csv(args.output / "job_results.csv", job_rows)

    summary = {}
    for tier in ("in_distribution", "extrapolation", "stress"):
        rows = [row for row in schedule_rows if row["evaluation_tier"] == tier]
        solved = [row for row in rows if row["postsolve_evaluation_status"] == "evaluated"]
        summary[tier] = {
            "instances": len(rows),
            "incumbents": len(solved),
            "median_build_seconds": statistics.median(
                row["model_build_seconds"] for row in rows
            ) if rows else None,
            "median_optimizer_seconds": statistics.median(
                row["runtime_seconds"] for row in rows
            ) if rows else None,
            "mc_point_feasible_rate": (
                sum(bool(row["mc_all_jobs_point_feasible"]) for row in solved)
                / len(solved) if solved else None
            ),
            "bonferroni_feasible_rate": (
                sum(bool(row["bonferroni_wilson_all_jobs_feasible"]) for row in solved)
                / len(solved) if solved else None
            ),
            "mean_minimum_mc_probability": (
                statistics.fmean(row["minimum_mc_ontime_probability"] for row in solved)
                if solved else None
            ),
            "mean_maximum_service_shortfall": (
                statistics.fmean(row["maximum_service_shortfall"] for row in solved)
                if solved else None
            ),
        }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    _write_scaling_plot(args.output, schedule_rows)
    return summary


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("data", "train", "instances", "solve", "all"),
        default="all",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-instances-per-size", type=int, default=3)
    parser.add_argument("--valid-instances-per-size", type=int, default=1)
    parser.add_argument("--test-instances-per-size", type=int, default=1)
    parser.add_argument("--samples-per-instance", type=int, default=8)
    parser.add_argument("--pilot-replications", type=int, default=128)
    parser.add_argument("--label-replications", type=int, default=3000)
    parser.add_argument("--candidate-time-limit", type=float, default=1.0)
    parser.add_argument("--pool-candidates", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=1200)
    parser.add_argument("--evaluation-instances-per-size", type=int, default=2)
    parser.add_argument("--solve-time-limit", type=float, default=30.0)
    parser.add_argument("--solve-mip-gap", type=float, default=0.10)
    parser.add_argument("--evaluation-replications", type=int, default=5000)
    args = parser.parse_args(argv)
    args.output = args.output.resolve()
    return args


def main(argv=None):
    args = _parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.phase in {"data", "all"}:
        generate_data(args)
    if args.phase in {"train", "all"}:
        train_model(args)
    if args.phase in {"instances", "all"}:
        generate_evaluation_instances(args)
    if args.phase in {"solve", "all"}:
        print(json.dumps(solve_and_evaluate(args), indent=2))


if __name__ == "__main__":
    main()
