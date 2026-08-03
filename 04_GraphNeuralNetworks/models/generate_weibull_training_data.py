import csv
import importlib
import json
import math
import random
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from helper.sequence_setup import (
    RELIABILITY_EDGE_FEATURE_NAMES,
    fixed_machine_multiedges,
    machine_processing_max,
    normalize_reliability_graph_config,
    reliability_graph_config_dict,
    reliability_node_feature_names,
    transition_edge_values,
)

_instance_generator = importlib.import_module("01_generator.instance_generator")
SPLIT_DIRECTORIES = _instance_generator.SPLIT_DIRECTORIES
SPLIT_CSV_FILENAMES = _instance_generator.SPLIT_CSV_FILENAMES
configured_instance_names_by_split = (
    _instance_generator.configured_instance_names_by_split
)
selected_instance_splits = _instance_generator.selected_instance_splits
generated_instance_names_by_split = _instance_generator.generated_instance_names_by_split
load_generated_instance = _instance_generator.load_generated_instance


CONFIG_PATH = ROOT_DIR / "config.json"


def _compact(value):
    return json.dumps(value, separators=(",", ":"))


def _sample_machine_values(instance, rng, config):
    def sample_range(key, default):
        low, high = config.get(key, default)
        return {
            machine: rng.uniform(float(low), float(high))
            for machine in range(instance.num_machines)
        }

    return (
        sample_range("initial_age_range", [0.0, 60.0]),
        sample_range("eta_range", [50.0, 120.0]),
        sample_range(
            "repair_duration_range",
            config.get("failure_cost_range", [10.0, 30.0]),
        ),
    )


def _random_feasible_schedule(
    instance,
    rng,
    initial_age,
    eta,
    repair_duration,
    graph_config,
):
    cfg = normalize_reliability_graph_config(graph_config)
    selected_machine = {
        operation: rng.choice(instance.eligible_machines[operation])
        for operation in instance.real_operations
    }
    next_position = {job: 0 for job in instance.jobs}
    job_ready = {job: 0.0 for job in instance.jobs}
    machine_ready = {machine: 0.0 for machine in range(instance.num_machines)}
    machine_age = dict(initial_age)
    machine_previous = {
        machine: None for machine in range(instance.num_machines)
    }
    processing_max = machine_processing_max(instance)
    remaining_jobs = list(instance.jobs)
    starts, completions, ages = {}, {}, {}
    probabilities, delays, predecessors = {}, {}, {}

    while remaining_jobs:
        job = rng.choice(remaining_jobs)
        operation = instance.jobs[job][next_position[job]]
        machine = selected_machine[operation]
        processing = float(instance.processing_times[operation, machine])
        age = machine_age[machine]
        predecessor = machine_previous[machine]
        transition = 0.0
        if predecessor is not None:
            transition = abs(
                processing
                - float(instance.processing_times[predecessor, machine])
            ) / processing_max[machine]
        weibull_increment = (
            ((age + processing) / eta[machine]) ** cfg.beta
            - (age / eta[machine]) ** cfg.beta
        )
        hazard = weibull_increment + cfg.transition_gamma * transition
        probability = 1.0 - math.exp(-hazard)
        delay = repair_duration[machine] * probability
        start = max(job_ready[job], machine_ready[machine])
        completion = start + processing + delay

        starts[operation], completions[operation] = start, completion
        ages[operation], predecessors[operation] = age, predecessor
        probabilities[operation], delays[operation] = probability, delay
        job_ready[job] = completion
        machine_ready[machine] = completion
        machine_age[machine] += processing
        machine_previous[machine] = operation
        next_position[job] += 1
        if next_position[job] == len(instance.jobs[job]):
            remaining_jobs.remove(job)

    return {
        "selected_machine": selected_machine,
        "starts": starts,
        "completions": completions,
        "ages": ages,
        "probabilities": probabilities,
        "delays": delays,
        "predecessors": predecessors,
    }


def _build_row(instance, rng, reliability_config):
    operations = list(instance.real_operations)
    initial_age, eta, repair_duration = _sample_machine_values(
        instance, rng, reliability_config
    )
    graph_cfg = normalize_reliability_graph_config(
        reliability_config.get("reliability_graph")
    )
    schedule = _random_feasible_schedule(
        instance,
        rng,
        initial_age,
        eta,
        repair_duration,
        graph_cfg,
    )
    selected = schedule["selected_machine"]
    predecessors = schedule["predecessors"]
    node_features, node_delays = [], []
    node_probabilities, node_repair_durations = [], []
    for operation in operations:
        machine = selected[operation]
        node_features.append([
            float(instance.processing_times[operation, machine]) / eta[machine],
            schedule["ages"][operation] / eta[machine],
            graph_cfg.beta / 5.0,
        ])
        node_delays.append(schedule["delays"][operation])
        node_probabilities.append(schedule["probabilities"][operation])
        node_repair_durations.append(repair_duration[machine])

    fixed_edges = fixed_machine_multiedges(instance, operations)
    processing_max = machine_processing_max(instance, operations)
    active_edge_indices, active_edge_features = [], []
    for edge_index, (source_index, target_index, machine) in enumerate(fixed_edges):
        source, target = operations[source_index], operations[target_index]
        if selected[target] != machine or predecessors[target] != source:
            continue
        active_edge_indices.append(edge_index)
        active_edge_features.append(list(transition_edge_values(
            instance, source, target, machine, eta, processing_max
        )))
    if len(active_edge_indices) != sum(
        predecessor is not None for predecessor in predecessors.values()
    ):
        raise AssertionError(
            "Every non-first operation needs exactly one direct predecessor edge."
        )

    return {
        "total_failure_delay": sum(node_delays),
        "gnn_feature_names": _compact(reliability_node_feature_names()),
        "gnn_edge_feature_names": _compact(RELIABILITY_EDGE_FEATURE_NAMES),
        "gnn_node_features": _compact(node_features),
        "gnn_active_edge_indices": _compact(active_edge_indices),
        "gnn_active_edge_features": _compact(active_edge_features),
        "operation_failure_delays": _compact(node_delays),
        "operation_failure_probabilities": _compact(node_probabilities),
        "operation_repair_durations": _compact(node_repair_durations),
        "reliability_graph_parameters": _compact(
            reliability_graph_config_dict(graph_cfg)
        ),
    }


def generate_from_config(generation=None):
    if generation is None:
        with CONFIG_PATH.open(encoding="utf-8") as file:
            config = json.load(file)
        source = config.get("training", {}).get("data_generation", {})
        instance_generation = config.get("instances", {}).get("generation", {})
        instance_specs = [
            {
                "num_jobs": jobs,
                "num_machines": machines,
                "operations_per_job": instance_generation["operations_per_job"],
                "count": instance_generation["instances_per_size"],
            }
            for jobs in instance_generation.get("num_jobs", [])
            for machines in instance_generation.get("num_machines", [])
        ]
        ranges = source.get("reliability_ranges", {})
        fixed_y = source.get("fixed_y", {})
        generation = {
            "method": source.get("method", "random_feasible"),
            "generate_splits": source.get("generate_splits"),
            "output_directory": source.get(
                "output_directory", "02_data/gnn_dataset"
            ),
            "random_seed": source.get("random_seed", 42),
            "samples_per_instance": int(source.get("samples_per_instance", 250)),
            "instance_splits": configured_instance_names_by_split(
                instance_specs,
                split_ratios=instance_generation.get("split_ratios"),
                random_seed=instance_generation.get("random_seed", 42),
            ),
            "initial_age_range": ranges.get("initial_age", [0.0, 60.0]),
            "eta_range": ranges.get("eta", [50.0, 120.0]),
            "repair_duration_range": ranges.get(
                "repair_duration", ranges.get("failure_cost", [10.0, 30.0])
            ),
            "reliability_graph": (
                config.get("constraint", {})
                .get("weibull", {})
                .get("reliability_graph", {})
            ),
            "fix_ratio_min": fixed_y.get("fix_ratio_min", 0.2),
            "fix_ratio_max": fixed_y.get("fix_ratio_max", 0.4),
        }

    output_directory = Path(
        generation.get("output_directory", "02_data/gnn_dataset")
    )
    if not output_directory.is_absolute():
        output_directory = ROOT_DIR / output_directory
    seed = int(generation.get("random_seed", 42))
    samples_per_instance = int(generation.get("samples_per_instance", 250))
    if samples_per_instance <= 0:
        raise ValueError("samples_per_instance muss groesser als 0 sein.")
    instance_splits = generation.get("instance_splits") or generated_instance_names_by_split()
    instance_splits = selected_instance_splits(
        instance_splits, generation.get("generate_splits")
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

    fieldnames = [
        "instance_name", "total_failure_delay", "gnn_feature_names",
        "gnn_edge_feature_names", "gnn_node_features",
        "gnn_active_edge_indices", "gnn_active_edge_features",
        "operation_failure_delays", "operation_failure_probabilities",
        "operation_repair_durations", "reliability_graph_parameters",
    ]
    split_instance_sets = {
        split_name: set(names) for split_name, names in instance_splits.items()
    }
    for split_name, names in split_instance_sets.items():
        for other_split, other_names in split_instance_sets.items():
            if split_name < other_split and names & other_names:
                raise ValueError(
                    "Instanzen kommen in mehreren Datensplits vor: "
                    f"{sorted(names & other_names)}"
                )

    for split_name, instance_names in instance_splits.items():
        rng = random.Random(f"{seed}:{split_name}")
        output_path = (
            output_directory / SPLIT_DIRECTORIES[split_name]
            / SPLIT_CSV_FILENAMES[split_name]
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        total_rows = len(instance_names) * samples_per_instance
        written_rows = 0
        print(
            f"[GNN data] Starting {split_name}: {len(instance_names)} "
            f"instances × {samples_per_instance} samples = "
            f"{total_rows} graphs -> {output_path}", flush=True,
        )
        with output_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for instance_index, instance_name in enumerate(instance_names, start=1):
                instance = load_generated_instance(instance_name)
                for _sample_id in range(samples_per_instance):
                    row = _build_row(instance, rng, generation)
                    row["instance_name"] = instance_name
                    writer.writerow(row)
                    written_rows += 1
                    if written_rows % 1000 == 0:
                        print(
                            f"[GNN data] {split_name}: {written_rows}/{total_rows} "
                            f"graphs ({100.0 * written_rows / total_rows:.1f}%)",
                            flush=True,
                        )
                file.flush()
                print(
                    f"[GNN data] {split_name}: instance "
                    f"{instance_index}/{len(instance_names)} ({instance_name}) "
                    f"complete; {written_rows}/{total_rows} graphs", flush=True,
                )
        print(
            f"[GNN data] Finished {split_name}: {written_rows} graphs "
            f"from {len(instance_names)} instances -> {output_path}", flush=True,
        )


if __name__ == "__main__":
    generate_from_config()
