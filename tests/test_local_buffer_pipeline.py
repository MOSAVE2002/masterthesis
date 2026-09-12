"""Checks of local target semantics, candidate diversity and metadata isolation."""
from dataclasses import replace
import importlib
import json
from pathlib import Path

import pytest

from helper.local_buffer import (
    expected_job_buffers, InvalidScheduleError, LABEL_METHOD, LABEL_SOURCE,
    TARGET_COLUMN, label_config_dict,
)
from helper.buffer_candidate_selection import input_signature, select_buffer_candidates
from helper.sequence_setup import RELIABILITY_GNN_GRAPH_SCHEMA, reliability_node_feature_names

simulation = importlib.import_module("05_Simulation.local_midpoint")
training = importlib.import_module("04_GraphNeuralNetworks.models.model_training_FJSP_GNN")


def schedule():
    return simulation.FixedSchedule(
        operations=(0, 1), selected_machines={0: 0, 1: 0},
        processing_times={0: 8., 1: 8.}, planned_starts={0: 10., 1: 20.},
        job_predecessors={0: (), 1: (0,)}, machine_edges=((0, 1, 0),),
        jobs={0: (0, 1)}, job_end_operations={0: 1}, due_dates={0: 40.},
        weibull_scale={0: 25., 1: 25.}, weibull_shape={0: 2., 1: 2.},
        repair_rate={0: .1, 1: .1},
    )


def test_local_expectation_independent_of_machine_identity_and_correlation():
    common = schedule()
    separate = replace(common, selected_machines={0: 0, 1: 1}, machine_edges=())
    assert expected_job_buffers(common) == expected_job_buffers(separate)
    for plan in (common, separate):
        values, errors = expected_job_buffers(plan)
        mc = simulation.simulate_fixed_schedule(plan, replications=200_000, seed=19)
        assert abs(values[0] - mc.job_mean_completion_delays[0]) < 5 * mc.job_completion_delay_standard_errors[0]
        assert errors[0] < 1e-6


def test_invalid_nominal_schedule_rejected():
    with pytest.raises(InvalidScheduleError):
        expected_job_buffers(replace(schedule(), planned_starts={0: 10., 1: 12.}))


def candidate(index, cost, buffer):
    return {
        "row": {"gnn_node_features": json.dumps([[index, 0], [index, 1]]),
                "operation_job_indices": "[0,0]", "gnn_active_edges": "[[0,1],[0,1]]"},
        "structure": (((0, index % 2), (1, index % 3)), ((0, 1, index % 2),)),
        "local_job_repair_buffers": [buffer], "nominal_schedule_cost": cost,
    }


def test_selection_retains_time_variants_and_expensive_low_buffers():
    pool = [candidate(i, 10 + i // 6 * 100, (i % 6) * 10) for i in range(24)]
    duplicate = dict(pool[0], row=dict(pool[0]["row"]))
    assert input_signature(pool[0]) != input_signature(pool[6])
    selected = select_buffer_candidates(pool + [duplicate], 12)
    assert len({input_signature(e["candidate"]) for e in selected}) == 12
    categories = [e["category"] for e in selected]
    assert "cost_1_buffer_0" in categories
    assert categories.count("buffer_coverage") == 6
    assert categories.count("structure") == 2
    duplicate["row"]["gnn_active_edges"] = "[[0,1]]"
    assert input_signature(pool[0]) != input_signature(duplicate)


def metadata():
    return {"status": "completed", "label": {
        "target_column": TARGET_COLUMN, "label_method": LABEL_METHOD,
        "source": LABEL_SOURCE, "parameters": label_config_dict(),
    }, "graph": {
        "graph_schema": RELIABILITY_GNN_GRAPH_SCHEMA, "service_scope": "job",
        "feature_names": reliability_node_feature_names(),
        "include_machine_predecessor_edges": True,
        "machine_predecessor_edge_scope": "direct", "include_job_precedence_edges": True,
    }}


def test_training_rejects_old_labels_and_unfinished_generation(tmp_path):
    path = tmp_path / "training" / "graphs_training.csv"
    summary_path = tmp_path / "generation_summary.json"
    good = metadata()
    summary_path.write_text(json.dumps(good))
    assert training._load_label_metadata(path, TARGET_COLUMN)[0] == LABEL_METHOD
    for field, value in (("label_method", "monte_carlo_old"), ("target_column", "simulated_expected_completion_delay")):
        bad = metadata(); bad["label"][field] = value
        summary_path.write_text(json.dumps(bad))
        with pytest.raises(ValueError):
            training._load_label_metadata(path, TARGET_COLUMN)
    good["status"] = "running"
    summary_path.write_text(json.dumps(good))
    with pytest.raises(ValueError, match="incomplete"):
        training._load_label_metadata(path, TARGET_COLUMN)


def test_partial_generation_cannot_relabel_retained_old_data(tmp_path):
    generator = importlib.import_module("04_GraphNeuralNetworks.models.generate_fix_and_optimize_training_data")
    retained = tmp_path / "valid" / "graphs_valid.csv"
    retained.parent.mkdir()
    retained.write_text("old validation data")
    summary = tmp_path / "generation_summary.json"
    summary.write_text(json.dumps({"status": "completed", "label": {"target_column": "old"}}))
    before = summary.read_bytes()
    with pytest.raises(ValueError, match="incompatible"):
        generator.generate_from_config({
            "method": "fix_and_optimize", "samples_per_instance": 12,
            "instance_splits": {"train": [], "valid": [], "test": []},
            "generate_splits": {"training": True, "valid": False, "test": False},
            "fixed_y": {"fix_ratios": [.25]}, "output_directory": str(tmp_path),
        })
    assert summary.read_bytes() == before
    assert retained.read_text() == "old validation data"
