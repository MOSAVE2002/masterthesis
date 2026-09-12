import importlib
import json

import pytest


analysis = importlib.import_module(
    "06_Evaluation.summarize_numerical_analysis"
)


def _row(instance, model, runtime, reached):
    return {
        "physical_instance_id": instance,
        "instance_name": instance.split("/")[-1],
        "due_date_condition": "offset:0",
        "due_date_relative_makespan_offset": 0.0,
        "evaluation_tier": "benchmark",
        "number_of_jobs": 5,
        "number_of_machines": 5,
        "model": model,
        "status": "OPTIMAL" if reached else "TIME_LIMIT",
        "has_incumbent": reached,
        "mip_gap": 0.0 if reached else 0.5,
        "runtime_seconds": runtime,
    }


def test_describe_reports_median_quartiles_and_iqr():
    result = analysis.describe([1.0, 2.0, 3.0, 4.0], "runtime")

    assert result["runtime_mean"] == pytest.approx(2.5)
    assert result["runtime_median"] == pytest.approx(2.5)
    assert result["runtime_q1"] == pytest.approx(1.75)
    assert result["runtime_q3"] == pytest.approx(3.25)
    assert result["runtime_iqr"] == pytest.approx(1.5)


def test_par2_uses_twice_time_limit_when_target_gap_is_not_reached():
    rows = [
        _row("benchmark/a", "model", 10.0, True),
        _row("benchmark/b", "model", 60.0, False),
    ]

    summary = analysis.solver_summary(
        rows, ("evaluation_tier", "model"), 60.0, 0.01
    )[0]

    assert summary["runtime_seconds_median"] == pytest.approx(35.0)
    assert summary["par2_target_gap_seconds_mean"] == pytest.approx(65.0)
    assert summary["par2_target_gap_seconds_median"] == pytest.approx(65.0)
    assert summary["target_gap_rate"] == pytest.approx(0.5)


def test_performance_profile_uses_common_physical_due_date_cases():
    rows = [
        _row("benchmark/a", "A", 10.0, True),
        _row("benchmark/a", "B", 20.0, True),
        _row("benchmark/b", "A", 60.0, False),
        _row("benchmark/b", "B", 30.0, True),
    ]

    profile = analysis.performance_profile_rows(rows, 0.01)
    at_two = {
        row["model"]: row["fraction_within_ratio"]
        for row in profile
        if row["performance_ratio"] == pytest.approx(2.0)
    }

    assert at_two == {"A": pytest.approx(0.5), "B": pytest.approx(1.0)}


def test_reference_cost_gap_uses_best_model_on_same_case():
    rows = [
        {**_row("benchmark/a", "A", 10.0, True), "reference_total_cost": 100.0},
        {**_row("benchmark/a", "B", 20.0, True), "reference_total_cost": 110.0},
    ]

    result = analysis._add_reference_cost_gaps(rows)

    assert result[0]["reference_cost_gap_percent"] == pytest.approx(0.0)
    assert result[1]["reference_cost_gap_percent"] == pytest.approx(10.0)


def test_due_date_effects_are_paired_by_physical_instance_and_model():
    tight = {
        **_row("benchmark/a", "A", 20.0, True),
        "due_date_relative_makespan_offset": 0.0,
        "due_date_condition": "offset:0",
        "reference_total_cost": 120.0,
    }
    loose = {
        **_row("benchmark/a", "A", 15.0, True),
        "due_date_relative_makespan_offset": 0.3,
        "due_date_condition": "offset:0.3",
        "reference_total_cost": 100.0,
    }

    result = analysis.due_date_effects([tight, loose])

    assert len(result) == 1
    assert result[0]["paired_physical_instances"] == 1
    assert result[0]["reference_total_cost_change_mean"] == pytest.approx(-20.0)


def test_complete_analysis_run_writes_checked_tables_and_plots(tmp_path):
    solution_a = tmp_path / "solution_nominal.txt"
    solution_b = tmp_path / "solution_gnn.txt"
    config = {
        "instances": {"generation": {"operations_per_job": [3, 5]}},
        "solve": {
            "solvers": ["gurobi", "gurobi_gnn"],
            "evaluation": {
                "benchmark": {
                    "enabled": True,
                    "num_jobs": [5],
                    "num_machines": [5],
                    "operations_per_job": [3, 5],
                    "instances_per_size": 1,
                    "due_dates": {
                        "relative_makespan_offsets": [0.0]
                    },
                },
                "extrapolation": {"enabled": False},
                "stress": {"enabled": False},
            },
        },
        "training": {
            "gnn": {
                "model_directory": str(tmp_path / "models"),
                "combinations": [{
                    "convolution": "linear",
                    "layers": [1],
                    "hidden_channels": [4],
                }],
            }
        },
        "solvers": {
            "gurobi": {"common": {"TimeLimit": 60, "MIPGap": 0.01}}
        },
        "evaluation": {"replications": 100},
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    physical = "benchmark/i5_k5_o3-5_1"
    schedule_rows = []
    for model, solution in (("nominal", solution_a), ("gnn_linear_layers1_hidden4", solution_b)):
        progress_path = solution.with_name(
            f"{solution.stem}_solver_progress.csv"
        )
        analysis._write_csv(progress_path, [
            {
                "runtime_seconds": 0.5,
                "incumbent_objective": 120.0,
                "best_bound": 80.0,
                "relative_gap": 1.0 / 3.0,
                "node_count": 0,
                "solution_count": 1,
                "event": "incumbent",
                "objective_sense": "minimize",
            },
            {
                "runtime_seconds": 10.0,
                "incumbent_objective": 100.0,
                "best_bound": 100.0,
                "relative_gap": 0.0,
                "node_count": 4,
                "solution_count": 2,
                "event": "final",
                "objective_sense": "minimize",
            },
        ])
        schedule_rows.append({
            **_row(physical, model, 10.0, True),
            "instance_name": f"i5_k5_o3-5_1_benchmark_twk_d0p00",
            "solution_file": str(solution),
            "simulation_seed": 73,
            "simulation_replications": 100,
            "postsolve_evaluation_status": "evaluated",
            "reference_total_cost": 100.0,
            "minimum_mc_ontime_probability": 0.9,
            "solver_progress_file": str(progress_path),
            "solver_progress_points": 2,
        })
    result_path = tmp_path / "result_table.csv"
    analysis._write_csv(result_path, schedule_rows)
    job_path = tmp_path / "job_comparison.csv"
    analysis._write_csv(job_path, [{
        "physical_instance_id": physical,
        "instance_name": schedule_rows[1]["instance_name"],
        "evaluation_tier": "benchmark",
        "model": "gnn_linear_layers1_hidden4",
        "due_date_relative_makespan_offset": 0.0,
        "solution_file": str(solution_b),
        "repair_buffer_error": -0.5,
        "mc_ontime_probability": 0.9,
        "mc_mean_completion_delay": 2.0,
        "mc_meets_service_threshold": True,
    }])
    manifest_path = tmp_path / "solve_manifest.json"
    manifest_path.write_text(json.dumps({"runs": [
        {"solution_path": str(solution_a)},
        {"solution_path": str(solution_b)},
    ]}), encoding="utf-8")
    output = tmp_path / "analysis"

    result = analysis.run_analysis(
        config_path=config_path,
        result_table=result_path,
        job_table=job_path,
        manifest=manifest_path,
        instances_root=tmp_path / "instances",
        output_directory=output,
        strict=True,
    )

    checks = analysis._read_csv(result["completeness_checks"])
    assert all(row["status"] == "passed" for row in checks)
    assert (output / "runtime_ecdf.pdf").exists()
    assert (output / "performance_profile.pdf").exists()
    assert (output / "solver_gap_over_time.pdf").exists()
    assert (output / "solver_incumbent_over_time.pdf").exists()
    assert (output / "solver_incumbent_improvement_over_time.pdf").exists()
    assert len(analysis._read_csv(result["solver_progress"])) == 4
    progress_summary = analysis._read_csv(result["solver_progress_summary"])
    assert {row["model"] for row in progress_summary} == {
        "nominal", "gnn_linear_layers1_hidden4",
    }
    assert result["quality_summary"].exists()
