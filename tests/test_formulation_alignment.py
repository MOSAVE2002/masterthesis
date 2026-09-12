"""Regression checks against the supplied thesis equations."""
from dataclasses import replace
import importlib
import inspect
import json
from types import SimpleNamespace

import gurobipy as gp
import pytest

import main
from helper.local_buffer import operation_expectation, expected_job_buffers
from tests.test_economic_objective import _instance
from tests.test_local_buffer_pipeline import schedule

nonlinear = importlib.import_module('03_Gurobi.build_fjsp_with_nonlinear')
embedding = importlib.import_module('03_Gurobi.build_fjsp_with_gnn')
evaluation = importlib.import_module('06_Evaluation.evaluate_solutions')
simulation = importlib.import_module('05_Simulation.preempt_resume')
training = importlib.import_module('04_GraphNeuralNetworks.models.model_training_FJSP_GNN')


@pytest.mark.parametrize('due_date', [4., 10., 20.])
def test_nonlinear_buffer_and_cost_match_unscaled_equations(due_date):
    completion = 10.
    instance = _instance(due_date=due_date)
    with gp.Model() as model:
        model.Params.OutputFlag = 0
        model.Params.Presolve = 0
        model.Params.NumericFocus = 3
        model.Params.TimeLimit = 5
        _, values = nonlinear.build_fjsp(model, instance, tardiness_cost_per_time=3.)
        values['Y'][1, 0].LB = 1.
        values['C'][1].LB = values['C'][1].UB = completion
        model.optimize()
        assert model.SolCount
        buffer = operation_expectation(completion - 5., 30., 2., .5, order=256)
        assert values['job_expected_repair_buffers'][1].getValue() == pytest.approx(buffer, abs=1e-5)
        violation = max(0., completion + buffer - due_date)
        assert values['job_due_date_violation'][1].X == pytest.approx(violation, abs=1e-5)
        assert model.ObjVal == pytest.approx(10. + completion + 3.*violation, abs=1e-4)
        assert 'service_level' not in values
        assert 'service_buffer_scale' not in values


def test_service_grade_is_absent_from_both_model_interfaces():
    for module in (nonlinear, embedding):
        assert 'service_level' not in inspect.signature(module.build_fjsp).parameters


@pytest.mark.parametrize('gap, expected_completion', [(0., 30.), (15., 35.)])
def test_execution_propagates_delay_and_absorbs_it_in_existing_gap(monkeypatch, gap, expected_completion):
    # First operation 0..10 is interrupted at 5 for 10 ZE; the second
    # operation runs on another machine and must wait for its job predecessor.
    fixed = replace(schedule(), selected_machines={0: 0, 1: 1},
        planned_starts={0: 0., 1: 10.+gap}, processing_times={0: 10., 1: 10.},
        machine_edges=(), weibull_scale={0: 10., 1: 1000.}, due_dates={0: 25.})
    monkeypatch.setattr(simulation, '_draw_machine_downtime',
        lambda rng, alpha, beta, rate: (5., 10.) if alpha == 10. else (1000., 1.))
    assert evaluation.simulate_fixed_schedule is simulation.simulate_fixed_schedule
    result = evaluation.simulate_fixed_schedule(fixed, replications=2, seed=42)
    assert result.job_mean_completion_times == (expected_completion,)
    assert result.job_ontime_probabilities == (0.,)


def test_local_label_stays_distinct_from_propagated_execution(monkeypatch):
    fixed = schedule()
    labels = expected_job_buffers(fixed)
    monkeypatch.setattr(simulation, '_draw_machine_downtime', lambda *a: (0., 100.))
    mc = evaluation.simulate_fixed_schedule(fixed, replications=2, seed=42)
    assert expected_job_buffers(fixed) == labels
    assert mc.job_mean_completion_delays[0] > labels[0][0]


def test_pipeline_passes_selected_config_to_training(monkeypatch):
    config = {'training': {'gnn': {'seed': 73, 'model_directory': 'isolated'}}}
    captured = []
    monkeypatch.setattr(training, 'train_from_config', lambda **kw: captured.append(kw))
    main._train(config)
    assert captured == [{'seed': 73, 'config': config}]


def test_real_solution_round_trip_has_unscaled_reference_cost(tmp_path, monkeypatch):
    instance = _instance(due_date=4.)
    with gp.Model() as model:
        model.Params.OutputFlag = 0
        model.Params.Presolve = 0
        model.Params.TimeLimit = 5
        _, values = nonlinear.build_fjsp(model, instance, tardiness_cost_per_time=3.)
        model.optimize()
        path = tmp_path / 'solution_example_gurobi_nonlinear.txt'
        nonlinear.write_solution_file(model, values, instance, path)
        monkeypatch.setattr(evaluation, '_load_instance', lambda *a: (instance, tmp_path/'example.pkl'))
        first, jobs = evaluation.evaluate_solution(path, instances_root=tmp_path,
            replications=20, base_seed=42, confidence=.95, simulation_config=None,
            service_level_threshold=.8)
        second, _ = evaluation.evaluate_solution(path, instances_root=tmp_path,
            replications=20, base_seed=42, confidence=.95, simulation_config=None,
            service_level_threshold=.99)
        assert first['reference_total_cost'] == pytest.approx(model.ObjVal, abs=1e-4)
        assert first['reference_total_cost'] == second['reference_total_cost']
        assert first['minimum_mc_ontime_probability'] == second['minimum_mc_ontime_probability']
        assert first['simulation_model'] == 'preempt_resume'
        assert 'Service level alpha:' not in path.read_text()
