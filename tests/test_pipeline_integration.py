import importlib
import json
from pathlib import Path
import os
from types import SimpleNamespace

import pytest

import main
from helper import pipeline_solver
from tests.test_local_buffer_pipeline import schedule

evaluation = importlib.import_module('06_Evaluation.evaluate_solutions')


@pytest.mark.parametrize('solve', [False, True])
def test_cli_workflow_reaches_evaluator_and_scopes_current_files(tmp_path, monkeypatch, solve):
    config = json.loads(main.CONFIG_PATH.read_text())
    config['workflow'] = {key: False for key in config['workflow']}
    config_path = tmp_path / 'config.json'
    config_path.write_text(json.dumps(config))
    monkeypatch.setattr(main, '_configured_instance_splits', lambda *a: {})
    monkeypatch.setattr(main, '_solve', lambda *a: [{'solution_path': 'current.txt'}, {'solution_path': None}])
    captured = []
    monkeypatch.setattr(evaluation, 'run_evaluation', lambda **kwargs: captured.append(kwargs))
    phases = ['solve', 'evaluate'] if solve else ['evaluate']
    main.main(['--config', str(config_path), '--workflow', *phases])
    assert len(captured) == 1
    assert captured[0]['config']['workflow']['evaluate'] is True
    assert captured[0]['solutions'] == (['current.txt'] if solve else None)


def test_train_only_workflow_does_not_require_instance_pickles(
    tmp_path, monkeypatch
):
    config = json.loads(main.CONFIG_PATH.read_text())
    config_path = tmp_path / 'config.json'
    config_path.write_text(json.dumps(config))
    monkeypatch.setattr(
        main,
        '_configured_instance_splits',
        lambda *_args, **_kwargs: pytest.fail(
            'Train-only workflow must not validate instance pickles.'
        ),
    )
    calls = []
    monkeypatch.setattr(main, '_train', lambda received: calls.append(received))

    main.main([
        '--config', str(config_path), '--workflow', 'train-gnn'
    ])

    assert len(calls) == 1


def test_benchmark_only_solve_does_not_require_instance_pickles(
    tmp_path, monkeypatch
):
    config = json.loads(main.CONFIG_PATH.read_text())
    config['workflow'] = {key: False for key in config['workflow']}
    config['workflow']['solve'] = True
    tiers = config['solve']['evaluation']
    tiers['in_distribution']['enabled'] = False
    tiers['benchmark']['enabled'] = True
    tiers['extrapolation']['enabled'] = False
    tiers['stress']['enabled'] = False
    config_path = tmp_path / 'config.json'
    config_path.write_text(json.dumps(config))
    monkeypatch.setattr(
        main,
        '_configured_instance_splits',
        lambda *_args, **_kwargs: pytest.fail(
            'Benchmark-only solving must not validate regular instance pickles.'
        ),
    )
    received_splits = []
    monkeypatch.setattr(
        main,
        '_solve',
        lambda _config, splits: received_splits.append(splits) or [],
    )

    main.main(['--config', str(config_path)])

    assert received_splits == [None]


@pytest.mark.parametrize('solver', ['gurobi', 'gurobi_gnn', 'gurobi_nonlinear'])
def test_solver_runs_in_calling_process_and_releases_model(monkeypatch, solver):
    from helper import start_solve_ins
    caller_pid = os.getpid()
    calls, disposed = [], []
    def solve(**parameters):
        calls.append((os.getpid(), parameters['solver']))
        values = {key: None for key in ('status', 'solution_count', 'objective',
            'best_bound', 'mip_gap', 'runtime', 'model_build_runtime',
            'optimizer_wall_runtime', 'build_plus_optimizer_runtime',
            'branch_and_bound_nodes', 'root_node_bound',
            'time_to_first_incumbent_seconds', 'first_incumbent_objective',
            'time_to_best_incumbent_seconds', 'best_incumbent_objective',
            'linear_matrix_nonzeros', 'solution_path', 'warm_start')}
        values['model'] = SimpleNamespace(dispose=lambda: disposed.append(True))
        return values
    monkeypatch.setattr(start_solve_ins, 'solve_instances_with_solver', solve)
    result = pipeline_solver.run_pipeline_solver({'solver': solver, 'instance_name': 'example'})
    assert calls == [(caller_pid, solver)]
    assert result['solver'] == solver
    assert disposed == [True]


def test_solver_errors_are_not_hidden_by_process_wrapper(monkeypatch):
    from helper import start_solve_ins
    def fail(**parameters):
        raise RuntimeError('solver error')
    monkeypatch.setattr(start_solve_ins, 'solve_instances_with_solver', fail)
    with pytest.raises(RuntimeError, match='solver error'):
        pipeline_solver.run_pipeline_solver({'solver': 'gurobi_nonlinear', 'instance_name': 'example'})


def test_pipeline_rejects_old_feature_weights_before_solving(tmp_path, monkeypatch):
    config = json.loads(main.CONFIG_PATH.read_text())
    monkeypatch.setattr(main._architectures, 'architecture_model_dir', lambda *a, **k: tmp_path)
    monkeypatch.setattr(main._architectures, 'architecture_stem', lambda *a, **k: 'old')
    (tmp_path / 'old.pt').write_bytes(b'')
    (tmp_path / 'old_meta.json').write_text(json.dumps({'input_size': 7}))
    with pytest.raises(ValueError, match='four-feature pipeline'):
        main._trained_models(config)


@pytest.mark.parametrize('solver', ['gurobi_gnn', 'gurobi'])
def test_reference_cost_is_unscaled_and_threshold_only_affects_assessment(tmp_path, monkeypatch, solver):
    fixed = schedule()
    parsed = {key: None for key in ('objective', 'makespan', 'tardiness_cost',
        'total_tardiness', 'total_cost', 'best_bound', 'mip_gap', 'runtime_seconds',
        'model_build_seconds', 'optimizer_wall_seconds')}
    parsed.update(instance_name='example', solution_path=tmp_path/'solution.txt',
        solver=solver, model_name='gnn', formulation='gnn_local', status='OPTIMAL',
        solution_count=1, operations={0: {}, 1: {}}, processing_cost=10., operating_cost=20.,
        service_level=.8, service_violation_cost_per_time=3., repair_buffer_label_method='local',
        jobs={0: {'nominal_completion': 28., 'internal_repair_buffer': 2.,
                  'protected_completion': 38., 'robust_slack': 2., 'optimization_tardiness': 0.}})
    if solver == 'gurobi':
        parsed['service_level'] = None
    monkeypatch.setattr(evaluation, 'parse_solution', lambda *a: parsed)
    monkeypatch.setattr(evaluation, '_load_instance', lambda *a:
        (SimpleNamespace(jobs=fixed.jobs, due_dates=fixed.due_dates), tmp_path/'example.pkl'))
    monkeypatch.setattr(evaluation, '_fixed_schedule', lambda *a: fixed)
    monkeypatch.setattr(evaluation, '_reference_expected_repair_buffers', lambda *a: {0: 15.})
    summary, jobs = evaluation.evaluate_solution(tmp_path/'solution.txt', instances_root=tmp_path,
        replications=100, base_seed=42, confidence=.95, simulation_config=None,
        service_level_threshold=.8)
    assert summary['reference_due_date_violation'] == pytest.approx(3.)
    assert summary['reference_total_cost'] == pytest.approx(39.)
    assert summary['maximum_repair_buffer_underestimation'] == 13.
    assert jobs[0]['reference_due_date_violation'] == pytest.approx(3.)


def test_nominal_solution_is_included_in_comparison(tmp_path):
    path = tmp_path / 'solution_example_gurobi.txt'
    path.write_text(
        'Formulation: nominal_fjsp_with_soft_due_date_violation_v2\n'
        'Status: TIME_LIMIT\n'
        'Solution count: 0\n'
        'Branch-and-bound nodes: 123\n'
        'Root-node bound: 17.5\n'
        'Time to first incumbent [s]: \n'
        'linear_matrix_nonzeros: 456\n'
        'variables: 78\n'
        'binary: 34\n'
        'linear_constraints: 90\n'
    )
    assert evaluation._solution_paths([path], tmp_path) == [path]
    parsed = evaluation.parse_solution(path)
    assert parsed['solver'] == 'gurobi'
    assert parsed['instance_name'] == 'example'
    assert parsed['branch_and_bound_nodes'] == 123.
    assert parsed['root_node_bound'] == 17.5
    assert parsed['time_to_first_incumbent_seconds'] is None
    assert parsed['number_of_linear_matrix_nonzeros'] == 456
    assert parsed['number_of_variables'] == 78
    assert parsed['number_of_binary_variables'] == 34
    behavior = evaluation._solver_behavior_fields(parsed)
    assert behavior['no_feasible_solution_found'] is True
    assert behavior['time_limit_with_incumbent'] is False


def test_empty_current_run_does_not_evaluate_previous_files(tmp_path):
    with pytest.raises(ValueError, match='current solve run'):
        evaluation.run_evaluation(config={}, solutions=[], solutions_root=tmp_path)


def test_invalid_incumbent_is_reported_without_invented_reference_costs(tmp_path, monkeypatch):
    from helper.local_buffer import InvalidScheduleError
    parsed = {key: None for key in ('objective', 'makespan', 'tardiness_cost',
        'total_tardiness', 'total_cost', 'best_bound', 'mip_gap', 'runtime_seconds',
        'model_build_seconds', 'optimizer_wall_seconds', 'processing_cost', 'operating_cost')}
    parsed.update(instance_name='example', solution_path=tmp_path/'solution.txt',
        solver='gurobi_gnn', model_name='linear_8', formulation='gnn_local', status='TIME_LIMIT',
        solution_count=1, operations={0: {}}, jobs={0: {}})
    monkeypatch.setattr(evaluation, 'parse_solution', lambda *a: parsed)
    monkeypatch.setattr(evaluation, '_load_instance', lambda *a:
        (SimpleNamespace(jobs={0: (0,)}), tmp_path/'example.pkl'))
    monkeypatch.setattr(evaluation, '_fixed_schedule', lambda *a: schedule())
    def invalid(*args):
        raise InvalidScheduleError('Nominal machine operations overlap.')
    monkeypatch.setattr(evaluation, '_reference_expected_repair_buffers', invalid)
    monkeypatch.setattr(evaluation, 'simulate_fixed_schedule', lambda *a, **k: pytest.fail('Invalid plan must not be simulated.'))
    row, jobs = evaluation.evaluate_solution(tmp_path/'solution.txt', instances_root=tmp_path,
        replications=100, base_seed=42, confidence=.95, simulation_config=None)
    assert row['postsolve_evaluation_status'] == 'not_evaluated_invalid_schedule'
    assert 'overlap' in row['evaluation_error']
    assert row['reference_total_cost'] is None
    assert row['repair_buffer_mae'] is None
    assert row['simulation_replications'] == 0
    assert jobs == []


@pytest.mark.parametrize('enabled', [True, False, None])
@pytest.mark.parametrize('nominal_first', [True, False])
def test_config_controls_start_and_solver_order(tmp_path, monkeypatch, enabled, nominal_first):
    config = {'evaluation': {'output_directory': str(tmp_path)},
              'solvers': {'gurobi': {}}}
    if enabled is not None:
        config['solve'] = {'nominal_warm_start': enabled}
    calls = []
    def run(parameters):
        calls.append(parameters)
        return {'solver': parameters['solver'], 'solution_path': None,
                'warm_start': {'instance': parameters['instance_name']}
                if parameters['solver'] == 'gurobi' else None}
    monkeypatch.setattr(main, 'run_pipeline_solver', run)
    requested = (['gurobi', 'gurobi_nonlinear', 'gurobi_gnn'] if nominal_first
                 else ['gurobi_nonlinear', 'gurobi_gnn', 'gurobi'])
    main._solve_plan(config, [{'instance_name': 'a'}, {'instance_name': 'b'}],
        requested,
        [{'convolution': 'job'}, {'convolution': 'sage'}], {}, {})
    assert len(calls) == 8
    if not enabled:
        expected = [solver for solver in requested for _ in range(2 if solver == 'gurobi_gnn' else 1)]
        assert [r['solver'] for r in calls] == expected * 2
        assert all('warm_start' not in r for r in calls)
        return
    for offset, instance in [(0, 'a'), (4, 'b')]:
        assert calls[offset]['solver'] == 'gurobi'
        assert 'warm_start' not in calls[offset]
        for call in calls[offset+1:offset+4]:
            assert call['warm_start'] == {'instance': instance}


@pytest.mark.parametrize('setting, solvers, message', [
    (True, ['gurobi_gnn'], 'requires gurobi'),
    ('false', ['gurobi', 'gurobi_gnn'], 'true or false'),
])
def test_invalid_warm_start_configuration_fails_before_solving(setting, solvers, message):
    with pytest.raises(ValueError, match=message):
        main._solve_plan({'solve': {'nominal_warm_start': setting}}, [], solvers, [], {}, {})


def test_array_index_selects_one_case_and_uses_own_manifest(
    tmp_path, monkeypatch
):
    config = {
        'solve': {
            'solvers': ['gurobi'],
            'evaluation': {
                'in_distribution': {'enabled': False},
                'benchmark': {'enabled': True},
                'extrapolation': {'enabled': False},
                'stress': {'enabled': False},
            },
        },
        'solvers': {'gurobi': {'common': {}}},
        'objective': {},
        'constraint': {'type': 'weibull', 'weibull': {'reliability_graph': {}}},
        'evaluation': {
            'output_directory': str(tmp_path),
            'solutions_directory': str(tmp_path / 'solutions'),
        },
    }
    cases = [
        {'instance_name': 'benchmark_a'},
        {'instance_name': 'benchmark_b'},
    ]
    monkeypatch.setattr(
        main,
        '_generated_tier_plan',
        lambda _config, tier: cases if tier == 'benchmark' else [],
    )
    received = []
    monkeypatch.setattr(
        main,
        '_solve_plan',
        lambda *args, **kwargs: received.append((args, kwargs)),
    )

    main._solve(config, None, solve_plan_index=1)

    assert received[0][0][1] == [cases[1]]
    assert received[0][1]['manifest_path'] == (
        tmp_path / 'solve_manifests' / 'solve_manifest_task_001.json'
    )


def test_array_index_outside_plan_is_rejected(tmp_path, monkeypatch):
    config = {
        'solve': {
            'solvers': ['gurobi'],
            'evaluation': {
                'in_distribution': {'enabled': False},
                'benchmark': {'enabled': True},
                'extrapolation': {'enabled': False},
                'stress': {'enabled': False},
            },
        },
        'solvers': {'gurobi': {'common': {}}},
        'objective': {},
        'constraint': {'type': 'weibull', 'weibull': {'reliability_graph': {}}},
        'evaluation': {
            'output_directory': str(tmp_path),
            'solutions_directory': str(tmp_path / 'solutions'),
        },
    }
    monkeypatch.setattr(
        main,
        '_generated_tier_plan',
        lambda _config, tier: (
            [{'instance_name': 'benchmark_a'}]
            if tier == 'benchmark' else []
        ),
    )

    with pytest.raises(ValueError, match='outside the configured range'):
        main._solve(config, None, solve_plan_index=1)
