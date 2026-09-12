"""Run every pipeline solver directly in the calling Python process."""
from __future__ import annotations


def run_pipeline_solver(parameters):
    """Solve synchronously, return a summary and release the Gurobi model."""
    from helper.start_solve_ins import solve_instances_with_solver
    result = solve_instances_with_solver(**parameters)
    try:
        keys = ('status', 'solution_count', 'objective', 'best_bound', 'mip_gap',
                'branch_and_bound_nodes', 'root_node_bound',
                'time_to_first_incumbent_seconds', 'first_incumbent_objective',
                'time_to_best_incumbent_seconds', 'best_incumbent_objective',
                'linear_matrix_nonzeros',
                'runtime', 'model_build_runtime', 'optimizer_wall_runtime',
                'build_plus_optimizer_runtime', 'solution_path',
                'solver_progress_path', 'warm_start')
        return {**{key: result.get(key) for key in keys},
                'solver': parameters['solver'], 'instance_name': parameters['instance_name'],
                'model_path': parameters.get('model_path')}
    finally:
        result['model'].dispose()
