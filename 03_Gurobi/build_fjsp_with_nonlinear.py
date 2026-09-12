"""Scenario-free nonlinear FJSP with midpoint Weibull probabilities."""

from __future__ import annotations

import importlib
import math

import gurobipy as gp

from helper.gurobi_solution_writer import write_comparable_solution
from helper.economic_objective import (
    add_economic_cost_objective,
    add_due_date_tardiness_constraints,
)
from helper.sequence_setup import (
    normalize_reliability_graph_config,
    reliability_graph_config_dict,
)
from helper.stochastic_fjsp import (
    ensure_stochastic_parameters,
    gauss_legendre_rule,
    stochastic_parameters,
)
from helper.surrogate_constraint import (
    CONSTRAINT_WEIBULL,
    validate_constraint_type,
)


_base_fjsp = importlib.import_module("03_Gurobi.build_fjsp")
STATUS_NAMES = _base_fjsp.STATUS_NAMES


def _add_midpoint_state(model, variables, instance):
    horizon = float(variables["H"])
    starts, midpoints, nominal_durations = {}, {}, {}
    for operation in variables["real_operations"]:
        duration = gp.quicksum(
            float(instance.processing_times[operation, machine])
            * variables["Y"][operation, machine]
            for machine in instance.eligible_machines[operation]
        )
        nominal_durations[operation] = duration
        starts[operation] = model.addVar(
            lb=0.0, ub=horizon, name=f"S_nominal[{operation}]"
        )
        midpoints[operation] = model.addVar(
            lb=0.0, ub=horizon, name=f"t_midpoint[{operation}]"
        )
        model.addConstr(
            variables["C"][operation] == starts[operation] + duration,
            name=f"nominal_start_def[{operation}]",
        )
        model.addConstr(
            midpoints[operation] == starts[operation] + 0.5 * duration,
            name=f"midpoint_def[{operation}]",
        )
    variables.update({"S": starts, "T": midpoints, "D": nominal_durations})


def _add_pd_quadrature(model, variables, instance, graph_cfg):
    nodes, weights = gauss_legendre_rule(graph_cfg.quadrature_points)
    probability, selected_probability = {}, {}
    for operation, machine in variables["Y_index"]:
        t = variables["T"][operation]
        alpha = float(variables["weibull_alpha"][machine])
        beta = float(variables["weibull_beta"][machine])
        repair_rate = float(variables["repair_rate"][machine])
        if math.isclose(beta, 2.0, rel_tol=0.0, abs_tol=1e-12):
            beta_degree = 2
        elif math.isclose(beta, 3.0, rel_tol=0.0, abs_tol=1e-12):
            beta_degree = 3
        else:
            raise ValueError(
                "The nonlinear reference model supports only the configured "
                "integer Weibull beta values 2 (new) and 3 (old); received "
                f"beta={beta:g} for machine {machine}."
            )

        horizon = float(variables["H"])
        terms = []
        for node_index, (node, weight) in enumerate(zip(nodes, weights)):
            fraction = 0.5 * (node + 1.0)
            scaled = fraction * t / alpha
            scaled_squared = scaled * scaled
            if beta_degree == 3:
                weibull_power = scaled_squared * scaled
                density_power = scaled_squared
            else:
                weibull_power = scaled_squared
                density_power = scaled
            exponent_argument = (
                -weibull_power - repair_rate * (1.0 - fraction) * t
            )
            integrand = model.addVar(
                lb=0.0,
                ub=(fraction * horizon / alpha) ** (beta_degree - 1),
                name=f"pd_integrand[{operation},{machine},{node_index}]",
            )
            model.addGenConstrNL(
                integrand,
                density_power * gp.nlfunc.exp(exponent_argument),
                name=f"pd_integrand_def[{operation},{machine},{node_index}]",
            )
            terms.append(
                weight * beta_degree / alpha * integrand
            )
        p = model.addVar(lb=0.0, ub=1.0, name=f"Pd[{operation},{machine}]")
        model.addGenConstrNL(
            p,
            0.5 * t * sum(terms),
            name=f"pd_weibull_quadrature[{operation},{machine}]",
        )
        w = model.addVar(
            lb=0.0, ub=1.0, name=f"W_pd_selected[{operation},{machine}]"
        )
        y = variables["Y"][operation, machine]
        model.addConstr(w <= p, name=f"pd_product_ub_p[{operation},{machine}]")
        model.addConstr(w <= y, name=f"pd_product_ub_y[{operation},{machine}]")
        model.addConstr(
            w >= p - (1.0 - y),
            name=f"pd_product_lb[{operation},{machine}]",
        )
        probability[operation, machine] = p
        selected_probability[operation, machine] = w
    variables.update({
        "Pd": probability,
        "pi_fail": selected_probability,
        "failure_probability_times_assignment": selected_probability,
        "quadrature_nodes": nodes,
        "quadrature_weights": weights,
    })


def _service_operations(instance, job, scope):
    return list(instance.jobs[job]) if scope == "job" else list(instance.real_operations)


def _add_expected_repair_buffers(
    model,
    variables,
    instance,
    graph_cfg,
):
    expected_delays = {}
    for job in instance.jobs:
        expected_disruption = gp.quicksum(
            variables["pi_fail"][operation, machine]
            / variables["repair_rate"][machine]
            for operation in _service_operations(instance, job, graph_cfg.service_scope)
            for machine in instance.eligible_machines[operation]
        )
        expected_delays[job] = expected_disruption
    add_due_date_tardiness_constraints(
        model,
        variables,
        instance,
        expected_delays,
    )
    variables.update({
        "service_scope": graph_cfg.service_scope,
        "due_dates": dict(instance.due_dates),
        "job_expected_delays": expected_delays,
        "job_repair_buffer_label_method": "weibull_expected_repair_buffer_v1",
    })


def build_fjsp(
    fjsp,
    instance,
    constraint_type=CONSTRAINT_WEIBULL,
    reliability_graph_config=None,
    service_probability_band=None,
    facility_cost_per_time=1.0,
    service_violation_cost_per_time=1.0,
    tardiness_cost_per_time=None,
):
    """Build the nonlinear stochastic reference formulation."""
    validate_constraint_type(constraint_type)
    graph_cfg = normalize_reliability_graph_config(reliability_graph_config)
    if graph_cfg.service_scope != "job":
        raise ValueError("The thesis formulation sums repair buffers per job.")
    if service_probability_band is not None:
        raise ValueError(
            "service_probability_band was removed; alpha is evaluated only "
            "by post-optimization Monte Carlo simulation."
        )
    ensure_stochastic_parameters(instance)
    parameters = stochastic_parameters(instance)
    model, variables = _base_fjsp.build_fjsp(
        fjsp,
        instance,
        include_makespan=False,
        horizon_upper_bound=None,
        enforce_due_dates=False,
        economic_objective=False,
    )
    model.Params.NonConvex = 2
    for operation in variables["real_operations"]:
        variables["C"][operation].ub = float(variables["H"])
    variables.update({
        "machine_modernity": parameters["theta"],
        "machine_speed": parameters["speed"],
        "weibull_alpha": parameters["alpha"],
        "weibull_beta": parameters["beta"],
        "repair_rate": parameters["repair_rate"],
        "repair_durations": {
            (operation, machine): parameters["repair_duration"][machine]
            for operation, machine in variables["Y_index"]
        },
        "reliability_graph_config": reliability_graph_config_dict(graph_cfg),
    })
    _add_midpoint_state(model, variables, instance)
    _add_pd_quadrature(model, variables, instance, graph_cfg)
    operation_probability, delta = {}, {}
    for operation in variables["real_operations"]:
        operation_probability[operation] = gp.quicksum(
            variables["pi_fail"][operation, machine]
            for machine in instance.eligible_machines[operation]
        )
        delta[operation] = gp.quicksum(
            variables["pi_fail"][operation, machine]
            / parameters["repair_rate"][machine]
            for machine in instance.eligible_machines[operation]
        )
    variables.update({
        "operation_pi_fail": operation_probability,
        "Delta": delta,
        "total_failure_delay": gp.quicksum(delta.values()),
    })
    _add_expected_repair_buffers(
        model,
        variables,
        instance,
        graph_cfg,
    )
    add_economic_cost_objective(
        model,
        variables,
        instance,
        facility_cost_per_time=facility_cost_per_time,
        service_violation_cost_per_time=service_violation_cost_per_time,
        tardiness_cost_per_time=tardiness_cost_per_time,
    )
    variables.update({
        "constraint_type": constraint_type,
        "formulation": "nonlinear_unscaled_local_repair_buffer_v12",
    })
    model.update()
    return model, variables


def write_solution_file(model, variables, instance, filename="solution.txt"):
    return write_comparable_solution(model, variables, filename, instance=instance)
