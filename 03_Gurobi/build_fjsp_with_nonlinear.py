"""Scenario-free nonlinear FJSP with midpoint Weibull probabilities."""

from __future__ import annotations

import importlib

import gurobipy as gp
from gurobipy import GRB

from helper.gurobi_solution_writer import write_comparable_solution
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
        alpha = variables["weibull_alpha"][machine]
        beta = variables["weibull_beta"][machine]
        repair_rate = variables["repair_rate"][machine]
        terms = []
        for node, weight in zip(nodes, weights):
            x = 0.5 * (node + 1.0) * t
            scaled = x / alpha
            exponent = -(scaled ** beta) - repair_rate * (t - x)
            terms.append(
                weight * beta / alpha * (scaled ** (beta - 1.0))
                * gp.nlfunc.exp(exponent)
            )
        p = model.addVar(lb=0.0, ub=1.0, name=f"Pd[{operation},{machine}]")
        model.addGenConstrNL(
            p,
            0.5 * t * sum(terms[1:], terms[0]),
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


def _add_service_constraints(
    model,
    variables,
    instance,
    graph_cfg,
    *,
    enforce_service_level=True,
):
    buffers, constraints, resolved_levels = {}, {}, {}
    probabilities, clipping_excesses, slacks, expected_delays = {}, {}, {}, {}
    for job in instance.jobs:
        alpha_target = float(
            getattr(instance, "service_levels", {}).get(job, graph_cfg.service_level)
        )
        resolved_levels[job] = alpha_target
        expected_disruption = gp.quicksum(
            variables["pi_fail"][operation, machine]
            / variables["repair_rate"][machine]
            for operation in _service_operations(instance, job, graph_cfg.service_scope)
            for machine in instance.eligible_machines[operation]
        )
        due_date = float(instance.due_dates[job])
        slack = model.addVar(
            lb=1e-6,
            ub=due_date,
            name=f"nominal_due_date_slack[{job}]",
        )
        disruption_upper_bound = sum(
            1.0 / float(variables["repair_rate"][machine])
            for operation in _service_operations(
                instance, job, graph_cfg.service_scope
            )
            for machine in instance.eligible_machines[operation]
        )
        probability = model.addVar(
            lb=0.0,
            ub=1.0,
            name=f"job_ontime_probability_lb[{job}]",
        )
        clipping_excess = model.addVar(
            lb=0.0,
            ub=disruption_upper_bound,
            name=f"job_probability_clipping_excess[{job}]",
        )
        model.addConstr(
            slack == due_date - variables["C"][instance.job_end_operations[job]],
            name=f"nominal_due_date_slack_def[{job}]",
        )
        model.addQConstr(
            (1.0 - probability) * slack + clipping_excess
            == expected_disruption,
            name=f"markov_job_probability_clipped_def[{job}]",
        )
        model.addQConstr(
            probability * clipping_excess == 0.0,
            name=f"markov_job_probability_complementarity[{job}]",
        )
        buffers[job] = expected_disruption / (1.0 - alpha_target)
        if enforce_service_level:
            constraints[job] = model.addConstr(
                probability >= alpha_target,
                name=f"alpha_service_level[{job}]",
            )
        probabilities[job] = probability
        clipping_excesses[job] = clipping_excess
        slacks[job] = slack
        expected_delays[job] = expected_disruption
    variables.update({
        "service_buffers": buffers,
        "service_constraints": constraints,
        "service_scope": graph_cfg.service_scope,
        "due_dates": dict(instance.due_dates),
        "service_levels": resolved_levels,
        "job_ontime_probabilities": probabilities,
        "job_probability_clipping_excesses": clipping_excesses,
        "job_probability_slacks": slacks,
        "job_expected_delays": expected_delays,
        "job_probability_label_method": (
            "clipped_markov_expected_delay_lower_bound_v2"
        ),
    })


def _add_min_probability_band(model, variables, probability_band):
    if probability_band is None:
        return
    lower, upper = map(float, probability_band)
    if not 0.0 <= lower <= upper <= 1.0:
        raise ValueError(
            "service_probability_band must satisfy "
            "0 <= lower <= upper <= 1."
        )
    probabilities = list(variables["job_ontime_probabilities"].values())
    minimum = model.addVar(
        lb=0.0,
        ub=1.0,
        name="minimum_job_ontime_probability_lb",
    )
    model.addGenConstrMin(
        minimum,
        probabilities,
        name="minimum_job_ontime_probability_def",
    )
    lower_constraint = model.addConstr(
        minimum >= lower,
        name="candidate_probability_band_lb",
    )
    upper_constraint = model.addConstr(
        minimum <= upper,
        name="candidate_probability_band_ub",
    )
    variables.update({
        "minimum_job_ontime_probability": minimum,
        "candidate_probability_band": (lower, upper),
        "candidate_probability_band_constraints": (
            lower_constraint,
            upper_constraint,
        ),
    })


def _set_cost_objective(model, variables, instance):
    assignment_cost = gp.quicksum(
        variables["machine_cost"][machine]
        * float(instance.processing_times[operation, machine])
        * variables["Y"][operation, machine]
        for operation, machine in variables["Y_index"]
    )
    model.setObjective(assignment_cost, GRB.MINIMIZE)
    variables["assignment_cost"] = assignment_cost
    variables["objective_definition"] = "machine_assignment_cost"


def build_fjsp(
    fjsp,
    instance,
    constraint_type=CONSTRAINT_WEIBULL,
    reliability_graph_config=None,
    service_probability_band=None,
):
    """Build the nonlinear stochastic reference formulation."""
    validate_constraint_type(constraint_type)
    graph_cfg = normalize_reliability_graph_config(reliability_graph_config)
    ensure_stochastic_parameters(instance)
    parameters = stochastic_parameters(instance)
    model, variables = _base_fjsp.build_fjsp(
        fjsp,
        instance,
        include_makespan=False,
    )
    model.Params.NonConvex = 2
    for operation in variables["real_operations"]:
        variables["C"][operation].ub = float(variables["H"])
    variables.update({
        "machine_modernity": parameters["theta"],
        "machine_speed": parameters["speed"],
        "machine_cost": parameters["cost"],
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
    _add_service_constraints(
        model,
        variables,
        instance,
        graph_cfg,
        enforce_service_level=service_probability_band is None,
    )
    _add_min_probability_band(
        model, variables, service_probability_band
    )
    _set_cost_objective(model, variables, instance)
    variables.update({
        "constraint_type": constraint_type,
        "formulation": (
            "nonlinear_per_job_ontime_probability_band_v3"
            if service_probability_band is not None
            else "nonlinear_per_job_ontime_probability_lb_v3"
        ),
    })
    model.update()
    return model, variables


def write_solution_file(model, variables, instance, filename="solution.txt"):
    return write_comparable_solution(model, variables, filename, instance=instance)
