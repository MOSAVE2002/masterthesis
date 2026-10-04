"""Build the nonlinear Weibull reference formulation for the stochastic FJSP.

The model evaluates disruption probabilities at operation midpoints with
Gauss-Legendre quadrature inside Gurobi. Selected-machine probabilities are
converted into expected repair buffers per job and added to the shared soft
due-date and economic-cost formulation.
"""

import importlib
import math

import gurobipy as gp

from helper.economic_objective import (
    add_economic_cost_objective,
    add_due_date_tardiness_constraints,
)
from helper.sequence_setup import (
    RELIABILITY_QUADRATURE_POINTS,
)
from helper.stochastic_fjsp import (
    gauss_legendre_rule,
    stochastic_parameters,
)
base_fjsp = importlib.import_module("03_Gurobi.build_fjsp")


def add_midpoint_state(model, variables, instance):
    """Add assignment-dependent durations, starts and operation midpoints.

    Args:
        model: Gurobi model receiving variables and linking constraints.
        variables: Shared FJSP dictionary with completion and assignment data.
        instance: FJSP instance providing eligible-machine durations.

    Side Effects:
        Adds nominal durations ``D``, starts ``S`` and midpoints ``T`` to the
        shared variable dictionary.
    """
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


def add_pd_quadrature(model, variables):
    """Approximate midpoint disruption probabilities by Gauss quadrature.

    The nonlinear Weibull-repair integral supports shape parameters two and
    three and is linked exactly to the selected machine.
    """
    nodes, weights = gauss_legendre_rule(RELIABILITY_QUADRATURE_POINTS)
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
    })


def add_expected_repair_buffers(
    model,
    variables,
    instance,
):
    """Aggregate selected failure probabilities into job repair buffers.

    Each selected operation-machine probability is divided by the machine's
    repair rate and summed along the job chain. The resulting expressions enter
    the shared soft due-date constraints without a service-level multiplier.

    Args:
        model: Gurobi model receiving the due-date constraints.
        variables: Shared dictionary containing selected probabilities and
            machine repair rates.
        instance: FJSP instance defining the operations of each job.

    Side Effects:
        Adds due-date variables and buffer metadata through the shared helper.
    """
    job_repair_buffers = {}
    for job in instance.jobs:
        expected_disruption = gp.quicksum(
            variables["pi_fail"][operation, machine]
            / variables["repair_rate"][machine]
            for operation in instance.jobs[job]
            for machine in instance.eligible_machines[operation]
        )
        job_repair_buffers[job] = expected_disruption
    add_due_date_tardiness_constraints(
        model,
        variables,
        instance,
        job_repair_buffers,
    )
    variables.update({
        "service_scope": "job",
        "job_repair_buffer_label_method": "weibull_expected_repair_buffer_v1",
    })


def build_fjsp(
    fjsp,
    instance,
    facility_cost_per_time=1.0,
    tardiness_cost_per_time=1.0,
):
    """Build the nonlinear Weibull reference formulation.

    Extend the nominal FJSP with midpoint disruption probabilities, expected
    repair buffers, soft due dates and the economic objective.
    """
    parameters = stochastic_parameters(instance)
    model, variables = base_fjsp.build_core_fjsp(fjsp, instance)
    model.Params.NonConvex = 2
    variables.update({
        "weibull_alpha": parameters["alpha"],
        "weibull_beta": parameters["beta"],
        "repair_rate": parameters["repair_rate"],
    })
    add_midpoint_state(model, variables, instance)
    add_pd_quadrature(model, variables)
    add_expected_repair_buffers(
        model,
        variables,
        instance,
    )
    add_economic_cost_objective(
        model,
        variables,
        instance,
        facility_cost_per_time=facility_cost_per_time,
        tardiness_cost_per_time=tardiness_cost_per_time,
    )
    variables["formulation"] = "nonlinear_unscaled_local_repair_buffer_v12"
    model.update()
    return model, variables
