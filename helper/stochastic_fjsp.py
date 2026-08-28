"""Common cost and reliability parameters for stochastic FJSP models.

The optimization model, offline simulation and GNN pipeline share this module
as the source of Weibull/repair parameters and due dates.
Monte-Carlo service-probability labels are generated in ``05_Simulation``.
"""

from __future__ import annotations

import math
from typing import Mapping

import numpy as np


INDEPENDENT_GENERATION_MODEL = "independent_machine_parameters_v1"
PROFILE_GENERATION_MODEL = "old_new_machine_profiles_v2"
DEFAULT_INDEPENDENT_MACHINE_PARAMETER_RANGES = {
    "hourly_cost": (1.0, 2.25),
    "weibull_alpha": (24.0, 42.0),
    "weibull_beta": (1.6, 3.0),
    "repair_rate": (0.3, 0.7),
}
DEFAULT_MACHINE_PROFILE_CONFIG = {
    "profiles": {
        "old": {
            "cost_rate": 1.0,
            "speed": 0.8,
            "weibull_alpha": 60.0,
            "weibull_beta": 3.0,
            "repair_rate": 0.15,
        },
        "new": {
            "cost_rate": 1.6,
            "speed": 1.25,
            "weibull_alpha": 100.0,
            "weibull_beta": 2.0,
            "repair_rate": 0.4,
        },
    },
    "parameter_jitter": {
        "cost_rate": 0.0,
        "speed": 0.0,
        "weibull_alpha": 0.0,
        "repair_rate": 0.0,
    },
    "operation_time_noise": (0.9, 1.1),
    "minimum_profile_classes_per_operation": 2,
    "all_profiles_probability": 0.0,
    "additional_same_profile_machine_probability": 0.25,
}


def normalize_machine_profile_config(config=None):
    raw = dict(config or {})
    result = dict(DEFAULT_MACHINE_PROFILE_CONFIG)
    result.update(raw)
    profiles = result["profiles"]
    if set(profiles) != {"old", "new"}:
        raise ValueError("machine profiles must be old and new.")
    required = {
        "cost_rate",
        "speed",
        "weibull_alpha",
        "weibull_beta",
        "repair_rate",
    }
    normalized_profiles = {}
    for name, values in profiles.items():
        merged = dict(DEFAULT_MACHINE_PROFILE_CONFIG["profiles"][name])
        merged.update(values)
        unknown = set(merged) - required
        if unknown or not required.issubset(merged):
            raise ValueError(f"Invalid fields for machine profile {name!r}.")
        normalized_profiles[name] = {
            key: float(merged[key]) for key in required
        }
        if any(
            normalized_profiles[name][key] <= 0.0 for key in required
        ):
            raise ValueError(
                "Profile parameters must be positive."
            )
    expected_betas = {"old": 3.0, "new": 2.0}
    for name, expected in expected_betas.items():
        if not math.isclose(
            normalized_profiles[name]["weibull_beta"],
            expected,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"Machine profile {name!r} must use Weibull beta {expected:g}."
            )
    result["profiles"] = normalized_profiles
    for key in ("operation_time_noise",):
        values = tuple(map(float, result[key]))
        if len(values) != 2 or values[0] <= 0.0 or values[0] > values[1]:
            raise ValueError(f"Invalid machine profile range {key!r}.")
        result[key] = values
    jitter_fields = {"cost_rate", "speed", "weibull_alpha", "repair_rate"}
    raw_jitter = dict(DEFAULT_MACHINE_PROFILE_CONFIG["parameter_jitter"])
    raw_jitter.update(result["parameter_jitter"])
    jitter = {key: float(value) for key, value in raw_jitter.items()}
    if set(jitter) != jitter_fields or any(
        not 0.0 <= value < 1.0 for value in jitter.values()
    ):
        raise ValueError("Invalid machine profile jitter configuration.")
    result["parameter_jitter"] = jitter
    result["minimum_profile_classes_per_operation"] = int(
        result["minimum_profile_classes_per_operation"]
    )
    if not (
        1
        <= result["minimum_profile_classes_per_operation"]
        <= len(normalized_profiles)
    ):
        raise ValueError(
            "minimum_profile_classes_per_operation must lie between 1 and "
            "the number of profiles."
        )
    for key in (
        "all_profiles_probability",
        "additional_same_profile_machine_probability",
    ):
        result[key] = float(result[key])
        if not 0.0 <= result[key] <= 1.0:
            raise ValueError(f"{key} must lie in [0, 1].")
    return result


def _ensure_profile_parameters(instance):
    machines = range(int(instance.num_machines))
    instance.machine_profile_config = normalize_machine_profile_config(
        getattr(instance, "machine_profile_config", None)
    )
    instance.machine_profiles = {
        machine: str(instance.machine_profiles[machine])
        for machine in machines
    }
    unknown_profiles = (
        set(instance.machine_profiles.values())
        - set(instance.machine_profile_config["profiles"])
    )
    if unknown_profiles:
        raise ValueError(
            f"Unknown machine profiles: {sorted(unknown_profiles)}"
        )
    for name in (
        "machine_cost",
        "machine_speed",
        "weibull_alpha",
        "weibull_beta",
        "repair_rate",
    ):
        if name == "machine_cost" and not hasattr(instance, name):
            values = {
                machine: instance.machine_profile_config["profiles"][
                    instance.machine_profiles[machine]
                ]["cost_rate"]
                for machine in machines
            }
        else:
            values = getattr(instance, name)
        setattr(
            instance,
            name,
            {machine: float(values[machine]) for machine in machines},
        )
    instance.repair_duration = {
        machine: 1.0 / instance.repair_rate[machine]
        for machine in machines
    }
    instance.stochastic_schema_version = 6
    return instance


def normalize_independent_machine_parameter_ranges(config=None):
    """Validate independent per-machine parameter ranges."""
    values = dict(DEFAULT_INDEPENDENT_MACHINE_PARAMETER_RANGES)
    values.update(dict(config or {}))
    unknown = set(values) - set(DEFAULT_INDEPENDENT_MACHINE_PARAMETER_RANGES)
    if unknown:
        raise ValueError(
            f"Unknown independent machine parameters: {sorted(unknown)}"
        )
    result = {}
    for name, raw in values.items():
        if not isinstance(raw, (list, tuple)) or len(raw) != 2:
            raise ValueError(
                f"instances.generation.machine_parameters.{name} must "
                "contain two values."
            )
        lower, upper = map(float, raw)
        if lower > upper:
            raise ValueError(
                f"Invalid range for machine parameter {name!r}."
            )
        result[name] = (lower, upper)
    if result["weibull_alpha"][0] <= 0.0:
        raise ValueError("Weibull alpha values must be positive.")
    if result["weibull_beta"][0] <= 1.0:
        raise ValueError("Weibull beta values must exceed one.")
    if result["repair_rate"][0] <= 0.0:
        raise ValueError("Repair rates must be positive.")
    if result["hourly_cost"][0] < 0.0:
        raise ValueError("Machine costs must be nonnegative.")
    return result


def _machine_value(value, machine, default):
    if isinstance(value, Mapping):
        return float(value.get(machine, default))
    if isinstance(value, (list, tuple, np.ndarray)):
        return float(value[machine]) if machine < len(value) else float(default)
    if value is None:
        return float(default)
    return float(value)


def machine_parameter(instance, names, machine, default):
    """Read a scalar, list or mapping parameter with backwards aliases."""
    for name in names:
        if hasattr(instance, name):
            return _machine_value(getattr(instance, name), machine, default)
    return float(default)


def _modernity_levels(num_machines):
    if num_machines <= 1:
        return {0: 1.0}
    return {
        machine: machine / float(num_machines - 1)
        for machine in range(num_machines)
    }


def _ensure_independent_parameters(instance):
    """Validate a neutral instance without changing its FJSP structure."""
    machines = list(range(int(instance.num_machines)))
    operations = list(instance.real_operations)
    processing_times = dict(instance.processing_times)
    eligible_machines = {
        operation: list(instance.eligible_machines[operation])
        for operation in operations
    }
    for operation in operations:
        if not eligible_machines[operation]:
            raise ValueError("Every operation needs an eligible machine.")
        for machine in eligible_machines[operation]:
            if machine not in machines:
                raise ValueError("Eligible machine index is out of range.")
            if float(processing_times[operation, machine]) <= 0.0:
                raise ValueError("Processing times must be positive.")

    instance.machine_cost = {
        machine: machine_parameter(
            instance, ("machine_cost", "cost_rate", "c"), machine, 1.0
        )
        for machine in machines
    }
    instance.weibull_alpha = {
        machine: machine_parameter(
            instance, ("weibull_alpha", "weibull_eta", "eta"),
            machine, 33.0,
        )
        for machine in machines
    }
    instance.weibull_beta = {
        machine: machine_parameter(
            instance, ("weibull_beta", "beta"), machine, 2.0
        )
        for machine in machines
    }
    instance.repair_rate = {
        machine: machine_parameter(
            instance,
            ("repair_rate", "repair_lambda", "lambda_repair"),
            machine,
            0.5,
        )
        for machine in machines
    }
    for machine in machines:
        if instance.machine_cost[machine] < 0.0:
            raise ValueError("Machine costs must be nonnegative.")
        if instance.weibull_alpha[machine] <= 0.0:
            raise ValueError("Weibull alpha values must be positive.")
        if instance.weibull_beta[machine] <= 1.0:
            raise ValueError("Weibull beta values must exceed one.")
        if instance.repair_rate[machine] <= 0.0:
            raise ValueError("Repair rates must be positive.")

    # Kept only as a compatibility field for model/report interfaces. The
    # neutral generator does not use a separate speed variable.
    instance.machine_speed = {machine: 1.0 for machine in machines}
    instance.repair_duration = {
        machine: 1.0 / instance.repair_rate[machine]
        for machine in machines
    }
    instance.eligible_machines = eligible_machines
    instance.processing_times = processing_times

    serial_horizon = sum(
        max(
            float(processing_times[operation, machine])
            for machine in eligible_machines[operation]
        )
        for operation in operations
    )
    if not hasattr(instance, "due_dates"):
        instance.due_dates = {
            job: 0.65 * serial_horizon for job in instance.jobs
        }
    instance.stochastic_schema_version = 3
    return instance


def ensure_stochastic_parameters(instance, *, rebuild_modernity=False):
    """Attach a complete, machine-specific stochastic parameter set.

    Neutral instances retain their sampled eligibility and processing times.
    The modernity-based transformation is retained only for upgrading legacy
    pickles created by the previous generator.
    """
    if (
        getattr(instance, "instance_generation_model", None)
        == INDEPENDENT_GENERATION_MODEL
    ):
        return _ensure_independent_parameters(instance)
    if (
        getattr(instance, "instance_generation_model", None)
        == PROFILE_GENERATION_MODEL
    ):
        return _ensure_profile_parameters(instance)

    machines = list(range(int(instance.num_machines)))
    operations = list(instance.real_operations)
    native_v2 = all(
        hasattr(instance, name)
        for name in ("machine_modernity", "operation_requirement")
    )
    theta = {
        machine: machine_parameter(
            instance, ("machine_modernity", "theta"), machine,
            _modernity_levels(len(machines))[machine],
        )
        for machine in machines
    }
    theta_min = min(theta.values())
    theta_span = max(theta.values()) - theta_min
    theta_scaled = {
        machine: (
            (theta[machine] - theta_min) / theta_span
            if theta_span > 1e-12 else 1.0
        )
        for machine in machines
    }
    speed = {
        machine: machine_parameter(
            instance, ("machine_speed", "speed_factor", "e"), machine,
            0.80 + 0.60 * theta_scaled[machine],
        )
        for machine in machines
    }
    alpha = {
        machine: (machine_parameter(
            instance, ("weibull_alpha", "weibull_eta", "eta"), machine,
            26.0 + 12.0 * theta_scaled[machine],
        ) if native_v2 else 26.0 + 12.0 * theta_scaled[machine])
        for machine in machines
    }
    beta = {
        machine: (machine_parameter(
            instance, ("weibull_beta", "beta"), machine, 2.0,
        ) if native_v2 else 2.0)
        for machine in machines
    }
    repair_rate = {
        machine: (machine_parameter(
            instance, ("repair_rate", "repair_lambda", "lambda_repair"),
            machine, 0.35 + 0.25 * theta_scaled[machine],
        ) if native_v2 else 0.35 + 0.25 * theta_scaled[machine])
        for machine in machines
    }
    machine_cost = {
        machine: machine_parameter(
            instance, ("machine_cost", "cost_rate", "c"), machine,
            1.0 + 1.25 * theta_scaled[machine],
        )
        for machine in machines
    }
    for machine in machines:
        if machine_cost[machine] < 0.0:
            raise ValueError("Machine costs must be nonnegative.")
        if speed[machine] <= 0.0:
            raise ValueError("Machine speed must be positive.")
        if alpha[machine] <= 0.0 or beta[machine] <= 1.0:
            raise ValueError("Weibull alpha must be positive and beta must exceed 1.")
        if repair_rate[machine] <= 0.0:
            raise ValueError("Exponential repair rates must be positive.")

    existing_base = getattr(instance, "operation_base_time", {})
    base_time = {}
    requirements = {}
    processing_times = dict(instance.processing_times)
    eligible_machines = {
        operation: list(instance.eligible_machines[operation])
        for operation in operations
    }
    sorted_machines = sorted(machines, key=lambda k: (theta[k], k))
    has_modernity_data = native_v2
    if rebuild_modernity or not has_modernity_data:
        for operation in operations:
            old_eligible = eligible_machines[operation]
            option_count = max(1, min(len(machines), len(old_eligible)))
            new_eligible = sorted_machines[-option_count:]
            requirement = theta[new_eligible[0]]
            estimates = [
                float(processing_times[operation, machine]) * speed[machine]
                for machine in old_eligible
                if (operation, machine) in processing_times
            ]
            base = float(existing_base.get(operation, np.mean(estimates)))
            base_time[operation] = max(1.0, base)
            requirements[operation] = requirement
            eligible_machines[operation] = list(new_eligible)
            for machine in new_eligible:
                processing_times[operation, machine] = int(math.ceil(
                    base_time[operation] / speed[machine]
                ))
    else:
        requirements = {
            operation: float(instance.operation_requirement[operation])
            for operation in operations
        }
        base_time = {
            operation: float(existing_base.get(
                operation,
                np.mean([
                    float(processing_times[operation, machine]) * speed[machine]
                    for machine in eligible_machines[operation]
                ]),
            ))
            for operation in operations
        }

    for operation in operations:
        expected = [
            machine for machine in machines
            if theta[machine] + 1e-12 >= requirements[operation]
        ]
        if expected != eligible_machines[operation]:
            eligible_machines[operation] = expected
            for machine in expected:
                processing_times[operation, machine] = int(math.ceil(
                    base_time[operation] / speed[machine]
                ))

    instance.machine_modernity = theta
    instance.machine_cost = machine_cost
    instance.operation_requirement = requirements
    instance.machine_speed = speed
    instance.operation_base_time = base_time
    instance.eligible_machines = eligible_machines
    instance.processing_times = processing_times
    instance.weibull_alpha = alpha
    instance.weibull_beta = beta
    instance.repair_rate = repair_rate
    instance.repair_duration = {
        machine: 1.0 / repair_rate[machine] for machine in machines
    }

    serial_horizon = sum(
        max(
            float(processing_times[operation, machine])
            for machine in eligible_machines[operation]
        )
        for operation in operations
    )
    legacy_default_due = 1.15 * serial_horizon
    replace_legacy_due = (
        not hasattr(instance, "stochastic_schema_version")
        and hasattr(instance, "due_dates")
        and all(
            math.isclose(float(value), legacy_default_due, abs_tol=1e-9)
            for value in instance.due_dates.values()
        )
    )
    if not hasattr(instance, "due_dates") or replace_legacy_due:
        instance.due_dates = {
            job: 0.65 * serial_horizon
            for job in instance.jobs
        }
    instance.stochastic_schema_version = 2
    return instance


def stochastic_parameters(instance):
    ensure_stochastic_parameters(instance)
    machines = range(instance.num_machines)
    return {
        "generation_model": getattr(instance, "instance_generation_model", None),
        "profiles": dict(getattr(instance, "machine_profiles", {})),
        "theta": dict(getattr(
            instance,
            "machine_modernity",
            {machine: 0.0 for machine in machines},
        )),
        "speed": dict(instance.machine_speed),
        "cost": dict(instance.machine_cost),
        "alpha": dict(instance.weibull_alpha),
        "beta": dict(instance.weibull_beta),
        "repair_rate": dict(instance.repair_rate),
        "repair_duration": dict(instance.repair_duration),
        "due_dates": dict(instance.due_dates),
        "machines": list(machines),
    }


def weibull_down_probability(t, alpha, beta, repair_rate, order=64):
    """Evaluate P(A <= t < A+R) without Monte-Carlo scenarios."""
    t = max(0.0, float(t))
    alpha = float(alpha)
    beta = float(beta)
    repair_rate = float(repair_rate)
    if t == 0.0:
        return 0.0
    nodes, weights = np.polynomial.legendre.leggauss(int(order))
    x = 0.5 * t * (nodes + 1.0)
    density = (
        beta / alpha
        * np.power(x / alpha, beta - 1.0)
        * np.exp(-np.power(x / alpha, beta))
    )
    value = 0.5 * t * np.dot(
        weights, density * np.exp(-repair_rate * (t - x))
    )
    return float(np.clip(value, 0.0, 1.0))


def gauss_legendre_rule(order):
    nodes, weights = np.polynomial.legendre.leggauss(int(order))
    return tuple(float(v) for v in nodes), tuple(float(v) for v in weights)
