"""Normalize shared stochastic machine parameters for all pipeline stages.

The helpers validate machine profiles and train-only jitter, normalize loaded
instances and expose the Weibull and repair mappings used by optimization,
training-data generation and simulation.
"""

import math
import numpy as np

def normalize_training_parameter_jitter(config):
    """Validate and normalize the train-only parameter-jitter settings.
       Returns the configuration with consistent numeric types and raises
       ``ValueError`` for missing fields or invalid values.
    """
    required = {
        "enabled",
        "fraction_per_size",
        "random_seed",
        "parameter_jitter",
    }
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError("training_parameter_jitter must contain exactly: "f"{sorted(required)}.")
    if not isinstance(config["enabled"], bool):
        raise ValueError("training_parameter_jitter.enabled must be boolean.")
    if isinstance(config["random_seed"], bool):
        raise ValueError("training_parameter_jitter.random_seed must be an integer.")

    fraction = float(config["fraction_per_size"])
    if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise ValueError("training_parameter_jitter.fraction_per_size must lie in [0, 1].")

    jitter_fields = {"cost_rate", "speed", "weibull_alpha", "repair_rate"}
    raw_jitter = config["parameter_jitter"]
    if not isinstance(raw_jitter, dict) or set(raw_jitter) != jitter_fields:
        raise ValueError("training_parameter_jitter.parameter_jitter must contain exactly: " f"{sorted(jitter_fields)}.")
    jitter = {key: float(raw_jitter[key]) for key in jitter_fields}
    if any(
        not math.isfinite(value) or not 0.0 <= value < 1.0
        for value in jitter.values()
    ):
        raise ValueError("Invalid training_parameter_jitter.parameter_jitter configuration.")

    if config["enabled"]:
        if fraction <= 0.0:
            raise ValueError("Enabled training parameter jitter requires a positive fraction_per_size.")
        if not any(jitter.values()):
            raise ValueError("Enabled training parameter jitter requires at least one positive jitter width.")
    return {
        "enabled": config["enabled"],
        "fraction_per_size": fraction,
        "random_seed": int(config["random_seed"]),
        "parameter_jitter": jitter,
    }


def normalize_machine_profile_config(config):
    """Validate and normalize the old and new machine-profile settings.
       Checks profile parameters, jitter ranges, processing-time noise and
       machine-selection probabilities before returning normalized values.
    """
    required_config = {
        "profiles",
        "parameter_jitter",
        "operation_time_noise",
        "minimum_profile_classes_per_operation",
        "all_profiles_probability",
        "additional_same_profile_machine_probability",
    }
    if not isinstance(config, dict) or set(config) != required_config:
        raise ValueError("machine_profiles must contain exactly: "f"{sorted(required_config)}.")

    profiles = config["profiles"]
    if not isinstance(profiles, dict) or set(profiles) != {"old", "new"}:
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
        if not isinstance(values, dict) or set(values) != required:
            raise ValueError(f"Invalid fields for machine profile {name!r}.")
        normalized_profiles[name] = {key: float(values[key]) for key in required}
        if any(normalized_profiles[name][key] <= 0.0 for key in required):
            raise ValueError("Profile parameters must be positive.")
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
    operation_time_noise = tuple(map(float, config["operation_time_noise"]))
    if (
        len(operation_time_noise) != 2
        or not all(math.isfinite(value) for value in operation_time_noise)
        or operation_time_noise[0] <= 0.0
        or operation_time_noise[0] > operation_time_noise[1]
    ):
        raise ValueError("Invalid machine profile range 'operation_time_noise'.")

    jitter_fields = {"cost_rate", "speed", "weibull_alpha", "repair_rate"}
    raw_jitter = config["parameter_jitter"]
    if not isinstance(raw_jitter, dict) or set(raw_jitter) != jitter_fields:
        raise ValueError("machine_profiles.parameter_jitter must contain exactly: "f"{sorted(jitter_fields)}.")
    jitter = {key: float(raw_jitter[key]) for key in jitter_fields}
    if any(
        not math.isfinite(value) or not 0.0 <= value < 1.0
        for value in jitter.values()
    ):
        raise ValueError("Invalid machine profile jitter configuration.")

    minimum_profiles = int(config["minimum_profile_classes_per_operation"])
    if not (
        1
        <= minimum_profiles
        <= len(normalized_profiles)
    ):
        raise ValueError("minimum_profile_classes_per_operation must lie between 1 and the number of profiles.")
    probabilities = {
        key: float(config[key])
        for key in ("all_profiles_probability","additional_same_profile_machine_probability")
    }
    for key, value in probabilities.items():
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{key} must lie in [0, 1].")

    return {
        "profiles": normalized_profiles,
        "parameter_jitter": jitter,
        "operation_time_noise": operation_time_noise,
        "minimum_profile_classes_per_operation": minimum_profiles,
        **probabilities,
    }


def ensure_profile_parameters(instance):
    """Validate and normalize all stochastic attributes of an instance.

    Machine IDs, profiles, costs, speeds, Weibull parameters, repair rates and
    due dates are converted to stable numeric mappings. Invalid signs, unknown
    profiles and unsupported Weibull shapes are rejected in place.
    """
    machines = list(range(int(instance.num_machines)))
    instance.machine_profile_config = normalize_machine_profile_config(
        instance.machine_profile_config
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
        values = getattr(instance, name)
        setattr(
            instance,
            name,
            {machine: float(values[machine]) for machine in machines},
        )
    for machine in machines:
        if instance.machine_cost[machine] < 0.0:
            raise ValueError("Machine costs must be nonnegative.")
        if instance.machine_speed[machine] <= 0.0:
            raise ValueError("Machine speeds must be positive.")
        if instance.weibull_alpha[machine] <= 0.0:
            raise ValueError("Weibull alpha values must be positive.")
        if instance.weibull_beta[machine] <= 1.0:
            raise ValueError("Weibull beta values must exceed one.")
        if instance.repair_rate[machine] <= 0.0:
            raise ValueError("Repair rates must be positive.")
    instance.due_dates = {
        job: float(instance.due_dates[job]) for job in instance.jobs
    }


def stochastic_parameters(instance):
    """Return independent Weibull and repair mappings for one instance.

    The instance is normalized first so downstream model builders receive
    complete numeric dictionaries indexed by every machine.
    """
    ensure_profile_parameters(instance)
    return {
        "alpha": dict(instance.weibull_alpha),
        "beta": dict(instance.weibull_beta),
        "repair_rate": dict(instance.repair_rate),
    }


def gauss_legendre_rule(order):
    """Return immutable Gauss-Legendre nodes and weights of a given order.

    Args:
        order: Positive quadrature order accepted by NumPy.

    Returns:
        Two tuples containing floating-point nodes and weights.
    """
    nodes, weights = np.polynomial.legendre.leggauss(int(order))
    return tuple(float(v) for v in nodes), tuple(float(v) for v in weights)
