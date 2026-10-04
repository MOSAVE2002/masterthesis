"""Build reproducible calibrated benchmark and extrapolation solve plans.

The module expands configured size pairs and due-date offsets, generates or
reuses the exact expected instance files and verifies that stored calibration
metadata still matches the active experiment configuration.
"""

import importlib
import math
from pathlib import Path

from helper.due_date_calibration import (
    calibrated_total_work_content_due_dates,
)

ROOT_DIR = Path(__file__).resolve().parents[1]
_instances = importlib.import_module("01_generator.instance_generator")


def _absolute(path):
    """Resolve a path against the repository root while preserving absolutes.

    The returned :class:`Path` is used consistently for generated tier files.
    """
    path = Path(path)
    return path if path.is_absolute() else ROOT_DIR / path


def _size_pairs(tier, tier_name):
    """Validate and return unique positive ``(jobs, machines)`` pairs.

    Raises:
        ValueError: If the tier contains no pairs or malformed values.
    """
    raw_pairs = tier.get("size_pairs")
    if not isinstance(raw_pairs, (list, tuple)) or not raw_pairs:
        raise ValueError(f"{tier_name}.size_pairs must be a non-empty list.")
    pairs = []
    for raw_pair in raw_pairs:
        if not isinstance(raw_pair, (list, tuple)) or len(raw_pair) != 2:
            raise ValueError(
                f"{tier_name}.size_pairs entries must be [jobs, machines]."
            )
        pair = tuple(raw_pair)
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
            for value in pair
        ):
            raise ValueError(
                f"{tier_name}.size_pairs entries must contain positive integers."
            )
        if pair in pairs:
            raise ValueError(
                f"{tier_name}.size_pairs must not contain duplicates."
            )
        pairs.append(pair)
    return pairs


def _relative_offsets(due_date_config, tier_name):
    """Validate due-date offsets and create filesystem-safe unique slugs.

    Returns:
        Ordered ``(relative_offset, slug)`` pairs used by the solve plan.
    """
    raw_offsets = due_date_config.get("relative_makespan_offsets")
    if not isinstance(raw_offsets, (list, tuple)) or not raw_offsets:
        raise ValueError(
            f"{tier_name}.due_dates.relative_makespan_offsets must be "
            "a non-empty list."
        )
    offsets = [float(value) for value in raw_offsets]
    if any(not math.isfinite(value) or value <= -1.0 for value in offsets):
        raise ValueError(
            f"{tier_name}.due_dates relative makespan offsets must be "
            "finite and greater than -1."
        )
    if len(offsets) != len(set(offsets)):
        raise ValueError(
            f"{tier_name}.due_dates.relative_makespan_offsets must not "
            "contain duplicates."
        )
    slugs = [
        f"twk_d{offset:.2f}".replace("-", "m").replace(".", "p")
        for offset in offsets
    ]
    if len(slugs) != len(set(slugs)):
        raise ValueError(
            f"{tier_name}.due_dates offsets must remain distinct when "
            "rounded to two decimals for directory names."
        )
    return list(zip(offsets, slugs))


def _calibrate_instance(instance, due_date_config, relative_offset):
    """Apply one calibrated due-date offset to a generated solve instance.

    The derived due dates, factors, work content and nominal makespan are stored
    on the instance so reuse can later be validated without recalibration.
    """
    calibrated = calibrated_total_work_content_due_dates(
        instance, due_date_config, relative_offset
    )
    nominal = calibrated["calibration"]
    instance.due_dates = dict(calibrated["due_dates"])
    instance.due_date_factor = calibrated["effective_factor"]
    instance.due_date_work_content = dict(calibrated["work_content"])
    instance.due_date_relative_makespan_offset = calibrated["relative_offset"]
    instance.nominal_makespan_calibration = nominal["makespan"]
    instance.nominal_twk_due_date_factor = calibrated["nominal_factor"]


def _expected_paths(directory, specs, operations, suffix):
    """Construct the complete ordered set of expected tier pickle paths.

    Returns:
        One path per configured size and one-based instance number.
    """
    return [
        directory / (
            f"i{spec['num_jobs']}_k{spec['num_machines']}_"
            f"o{operations[0]}-{operations[1]}_{instance_number}_"
            f"{suffix}.pkl"
        )
        for spec in specs
        for instance_number in range(1, spec["count"] + 1)
    ]


def _validate_existing_instances(paths, tier_name, relative_offset):
    """Verify stored due-date calibration for every reused tier instance.

    Raises:
        ValueError: If offsets, factors, work content or due dates disagree.
    """
    for path in paths:
        instance = _instances.load_generated_instance(path.stem, path.parent)
        actual_offset = getattr(
            instance, "due_date_relative_makespan_offset", None
        )
        nominal_factor = getattr(instance, "nominal_twk_due_date_factor", None)
        actual_factor = getattr(instance, "due_date_factor", None)
        work_content = getattr(instance, "due_date_work_content", {})
        stored_due_dates = getattr(instance, "due_dates", {})
        factor_matches = (
            nominal_factor is not None
            and actual_factor is not None
            and math.isclose(
                float(actual_factor),
                (1.0 + relative_offset) * float(nominal_factor),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        )
        due_dates_match = (
            factor_matches
            and set(work_content) == set(instance.jobs)
            and set(stored_due_dates) == set(instance.jobs)
            and all(
                math.isclose(
                    float(stored_due_dates[job]),
                    float(math.ceil(
                        float(actual_factor) * float(work_content[job]) - 1e-12
                    )),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                for job in instance.jobs
            )
        )
        if (
            actual_offset is None
            or not math.isclose(
                float(actual_offset),
                relative_offset,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or not due_dates_match
        ):
            raise ValueError(
                f"Existing {tier_name} instance {path.name} does not match "
                "the calibrated TWK configuration."
            )


def generated_tier_plan(config, tier_name):
    """Create or validate one benchmark or extrapolation solve plan.

    Depending on ``solve.create_instances``, the function generates the exact
    configured set or verifies existing files and calibration metadata. Disabled
    tiers return an empty plan.

    Returns:
        Ordered dictionaries containing tier, instance name and directory.
    """
    experiment_plan = config["solve"]["experiment_plan"]
    tier = experiment_plan.get(tier_name, {})
    if not tier.get("enabled", False):
        return []

    generation = config["instances"]["generation"]
    operations = list(generation["operations_per_job"])
    count = tier.get("instances_per_size")
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError(
            f"{tier_name}.instances_per_size must be a positive integer."
        )
    specs = [
        {
            "num_jobs": jobs,
            "num_machines": machines,
            "operations_per_job": operations,
            "count": count,
        }
        for jobs, machines in _size_pairs(tier, tier_name)
    ]
    due_date_config = dict(tier.get("due_dates") or {})
    create_instances = bool(config["solve"].get("create_instances", False))
    reuse_existing = bool(tier.get("reuse_existing_instances", True))
    root = _absolute(experiment_plan["instance_directory"])
    plan = []

    for relative_offset, factor_slug in _relative_offsets(
        due_date_config, tier_name
    ):
        directory = root / tier_name / factor_slug
        suffix = f"{tier_name}_{factor_slug}"
        expected_paths = _expected_paths(directory, specs, operations, suffix)
        existing_paths = (
            sorted(directory.glob("*.pkl")) if directory.exists() else []
        )

        if create_instances:
            def postprocess(instance, *, _offset=relative_offset):
                """Apply this iteration's captured due-date offset before saving.

                The default argument prevents later loop iterations from
                changing the callback's calibration scenario.
                """
                _calibrate_instance(instance, due_date_config, _offset)

            paths = _instances.generate_solve_instance_specs(
                specs,
                machine_profile_config=generation["machine_profiles"],
                due_date_config=generation["due_dates"],
                random_seed=int(experiment_plan.get("random_seed", 2026)),
                output_directory=directory,
                processing_time_range=generation["processing_times"].get(
                    "base_range"
                ),
                instance_postprocessor=postprocess,
                instance_name_suffix=suffix,
            )
        elif reuse_existing and existing_paths:
            expected_set = {path.resolve() for path in expected_paths}
            existing_set = {path.resolve() for path in existing_paths}
            if existing_set != expected_set:
                missing = sorted(
                    path.name for path in expected_set - existing_set
                )
                unexpected = sorted(
                    path.name for path in existing_set - expected_set
                )
                raise ValueError(
                    f"Existing {tier_name} instances do not match the "
                    f"configured solve plan. Missing={missing}; "
                    f"unexpected={unexpected}. Enable solve.create_instances "
                    "to regenerate the configured set."
                )
            paths = expected_paths
            _validate_existing_instances(paths, tier_name, relative_offset)
            print(
                f"Reusing {len(paths)} unchanged {tier_name} instances "
                f"from: {directory}",
                flush=True,
            )
        elif not existing_paths:
            raise FileNotFoundError(
                f"No {tier_name} instances exist in {directory}. Enable "
                "solve.create_instances to generate the configured set."
            )
        else:
            raise ValueError(
                f"Existing {tier_name} instances may not be regenerated while "
                "solve.create_instances is false and reuse_existing_instances "
                "is disabled."
            )

        plan.extend({
            "tier": tier_name,
            "instance_name": path.stem,
            "instance_directory": str(directory),
        } for path in paths)
    return plan
