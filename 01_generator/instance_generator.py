"""Generate, split, persist and validate stochastic FJSP instances.

The module creates reproducible flexible job-shop instances with heterogeneous
machine profiles, due dates and optional train-only parameter jitter. It also
owns the canonical train/validation/test split and the pickle-based persistence
used by graph generation and all solver tiers.
"""

import pickle
import random
import math
import copy
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
INSTANCE_DIRECTORY = ROOT_DIR / "02_data" / "fjsp_instances"
SPLIT_NAMES = ("train", "valid", "test")

SPLIT_DIRECTORIES = {
    "train": "training",
    "valid": "valid",
    "test": "test",
}
SPLIT_CSV_FILENAMES = {
    "train": "graphs_training.csv",
    "valid": "graphs_valid.csv",
    "test": "graphs_test.csv",
}
class FJSPData:
    """Represent one stochastic flexible job-shop scheduling instance.

    The class generates and stores all data required by the nominal,
    nonlinear and GNN-embedded optimization models.  An instance contains the
    job precedence structure, eligible machines and processing times as well
    as machine-specific cost, speed, Weibull and repair parameters.  Generated
    objects are serialized as pickle files and later loaded without rebuilding
    the random instance.

    Instance construction proceeds in five stages:

    1. Draw the number of operations for every job.
    2. Build the job and predecessor mappings.
    3. Assign the configured old/new machine profiles and generate final
       operation eligibility and profile-dependent processing times.
    4. Calculate job-specific due dates from total expected work content.
    5. Validate and normalize the stochastic instance schema.

    Parameters
    ----------
    nb_instance : int
        One-based instance number.  It is part of the instance name and
        determines the cyclic due-date factor.
    num_jobs : int
        Number of jobs in the instance.
    num_machines : int
        Number of available machines.  Machine identifiers are zero-based.
    operations_per_job_min : int
        Minimum number of operations assigned to a job.
    operations_per_job_max : int
        Maximum number of operations assigned to a job.  Each job receives an
        independently sampled integer in the inclusive minimum/maximum range.
    machine_profile_config : dict
        Configuration of the ``old`` and ``new`` machine profiles, parameter
        jitter, operation-time noise and eligibility probabilities.
    due_date_config : dict
        Due-date configuration containing the positive total-work-content
        factors.  Factors are assigned cyclically by instance number.
    processing_time_range : sequence of int, optional
        Inclusive lower and upper bounds for sampled operation base times.
        The default is ``(1, 10)``.
    random_source : random.Random, optional
        Random-number source used for all draws.  Supplying a seeded source
        makes instance generation reproducible; otherwise the module-level
        :mod:`random` generator is used.

    Attributes
    ----------
    instance_name : str
        Stable name containing jobs, machines, operation range and instance
        number.
    jobs : dict[int, list[int]]
        Mapping from one-based job identifiers to their ordered operations.
        Operation identifiers are also one-based.
    real_operations : list[int]
        Flat list of all operation identifiers.
    predecessors : dict[int, list[int]]
        Direct technological predecessors of every operation.
    job_end_operations : dict[int, int]
        Final operation of every job.
    eligible_machines : dict[int, list[int]]
        Zero-based machines that may process each operation.
    processing_times : dict[tuple[int, int], int]
        Processing duration for every eligible operation-machine pair.
    machine_profiles : dict[int, str]
        Assigned ``old`` or ``new`` profile for every machine.
    machine_speed, machine_cost : dict[int, float]
        Speed multiplier and operating-cost rate of every machine.
    weibull_alpha, weibull_beta : dict[int, float]
        Machine-specific Weibull scale and shape parameters.
    repair_rate : dict[int, float]
        Exponential repair rate of every machine.
    due_dates : dict[int, float]
        Job-specific total-work-content due dates.
    due_date_factor : float
        Factor selected for this instance from ``due_date_config``.
    due_date_work_content : dict[int, float]
        Mean-machine work content used to calculate each job's due date.
    """

    def set_profile_machine_parameters(self, rng, config):
        """Generate machine attributes, operation eligibility and durations.

        Machine profiles are distributed across all machines and their numeric
        parameters are jittered reproducibly. Each operation receives machines
        from the required number of profile classes plus optional alternatives;
        processing times follow profile speed and configured random noise.

        Args:
            rng: Random-number source shared by instance generation.
            config: Validated or raw machine-profile configuration.
        """
        from helper.stochastic_fjsp import normalize_machine_profile_config

        cfg = normalize_machine_profile_config(config)
        profile_names = list(cfg["profiles"])
        assignments = [profile_names[index % len(profile_names)]for index in range(self.num_machines)]
        rng.shuffle(assignments)
        self.machine_profile_config = cfg
        self.machine_profiles = dict(enumerate(assignments))

        def jittered(machine, field):
            """Draw one profile parameter inside its symmetric jitter range.

            Args:
                machine: Zero-based machine identifier.
                field: Profile parameter to retrieve and perturb.

            Returns:
                The profile value multiplied by a reproducible uniform factor.
            """
            profile = cfg["profiles"][self.machine_profiles[machine]]
            width = cfg["parameter_jitter"][field]
            return profile[field] * rng.uniform(1.0 - width, 1.0 + width)

        machines = range(self.num_machines)
        self.machine_speed = {machine: jittered(machine, "speed")for machine in machines}
        self.machine_cost = {machine: jittered(machine, "cost_rate")for machine in machines}
        self.weibull_alpha = {machine: jittered(machine, "weibull_alpha")for machine in machines}
        self.repair_rate = {machine: jittered(machine, "repair_rate")for machine in machines}
        self.weibull_beta = {machine: float(cfg["profiles"][self.machine_profiles[machine]]["weibull_beta"])for machine in machines }
        by_profile = {
            profile: [
                machine
                for machine in machines
                if self.machine_profiles[machine] == profile
            ]
            for profile in profile_names
        }
        noise_lower, noise_upper = cfg["operation_time_noise"]
        self.eligible_machines = {}
        self.processing_times = {}
        for operation in self.real_operations:
            base = rng.randint(
                self.processing_time_per_ope_min,
                self.processing_time_per_ope_max,
            )
            available_profiles = [name for name in profile_names if by_profile[name]]
            minimum = min(cfg["minimum_profile_classes_per_operation"],len(available_profiles),)
            selected_profiles = rng.sample(available_profiles, minimum)
            if (
                len(available_profiles) > minimum
                and rng.random() < cfg["all_profiles_probability"]
            ):
                selected_profiles = available_profiles
            eligible = [rng.choice(by_profile[name]) for name in selected_profiles]
            for machine in machines:
                if (
                    machine not in eligible
                    and rng.random()
                    < cfg["additional_same_profile_machine_probability"]
                ):
                    eligible.append(machine)
            eligible = sorted(set(eligible))
            self.eligible_machines[operation] = eligible
            for machine in eligible:
                duration = max(
                    1,
                    math.ceil(
                        base
                        / self.machine_speed[machine]
                        * rng.uniform(noise_lower, noise_upper)
                    ),
                )
                self.processing_times[operation, machine] = duration

    def set_due_dates(self, due_date_config):
        """Derive reproducible job due dates from total work content.

        The instance number selects a configured factor cyclically. For each
        job, operation times are averaged across eligible machines, summed and
        multiplied by that factor before rounding upward.

        Args:
            due_date_config: Mapping containing positive ``factors``.

        Raises:
            ValueError: If factors or resulting work contents are nonpositive.
        """
        factors = [float(value)for value in due_date_config["factors"]]
        if not factors or any(
            not math.isfinite(value) or value <= 0.0
            for value in factors
        ):
            raise ValueError("due_dates.factors must contain positive values.")
        factor = factors[(int(self.nb_instance) - 1) % len(factors)]
        work_content = {job: sum(sum(float(self.processing_times[operation, machine]) for machine in self.eligible_machines[operation]) / len(self.eligible_machines[operation]) for operation in operations)for job, operations in self.jobs.items()}
        if any(value <= 0.0 for value in work_content.values()):
            raise ValueError("Every job must have positive total work content.")
        self.due_date_factor = factor
        self.due_date_work_content = work_content
        self.due_dates = {job: float(math.ceil(factor * work_content[job]))for job in self.jobs}

    def build_operation_metadata(self):
        """Build canonical operation identifiers and technological precedence.

        Operations receive consecutive one-based IDs. The method constructs
        the ordered operation list of every job, the direct predecessor of each
        operation and the final operation used by due-date constraints.
        """
        self.jobs = {}
        self.predecessors = {}
        self.job_end_operations = {}
        self.real_operations = []

        operation_id = 1

        for job_index in range(self.num_jobs):
            num_operations = self.nums_operation[job_index]
            job_operations = []

            for local_operation_index in range(num_operations):
                current_operation = operation_id
                operation_id += 1
                job_operations.append(current_operation)
                self.real_operations.append(current_operation)

                if local_operation_index == 0:
                    self.predecessors[current_operation] = []
                else:
                    self.predecessors[current_operation] = [job_operations[local_operation_index - 1]]

            self.jobs[job_index + 1] = job_operations
            self.job_end_operations[job_index + 1] = job_operations[-1]

    def __init__(
        self,
        nb_instance: int,
        num_jobs: int,
        num_machines: int,
        operations_per_job_min: int,
        operations_per_job_max: int,
        machine_profile_config,
        due_date_config,
        processing_time_range=None,
        random_source=None):
        """Initialize and fully generate one stochastic FJSP instance.

        Constructor arguments follow the class documentation. Generation is
        completed immediately: operation counts, precedence, machine profiles,
        processing times and due dates are created and normalized.

        Raises:
            ValueError: If the processing-time interval is malformed or not
                strictly positive.
        """
        
        self.nb_instance = nb_instance
        rng = random_source if random_source is not None else random

        # instance parameters
        self.num_jobs = num_jobs
        self.num_machines = num_machines

        # Processing time parameters
        processing_time_range = processing_time_range or (1, 10)
        if len(processing_time_range) != 2:
            raise ValueError("processing_time_range must contain two values.")
        self.processing_time_per_ope_min = int(processing_time_range[0])
        self.processing_time_per_ope_max = int(processing_time_range[1])
        if (
            self.processing_time_per_ope_min <= 0
            or self.processing_time_per_ope_min
            > self.processing_time_per_ope_max
        ):
            raise ValueError("Invalid positive processing-time range.")
         # Instance Name
        self.instance_name = (
            f"i{self.num_jobs}_k{self.num_machines}_"
            f"o{operations_per_job_min}-{operations_per_job_max}_{self.nb_instance}"
        )


        if operations_per_job_min == operations_per_job_max:
            self.nums_operation = [operations_per_job_min for _ in range(self.num_jobs)]
        else:
            self.nums_operation = [rng.randint(operations_per_job_min, operations_per_job_max) for _ in range(self.num_jobs)]
        self.num_operations = sum(self.nums_operation) # Amount of operations
        self.build_operation_metadata()
        self.set_profile_machine_parameters(rng, machine_profile_config)
        self.set_due_dates(due_date_config)
        from helper.stochastic_fjsp import ensure_profile_parameters
        ensure_profile_parameters(self)

    def __repr__(self):
        """Return a compact representation containing the stable instance name.

        Returns:
            Text in the form ``FJSP(<instance_name>)``.
        """
        return f"FJSP({self.instance_name})"

def generate_instance_specs(
    specs,
    machine_profile_config,
    due_date_config,
    split_ratios,
    random_seed,
    processing_time_range,
    training_parameter_jitter,
    output_directory=None,
):
    """Generate configured sizes and split every size independently.

    A configured training jitter is assigned to an exact, reproducible share
    of each size's training instances. Validation and test instances retain
    the unjittered machine profiles.
    """
    from helper.stochastic_fjsp import (
        normalize_machine_profile_config,
        normalize_training_parameter_jitter,
    )

    jitter_config = normalize_training_parameter_jitter(training_parameter_jitter)
    base_profile_config = normalize_machine_profile_config(machine_profile_config)
    if jitter_config["enabled"] and any(
        base_profile_config["parameter_jitter"].values()
    ):
        raise ValueError(
            "Train-only jitter requires zero baseline parameter_jitter values."
        )
    jittered_profile_config = copy.deepcopy(base_profile_config)
    if jitter_config["enabled"]: jittered_profile_config["parameter_jitter"] = dict(jitter_config["parameter_jitter"])

    split_instances = {split_name: [] for split_name in SPLIT_NAMES}
    generation_rng = random.Random(int(random_seed))
    for spec_index, spec in enumerate(specs):
        operation_min = min(spec["operations_per_job"])
        operation_max = max(spec["operations_per_job"])
        instance_names = [
            f"i{int(spec['num_jobs'])}_k{int(spec['num_machines'])}_"
            f"o{operation_min}-{operation_max}_{instance_nb}"
            for instance_nb in range(1, int(spec["count"]) + 1)
        ]
        size_split_names = split_items(instance_names,split_ratios=split_ratios,random_seed=int(random_seed) + spec_index,)
        jitter_count = (
            round(
                len(size_split_names["train"])
                * jitter_config["fraction_per_size"]
            )
            if jitter_config["enabled"] else 0
        )
        jittered_names = set(random.Random(jitter_config["random_seed"] + spec_index).sample(size_split_names["train"], jitter_count))

        size_instances = {}
        for instance_nb, instance_name in enumerate(instance_names, start=1):
            use_jitter = instance_name in jittered_names
            instance = FJSPData(
                nb_instance=instance_nb,
                num_jobs=spec["num_jobs"],
                num_machines=spec["num_machines"],
                operations_per_job_min=operation_min,
                operations_per_job_max=operation_max,
                machine_profile_config=(
                    jittered_profile_config
                    if use_jitter else base_profile_config
                ),
                due_date_config=due_date_config,
                processing_time_range=processing_time_range,
                random_source=generation_rng,
            )
            instance.training_parameter_jitter_applied = use_jitter
            size_instances[instance_name] = instance
        for split_name in SPLIT_NAMES:
            split_instances[split_name].extend(
                size_instances[name] for name in size_split_names[split_name]
            )
    save_pre_split_instances(split_instances,output_directory=output_directory,)


def generate_solve_instance_specs(
    specs,
    machine_profile_config,
    due_date_config,
    random_seed=42,
    output_directory=None,
    processing_time_range=None,
    instance_postprocessor=None,
    instance_name_suffix=None,
):
    """Generate a reproducible flat set of solver instances.

    Args:
        specs: Instance-size specifications with jobs, machines and counts.
        machine_profile_config: Machine profile generation settings.
        due_date_config: Total-work-content due-date factors.
        random_seed: Seed shared by all generated instances.
        output_directory: Destination for flat pickle files.
        processing_time_range: Inclusive operation base-time interval.
        instance_postprocessor: Optional callback applied before persistence.
        instance_name_suffix: Optional suffix distinguishing a solve tier.

    Returns:
        Paths of all newly written pickle files.

    Raises:
        ValueError: If the suffix is blank or generated names are not unique.
    """
    output_directory = Path(output_directory or INSTANCE_DIRECTORY)
    generation_rng = random.Random(int(random_seed))
    instances = []
    for spec in specs:
        instances.extend(
            FJSPData(
                nb_instance=instance_number,
                num_jobs=spec["num_jobs"],
                num_machines=spec["num_machines"],
                operations_per_job_min=min(spec["operations_per_job"]),
                operations_per_job_max=max(spec["operations_per_job"]),
                machine_profile_config=machine_profile_config,
                due_date_config=due_date_config,
                processing_time_range=processing_time_range,
                random_source=generation_rng,
            )
            for instance_number in range(1, spec["count"] + 1)
        )

    names = [instance.instance_name for instance in instances]
    if instance_name_suffix:
        suffix = str(instance_name_suffix).strip().strip("_")
        if not suffix:
            raise ValueError("instance_name_suffix must not be blank.")
        for instance in instances:
            instance.instance_name = f"{instance.instance_name}_{suffix}"
        names = [instance.instance_name for instance in instances]
    if instance_postprocessor is not None:
        for instance in instances:
            instance_postprocessor(instance)
    if len(names) != len(set(names)):
        raise ValueError("Solve instance names must be unique.")

    output_directory.mkdir(parents=True, exist_ok=True)
    removed = []
    for existing_path in output_directory.glob("*.pkl"):
        existing_path.unlink()
        removed.append(existing_path)
    paths = []
    for instance in instances:
        path = output_directory / f"{instance.instance_name}.pkl"
        with path.open("wb") as output_file:
            pickle.dump(instance, output_file)
        paths.append(path)
    print(
        f"Replaced {len(removed)} old and wrote {len(paths)} solve "
        f"instances to: {output_directory}"
    )
    return paths


def save_pre_split_instances(
    split_instances,
    output_directory=None,
):
    """Persist a disjoint train/validation/test instance assignment.

    Existing generated pickles in the three split directories are removed
    first. The supplied instances are then serialized under their stable names.

    Args:
        split_instances: Mapping from canonical split names to instances.
        output_directory: Root directory containing the split subdirectories.

    Raises:
        ValueError: If a split is unknown or an instance name occurs twice.
    """
    unknown = set(split_instances) - set(SPLIT_NAMES)
    if unknown:
        raise ValueError(f"Unknown dataset splits: {sorted(unknown)}")
    normalized = {
        split_name: list(split_instances.get(split_name, []))
        for split_name in SPLIT_NAMES
    }
    names = [
        instance.instance_name
        for values in normalized.values()
        for instance in values
    ]
    if len(names) != len(set(names)):
        raise ValueError("Generated instance names must be unique across all sizes.")

    output_directory = Path(output_directory or INSTANCE_DIRECTORY)
    clear_generated_instance_splits(output_directory)
    for split_name, values in normalized.items():
        split_directory = output_directory / SPLIT_DIRECTORIES[split_name]
        split_directory.mkdir(parents=True, exist_ok=True)
        for instance in values:
            path = split_directory / f"{instance.instance_name}.pkl"
            with path.open("wb") as output_file:
                pickle.dump(instance, output_file)
        print(
            f"Wrote {len(values)} {SPLIT_DIRECTORIES[split_name]} instances to: "
            f"{split_directory}"
        )


def normalized_split_ratios(split_ratios):
    """Validate and normalize train, validation and test proportions.

    Args:
        split_ratios: Mapping containing exactly the canonical split names.

    Returns:
        Floating-point ratios whose sum equals one.

    Raises:
        ValueError: If keys are missing, values are invalid or the total is
            not positive.
    """
    if not isinstance(split_ratios, dict) or set(split_ratios) != set(SPLIT_NAMES):
        raise ValueError(
            f"Split ratios must contain exactly: {sorted(SPLIT_NAMES)}."
        )
    values = {name: float(split_ratios[name]) for name in SPLIT_NAMES}
    if any(
        not math.isfinite(value) or value < 0.0
        for value in values.values()
    ):
        raise ValueError("Split ratios must be finite and nonnegative.")
    total = sum(values.values())
    if total <= 0.0:
        raise ValueError("At least one split ratio must be positive.")
    return {name: value / total for name, value in values.items()}


def split_items(items, split_ratios, random_seed):
    """Shuffle and distribute items reproducibly across dataset splits.

    Fractional target counts are rounded by largest remainder. Whenever enough
    items exist, every split with a positive ratio receives at least one item.

    Args:
        items: Values to distribute without duplication or loss.
        split_ratios: Train, validation and test proportions.
        random_seed: Seed controlling the initial permutation.

    Returns:
        Mapping from split name to its ordered list of assigned items.
    """
    ratios = normalized_split_ratios(split_ratios)
    shuffled = list(items)
    random.Random(int(random_seed)).shuffle(shuffled)
    raw_counts = {name: len(shuffled) * ratios[name] for name in SPLIT_NAMES}
    counts = {name: int(raw_counts[name]) for name in SPLIT_NAMES}
    remaining = len(shuffled) - sum(counts.values())
    remainder_order = sorted(
        SPLIT_NAMES,
        key=lambda name: (raw_counts[name] - counts[name], ratios[name]),
        reverse=True,
    )
    for name in remainder_order[:remaining]:
        counts[name] += 1

    positive_splits = [name for name in SPLIT_NAMES if ratios[name] > 0.0]
    if len(shuffled) >= len(positive_splits):
        for empty_name in [name for name in positive_splits if counts[name] == 0]:
            donors = [name for name in positive_splits if counts[name] > 1]
            donor = max(
                donors,
                key=lambda name: (counts[name] - raw_counts[name], counts[name]),
            )
            counts[donor] -= 1
            counts[empty_name] += 1

    result = {}
    offset = 0
    for name in SPLIT_NAMES:
        result[name] = shuffled[offset:offset + counts[name]]
        offset += counts[name]
    return result


def clear_generated_instance_splits(output_directory=None):
    """Remove generated pickle files from all canonical split directories.

    Args:
        output_directory: Dataset root; defaults to ``INSTANCE_DIRECTORY``.

    Side Effects:
        Creates missing split directories and deletes their ``.pkl`` files.
    """
    output_directory = Path(output_directory or INSTANCE_DIRECTORY)
    removed_count = 0
    for split_directory_name in SPLIT_DIRECTORIES.values():
        split_directory = output_directory / split_directory_name
        split_directory.mkdir(parents=True, exist_ok=True)
        for existing_path in split_directory.glob("*.pkl"):
            existing_path.unlink()
            removed_count += 1
    print(
        f"Removed {removed_count} old instances from: "
        f"{output_directory}"
    )


def load_generated_instance(instance_name, instance_directory=None):
    """Load and normalize an instance from a flat or split directory.

    Args:
        instance_name: Filename stem of the requested generated instance.
        instance_directory: Root directory searched before the standard root.

    Returns:
        The deserialized instance with normalized stochastic parameters.

    Raises:
        FileNotFoundError: If no matching pickle exists in any location.
    """
    from helper.stochastic_fjsp import ensure_profile_parameters

    def loaded(path):
        """Deserialize one pickle and normalize its stochastic attributes.

        Args:
            path: Concrete pickle path selected by the outer search.

        Returns:
            A validated generated instance.
        """
        with path.open("rb") as instance_file:
            instance = pickle.load(instance_file)
        ensure_profile_parameters(instance)
        return instance

    directory = Path(instance_directory or INSTANCE_DIRECTORY)
    direct_path = directory / f"{instance_name}.pkl"
    if direct_path.exists():
        return loaded(direct_path)
    for split_name in SPLIT_NAMES:
        instance_path = (
            directory
            / SPLIT_DIRECTORIES[split_name]
            / f"{instance_name}.pkl"
        )
        if instance_path.exists():
            return loaded(instance_path)
    raise FileNotFoundError(f"Generated instance not found: {instance_name}")


def generated_instance_names(instance_directory=None):
    """List every generated instance available below one dataset root.

    Args:
        instance_directory: Root containing flat and split pickle files.

    Returns:
        Sorted unique filename stems from all supported locations.
    """
    directory = Path(instance_directory or INSTANCE_DIRECTORY)
    names = {
        path.stem
        for split_name in SPLIT_NAMES
        for path in (directory / SPLIT_DIRECTORIES[split_name]).glob("*.pkl")
    }
    names.update(path.stem for path in directory.glob("*.pkl"))
    return sorted(names)


def configured_instance_names_by_split(
    specs,
    machine_profile_config,
    due_date_config,
    split_ratios,
    random_seed,
    processing_time_range,
    training_parameter_jitter,
    instance_directory=None,
):
    """Reconstruct configured splits and validate every stored instance.

    Expected names, split membership and train-only jitter assignment are
    derived deterministically from the current configuration. Loaded pickles
    are checked for dimensions, operation counts, processing-time range,
    machine profiles, jitter status and due-date metadata before their names
    are returned to graph generation.

    Args:
        specs: Configured Cartesian instance-size specifications.
        machine_profile_config: Expected baseline machine profiles.
        due_date_config: Expected cyclic due-date factors.
        split_ratios: Expected train/validation/test proportions.
        random_seed: Seed used when the stored splits were generated.
        processing_time_range: Expected operation base-time interval.
        training_parameter_jitter: Expected train-only jitter configuration.
        instance_directory: Optional root containing generated instances.

    Returns:
        Mapping from split name to validated instance-name lists.

    Raises:
        FileNotFoundError: If a configured instance is absent.
        ValueError: If configuration or stored instance metadata disagree.
    """
    from helper.stochastic_fjsp import (
        normalize_machine_profile_config,
        normalize_training_parameter_jitter,
    )

    expected_profile_config = normalize_machine_profile_config(
        machine_profile_config
    )
    jitter_config = normalize_training_parameter_jitter(
        training_parameter_jitter
    )
    if jitter_config["enabled"] and any(
        expected_profile_config["parameter_jitter"].values()
    ):
        raise ValueError(
            "Train-only jitter requires zero baseline parameter_jitter values."
        )
    jittered_profile_config = copy.deepcopy(expected_profile_config)
    if jitter_config["enabled"]:
        jittered_profile_config["parameter_jitter"] = dict(
            jitter_config["parameter_jitter"]
        )

    due_date_factors = [
        float(value) for value in due_date_config["factors"]
    ]
    if not due_date_factors or any(
        not math.isfinite(value) or value <= 0.0
        for value in due_date_factors
    ):
        raise ValueError("due_dates.factors must contain positive values.")

    specs = list(specs)
    if not specs:
        raise ValueError("Mindestens eine Instanzgröße muss konfiguriert sein.")

    instance_splits = {split_name: [] for split_name in SPLIT_NAMES}
    expected_specs = {}
    jittered_instance_names = set()
    for spec_index, spec in enumerate(specs):
        num_jobs = int(spec["num_jobs"])
        num_machines = int(spec["num_machines"])
        operation_min = min(spec["operations_per_job"])
        operation_max = max(spec["operations_per_job"])
        count = int(spec["count"])
        if min(num_jobs, num_machines, count) <= 0:
            raise ValueError(
                "Jobs, Maschinen, Operationen und Instanzanzahl müssen "
                "größer als 0 sein."
            )

        names = [
            f"i{num_jobs}_k{num_machines}_"
            f"o{operation_min}-{operation_max}_{instance_number}"
            for instance_number in range(1, count + 1)
        ]
        for name in names:
            expected_specs[name] = {
                "num_jobs": num_jobs,
                "num_machines": num_machines,
                "operations_per_job_min": operation_min,
                "operations_per_job_max": operation_max,
            }
        size_split = split_items(
            names,
            split_ratios=split_ratios,
            random_seed=int(random_seed) + spec_index,
        )
        if jitter_config["enabled"]:
            jitter_count = round(
                len(size_split["train"])
                * jitter_config["fraction_per_size"]
            )
            jittered_instance_names.update(random.Random(
                jitter_config["random_seed"] + spec_index
            ).sample(size_split["train"], jitter_count))
        for split_name in SPLIT_NAMES:
            instance_splits[split_name].extend(size_split[split_name])

    available_names = set(generated_instance_names(instance_directory))
    missing_names = sorted(set(expected_specs) - available_names)
    if missing_names:
        preview = ", ".join(missing_names[:5])
        remaining = len(missing_names) - min(len(missing_names), 5)
        suffix = f" und {remaining} weitere" if remaining else ""
        raise FileNotFoundError(
            "Konfigurierte Instanzen fehlen: "
            f"{preview}{suffix}. Setze workflow.create_instances einmal auf true."
        )

    for instance_name, expected in expected_specs.items():
        instance = load_generated_instance(
            instance_name,
            instance_directory=instance_directory,
        )
        expected_instance_profile_config = (
            jittered_profile_config
            if instance_name in jittered_instance_names
            else expected_profile_config
        )
        if (
            getattr(instance, "machine_profile_config", None)
            != expected_instance_profile_config
        ):
            raise ValueError(
                f"Gespeicherte Instanz {instance_name} verwendet andere "
                "Maschinenprofile. Setze workflow.create_instances einmal auf true."
            )
        expected_processing_range = tuple(map(int, processing_time_range))
        actual_processing_range = (
            int(instance.processing_time_per_ope_min),
            int(instance.processing_time_per_ope_max),
        )
        if actual_processing_range != expected_processing_range:
            raise ValueError(
                f"Gespeicherte Instanz {instance_name} verwendet einen "
                "anderen Bearbeitungszeitbereich. Setze "
                "workflow.create_instances einmal auf true."
            )
        actual_operations = [int(value) for value in instance.nums_operation]
        if (
            int(instance.num_jobs) != expected["num_jobs"]
            or int(instance.num_machines) != expected["num_machines"]
            or min(actual_operations) < expected["operations_per_job_min"]
            or max(actual_operations) > expected["operations_per_job_max"]
        ):
            raise ValueError(
                f"Gespeicherte Instanz {instance_name} passt nicht zur "
                "aktuellen Config. Setze workflow.create_instances einmal "
                "auf true."
            )

        expected_factor = due_date_factors[
            (int(instance.nb_instance) - 1) % len(due_date_factors)
        ]
        actual_factor = getattr(instance, "due_date_factor", None)
        if (
            actual_factor is None
            or not math.isclose(
                float(actual_factor),
                expected_factor,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise ValueError(
                f"Instance {instance_name} does not match the configured "
                "due-date factors and fixed total-work-content method. "
                "Regenerate the instances before running the pipeline."
            )

    return {
        split_name: sorted(instance_splits[split_name])
        for split_name in SPLIT_NAMES
    }
