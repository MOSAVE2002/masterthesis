import pickle
import random
import os
import sys
import types
import math
from pathlib import Path


_legacy_package = sys.modules.get("generator")
if _legacy_package is None:
    _legacy_package = types.ModuleType("generator")
    _legacy_package.__path__ = []
    sys.modules["generator"] = _legacy_package
sys.modules["generator.instance_generator"] = sys.modules[__name__]
_legacy_package.instance_generator = sys.modules[__name__]


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
SPLIT_CONFIG_KEYS = {
    "training": "train",
    "valid": "valid",
    "test": "test",
}
class FJSPData:
    def _set_default_nonlinear_parameters(self):
        """Install the scenario-free Weibull/repair model defaults."""
        self.weibull_beta = 2.0

    def _set_independent_machine_parameters(self, rng, ranges=None):
        """Draw cost and reliability independently for every machine."""
        from helper.stochastic_fjsp import (
            INDEPENDENT_GENERATION_MODEL,
            normalize_independent_machine_parameter_ranges,
        )

        normalized = normalize_independent_machine_parameter_ranges(ranges)
        self.instance_generation_model = INDEPENDENT_GENERATION_MODEL
        self.machine_parameter_ranges = normalized

        def sampled(name):
            lower, upper = normalized[name]
            return {
                machine: rng.uniform(lower, upper)
                for machine in range(self.num_machines)
            }

        self.machine_cost = sampled("hourly_cost")
        self.weibull_alpha = sampled("weibull_alpha")
        self.weibull_beta = sampled("weibull_beta")
        self.repair_rate = sampled("repair_rate")
        self.repair_duration = {
            machine: 1.0 / rate
            for machine, rate in self.repair_rate.items()
        }

    def _set_profile_machine_parameters(self, rng, config):
        from helper.stochastic_fjsp import (
            PROFILE_GENERATION_MODEL,
            normalize_machine_profile_config,
        )

        cfg = normalize_machine_profile_config(config)
        profile_names = list(cfg["profiles"])
        assignments = [
            profile_names[index % len(profile_names)]
            for index in range(self.num_machines)
        ]
        rng.shuffle(assignments)
        self.instance_generation_model = PROFILE_GENERATION_MODEL
        self.machine_profile_config = cfg
        self.machine_profiles = dict(enumerate(assignments))

        def jittered(machine, field):
            profile = cfg["profiles"][self.machine_profiles[machine]]
            width = cfg["parameter_jitter"][field]
            return profile[field] * rng.uniform(1.0 - width, 1.0 + width)

        machines = range(self.num_machines)
        self.machine_speed = {
            machine: jittered(machine, "speed")
            for machine in machines
        }
        self.machine_cost = {
            machine: jittered(machine, "cost_rate")
            for machine in machines
        }
        self.weibull_alpha = {
            machine: jittered(machine, "weibull_alpha")
            for machine in machines
        }
        self.repair_rate = {
            machine: jittered(machine, "repair_rate")
            for machine in machines
        }
        self.weibull_beta = {
            machine: float(
                cfg["profiles"][self.machine_profiles[machine]][
                    "weibull_beta"
                ]
            )
            for machine in machines
        }
        self.repair_duration = {
            machine: 1.0 / self.repair_rate[machine]
            for machine in machines
        }

        by_profile = {
            profile: [
                machine
                for machine in machines
                if self.machine_profiles[machine] == profile
            ]
            for profile in profile_names
        }
        noise_lower, noise_upper = cfg["operation_time_noise"]
        self.operation_base_time = {}
        self.eligible_machines = {}
        self.processing_times = {}
        self.nums_option, self.ope_machine, self.processing_time = [], [], []
        for operation in self.real_operations:
            base = rng.randint(
                self.processing_time_per_ope_min,
                self.processing_time_per_ope_max,
            )
            self.operation_base_time[operation] = float(base)
            available_profiles = [
                name for name in profile_names if by_profile[name]
            ]
            minimum = min(
                cfg["minimum_profile_classes_per_operation"],
                len(available_profiles),
            )
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
            self.nums_option.append(len(eligible))
            for machine in eligible:
                duration = max(
                    1,
                    int(math.ceil(
                        base
                        / self.machine_speed[machine]
                        * rng.uniform(noise_lower, noise_upper)
                    )),
                )
                self.processing_times[operation, machine] = duration
                self.ope_machine.append(machine)
                self.processing_time.append(duration)
        self.nums_options = sum(self.nums_option)
        self.num_machine_bias = [
            sum(self.nums_option[:index])
            for index in range(self.num_operations)
        ]
        self.processing_times_mean = [
            self.operation_base_time[operation]
            for operation in self.real_operations
        ]

    def _set_due_dates(self, due_date_config=None):
        """Assign reproducible job-specific total-work-content due dates."""
        due_cfg = dict(due_date_config or {})
        method = str(due_cfg.get("method", "total_work_content"))
        if method != "total_work_content":
            raise ValueError(
                "due_dates.method must be 'total_work_content'."
            )
        machine_aggregation = str(
            due_cfg.get("machine_aggregation", "mean")
        )
        if machine_aggregation != "mean":
            raise ValueError(
                "due_dates.machine_aggregation must be 'mean'."
            )
        assignment = str(due_cfg.get("assignment", "cyclic"))
        if assignment != "cyclic":
            raise ValueError("due_dates.assignment must be 'cyclic'.")
        factors = [
            float(value)
            for value in due_cfg.get(
                "factors", [1.25, 1.30, 1.40, 1.55]
            )
        ]
        if not factors or any(
            not math.isfinite(value) or value <= 0.0
            for value in factors
        ):
            raise ValueError("due_dates.factors must contain positive values.")
        factor = factors[(int(self.nb_instance) - 1) % len(factors)]
        work_content = {
            job: sum(
                sum(
                    float(self.processing_times[operation, machine])
                    for machine in self.eligible_machines[operation]
                ) / len(self.eligible_machines[operation])
                for operation in operations
            )
            for job, operations in self.jobs.items()
        }
        if any(value <= 0.0 for value in work_content.values()):
            raise ValueError("Every job must have positive total work content.")
        self.due_date_method = method
        self.due_date_machine_aggregation = machine_aggregation
        self.due_date_assignment = assignment
        self.due_date_factor = factor
        self.due_date_work_content = work_content
        self.due_dates = {
            job: float(math.ceil(factor * work_content[job]))
            for job in self.jobs
        }

    def _build_operation_metadata(self):
        """
        
        Build derived scheduling structures from the flat instance encoding.
        
        """
        self.jobs = {}
        self.eligible_machines = {}
        self.processing_times = {}
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

                global_op_idx = self.num_ope_bias[job_index] + local_operation_index
                num_options = self.nums_option[global_op_idx]
                machine_offset = self.num_machine_bias[global_op_idx]

                eligible = []
                for option_idx in range(num_options):
                    machine = self.ope_machine[machine_offset + option_idx]
                    processing_time = self.processing_time[machine_offset + option_idx]
                    eligible.append(machine)
                    self.processing_times[current_operation, machine] = processing_time

                self.eligible_machines[current_operation] = eligible
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
        num_operations = None,
        path = '../FJSP_Simulation/02_data/instances_text/',
        flag_same_operations = False,
        flag_save_file = True,
        processing_time_range=None,
        processing_time_deviation=0.2,
        machine_parameter_ranges=None,
        machine_profile_config=None,
        due_date_config=None,
        time_unit_minutes=1.0,
        random_source=None):

        """
        Initializes the FJSPInstanceGenerator with the specified parameters.

        """
        
        self.nb_instance = nb_instance
        rng = random_source if random_source is not None else random

        if num_operations is None:
            num_operations = []
        self.path = path
        
        # flags
        self.flag_same_operations = flag_same_operations # if True, all jobs will have the same number of operations, determined by the average of operations_per_job_min and operations_per_job_max. If False, the number of operations per job will be randomly generated within the specified range.
        self.flag_save_file = flag_save_file 

        # instance parameters
        self.num_jobs = num_jobs
        self.num_machines = num_machines
        self.time_unit_minutes = float(time_unit_minutes)
        if self.time_unit_minutes <= 0.0:
            raise ValueError("time_unit_minutes must be positive.")

        # Operations per job parameters
        self.ope_per_job_min = operations_per_job_min
        self.ope_per_job_max = operations_per_job_max
        
        # Machine options per operation parameters
        self.nums_operation = num_operations
        self.machine_per_ope_min = 1 # at least one machine must be able to process each operation
        self.machine_per_ope_max = num_machines
        
        # Processing time parameters
        #TODO Operation time sollte einstellbar sein -> gute Values finden
        processing_time_range = processing_time_range or (1, 10)
        if len(processing_time_range) != 2:
            raise ValueError("processing_time_range must contain two values.")
        self.processing_time_per_ope_min = int(processing_time_range[0])
        self.processing_time_per_ope_max = int(processing_time_range[1])
        self.proctime_deviation = float(processing_time_deviation)
        if (
            self.processing_time_per_ope_min <= 0
            or self.processing_time_per_ope_min
            > self.processing_time_per_ope_max
        ):
            raise ValueError("Invalid positive processing-time range.")
        if not 0.0 <= self.proctime_deviation < 1.0:
            raise ValueError("processing_time_deviation must lie in [0, 1).")
        self._set_default_nonlinear_parameters()

         # Instance Name
         #TODO mit zfill() sieht besser aus, wenn ich nacher mti tausenden datein arbeite, sieht das übersichtlicher aus
        self.instance_name = (
            f"i{self.num_jobs}_k{self.num_machines}_"
            f"o{self.ope_per_job_min}-{self.ope_per_job_max}_{self.nb_instance}"
        )


        if self.nums_operation:
            pass
        elif self.ope_per_job_min == self.ope_per_job_max:
            self.nums_operation = [self.ope_per_job_min for _ in range(self.num_jobs)]
        elif self.flag_same_operations:
            num_operations_per_job = round((self.ope_per_job_min + self.ope_per_job_max) / 2)
            self.nums_operation = [num_operations_per_job for _ in range(self.num_jobs)]
        else:
            self.nums_operation = [rng.randint(self.ope_per_job_min, self.ope_per_job_max) for _ in range(self.num_jobs)]
        self.num_operations = sum(self.nums_operation) # Amount of operations

        self.nums_option = [rng.randint(1, self.num_machines) for _ in range(self.num_operations)]
        self.nums_options = sum(self.nums_option)

        self.ope_machine = []
        for val in self.nums_option:
            self.ope_machine = self.ope_machine + sorted(rng.sample(range(self.num_machines), val)) # flache Liste von möglichen Maschinen

        self.processing_time = []
        self.processing_times_mean = [rng.randint(self.processing_time_per_ope_min, self.processing_time_per_ope_max) for _ in range(self.num_operations)]
        
        for i in range(len(self.nums_option)):
            low_bound = max(self.processing_time_per_ope_min, round(self.processing_times_mean[i] * (1 - self.proctime_deviation))) #
            high_bound = min(self.processing_time_per_ope_max, round(self.processing_times_mean[i] * (1 + self.proctime_deviation))) #
            process_time_ope = [rng.randint(low_bound, high_bound) for _ in range(self.nums_option[i])]
            self.processing_time = self.processing_time + process_time_ope 

        self.num_ope_bias = [sum(self.nums_operation[0:i]) for i in range(self.num_jobs)] # kumulierte Summe als Liste
        self.num_machine_bias = [sum(self.nums_option[0:i]) for i in range(self.num_operations)] # kumulierte Summe als Liste

       

        line0 = '{0}\t{1}\t{2}\n'.format(self.num_jobs, self.num_machines, self.nums_options / self.num_operations)
        lines = []
        lines_doc = []
        lines.append(line0)
       
        lines_doc.append('{0}\t{1}\t{2}\n'.format(self.num_jobs, self.num_machines, self.nums_options / self.num_operations))
    
        idx = self.nb_instance
        for i in range(self.num_jobs):
            flag = 0
            flag_time = 0
            flag_new_ope = 1
            idx_ope = -1 
            idx_machine = 0
            line = []
            option_max = sum(self.nums_option[self.num_ope_bias[i]:self.num_ope_bias[i]+self.nums_operation[i]]) # da sind alle 
            idx_option = 0
            while True:
                if flag == 0:
                    line.append(self.nums_operation[i])
                    flag += 1
                elif flag == flag_new_ope:
                    idx_ope += 1
                    idx_machine = 0
                    flag_new_ope += self.nums_option[self.num_ope_bias[i]+idx_ope] * 2 + 1
                    line.append(self.nums_option[self.num_ope_bias[i]+idx_ope])
                    flag += 1
                elif flag_time == 0:
                    line.append(self.ope_machine[self.num_machine_bias[self.num_ope_bias[i]+idx_ope] + idx_machine])
                    flag += 1
                    flag_time = 1
                else:
                    line.append(self.processing_time[self.num_machine_bias[self.num_ope_bias[i]+idx_ope] + idx_machine])
                    flag += 1
                    flag_time = 0
                    idx_option += 1
                    idx_machine+= 1
                if idx_option == option_max:
                    str_line = " ".join([str(val) for val in line])
                    lines.append(str_line + '\n')
                    lines_doc.append(str_line)
                    break
        lines.append('\n')

        self.lines = lines
        self._build_operation_metadata()
        if machine_profile_config is not None:
            self._set_profile_machine_parameters(rng, machine_profile_config)
        else:
            self._set_independent_machine_parameters(
                rng, machine_parameter_ranges
            )
        self._set_due_dates(due_date_config)
        from helper.stochastic_fjsp import ensure_stochastic_parameters
        ensure_stochastic_parameters(self)
        
        # Text file zum einfacheren Lesen lassen
        if self.flag_save_file:
            if not os.path.exists(self.path):
                os.makedirs(self.path)
            document = open(self.path + 'i{0}_j{1}_{2}.fjsp'.format(self.num_jobs, self.num_machines, str.zfill(str(idx),3)),'w')
            for i in range(len(lines_doc)):
                print(lines_doc[i], file=document)
            document.close()
    def __repr__(self):
        return f"FJSP({self.instance_name})"

def generate_instances(
    nb_instances,
    num_jobs,
    num_machines,
    operations_per_job_min,
    operations_per_job_max,
    num_operations,
    split_ratios=None,
    random_seed=42,
    output_directory=None,
    processing_time_range=None,
    processing_time_deviation=0.2,
    machine_parameter_ranges=None,
    machine_profile_config=None,
    due_date_config=None,
    time_unit_minutes=1.0,
):
    """
    Generate multiple instances 

    Parameters:

    Returns the paths of the individual pickle files grouped by split.

    No text files are created.
    """
    generation_rng = random.Random(int(random_seed))
    instances = [
        FJSPData(
            nb_instance=instance_nb,
            num_jobs=num_jobs,
            num_machines=num_machines,
            operations_per_job_min=operations_per_job_min,
            operations_per_job_max=operations_per_job_max,
            num_operations=num_operations,
            flag_save_file=False,
            processing_time_range=processing_time_range,
            processing_time_deviation=processing_time_deviation,
            machine_parameter_ranges=machine_parameter_ranges,
            machine_profile_config=machine_profile_config,
            due_date_config=due_date_config,
            time_unit_minutes=time_unit_minutes,
            random_source=generation_rng,
        )
        for instance_nb in range(1, nb_instances + 1)
    ]
    return save_instance_splits(
        instances,
        split_ratios=split_ratios,
        random_seed=random_seed,
        output_directory=output_directory,
    )


def generate_instance_specs(
    specs,
    split_ratios=None,
    random_seed=42,
    output_directory=None,
    processing_time_range=None,
    processing_time_deviation=0.2,
    machine_parameter_ranges=None,
    machine_profile_config=None,
    due_date_config=None,
    time_unit_minutes=1.0,
):
    """Generate configured sizes and split every size independently."""
    split_instances = {split_name: [] for split_name in SPLIT_NAMES}
    generation_rng = random.Random(int(random_seed))
    for spec_index, spec in enumerate(specs):
        size_instances = [
            FJSPData(
                nb_instance=instance_nb,
                num_jobs=spec["num_jobs"],
                num_machines=spec["num_machines"],
                operations_per_job_min=min(spec["operations_per_job"]),
                operations_per_job_max=max(spec["operations_per_job"]),
                num_operations=None,
                flag_save_file=False,
                processing_time_range=processing_time_range,
                processing_time_deviation=processing_time_deviation,
                machine_parameter_ranges=machine_parameter_ranges,
                machine_profile_config=machine_profile_config,
                due_date_config=due_date_config,
                time_unit_minutes=time_unit_minutes,
                random_source=generation_rng,
            )
            for instance_nb in range(1, spec["count"] + 1)
        ]
        size_split = split_items(
            size_instances,
            split_ratios=split_ratios,
            random_seed=int(random_seed) + spec_index,
        )
        for split_name in SPLIT_NAMES:
            split_instances[split_name].extend(size_split[split_name])
    return save_pre_split_instances(
        split_instances,
        output_directory=output_directory,
    )


def generate_evaluation_instance_specs(
    specs,
    random_seed=42,
    output_directory=None,
    processing_time_range=None,
    processing_time_deviation=0.2,
    machine_parameter_ranges=None,
    machine_profile_config=None,
    due_date_config=None,
    instance_postprocessor=None,
    instance_name_suffix=None,
    time_unit_minutes=1.0,
):
    """Generate a flat holdout set without train/valid/test subdirectories."""
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
                num_operations=None,
                flag_save_file=False,
                processing_time_range=processing_time_range,
                processing_time_deviation=processing_time_deviation,
                machine_parameter_ranges=machine_parameter_ranges,
                machine_profile_config=machine_profile_config,
                due_date_config=due_date_config,
                time_unit_minutes=time_unit_minutes,
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
        raise ValueError("Evaluation instance names must be unique.")

    output_directory.mkdir(parents=True, exist_ok=True)
    removed = []
    for existing_path in output_directory.glob("*.pkl"):
        existing_path.unlink()
        removed.append(existing_path)
    # Remove the obsolete split layout only inside this evaluation tier.
    for split_directory_name in SPLIT_DIRECTORIES.values():
        split_directory = output_directory / split_directory_name
        if not split_directory.exists():
            continue
        for existing_path in split_directory.glob("*.pkl"):
            existing_path.unlink()
            removed.append(existing_path)
        if not any(split_directory.iterdir()):
            split_directory.rmdir()

    paths = []
    for instance in instances:
        path = output_directory / f"{instance.instance_name}.pkl"
        with path.open("wb") as output_file:
            pickle.dump(instance, output_file)
        paths.append(path)
    print(
        f"Replaced {len(removed)} old and wrote {len(paths)} evaluation "
        f"instances to: {output_directory}"
    )
    return paths


def save_pre_split_instances(
    split_instances,
    output_directory=None,
):
    """Persist an existing disjoint train/valid/test instance assignment."""
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
    paths = {}
    for split_name, values in normalized.items():
        split_directory = output_directory / SPLIT_DIRECTORIES[split_name]
        split_directory.mkdir(parents=True, exist_ok=True)
        paths[split_name] = []
        for instance in values:
            path = split_directory / f"{instance.instance_name}.pkl"
            with path.open("wb") as output_file:
                pickle.dump(instance, output_file)
            paths[split_name].append(path)
        print(
            f"Wrote {len(values)} {SPLIT_DIRECTORIES[split_name]} instances to: "
            f"{split_directory}"
        )
    return paths


def _normalized_split_ratios(split_ratios=None):
    ratios = dict(split_ratios or {"train": 0.8, "valid": 0.1, "test": 0.1})
    unknown = set(ratios) - set(SPLIT_NAMES)
    if unknown:
        raise ValueError(f"Unknown dataset splits: {sorted(unknown)}")
    values = {name: float(ratios.get(name, 0.0)) for name in SPLIT_NAMES}
    if any(value < 0.0 for value in values.values()):
        raise ValueError("Split ratios must be nonnegative.")
    total = sum(values.values())
    if total <= 0.0:
        raise ValueError("At least one split ratio must be positive.")
    return {name: value / total for name, value in values.items()}


def split_items(items, split_ratios=None, random_seed=42):
    """Shuffle and split items deterministically while preserving every item."""
    ratios = _normalized_split_ratios(split_ratios)
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
    """Remove every previously generated pickle from all dataset splits."""
    output_directory = Path(output_directory or INSTANCE_DIRECTORY)
    removed_paths = []
    for split_directory_name in SPLIT_DIRECTORIES.values():
        split_directory = output_directory / split_directory_name
        split_directory.mkdir(parents=True, exist_ok=True)
        for existing_path in split_directory.glob("*.pkl"):
            existing_path.unlink()
            removed_paths.append(existing_path)
    print(
        f"Removed {len(removed_paths)} old instances from: "
        f"{output_directory}"
    )
    return removed_paths


def save_instance_splits(
    instances,
    split_ratios=None,
    random_seed=42,
    output_directory=None,
):
    """Write every generated instance to its own split-specific pickle file."""
    instances = list(instances)
    split_instances = split_items(instances, split_ratios, random_seed)
    return save_pre_split_instances(
        split_instances,
        output_directory=output_directory,
    )


def load_generated_instance(instance_name, instance_directory=None):
    """Load an instance from a flat evaluation tier or a dataset split."""
    from helper.stochastic_fjsp import ensure_stochastic_parameters

    def upgraded(path):
        with path.open("rb") as instance_file:
            instance = pickle.load(instance_file)
        return ensure_stochastic_parameters(instance)

    directory = Path(instance_directory or INSTANCE_DIRECTORY)
    direct_path = directory / f"{instance_name}.pkl"
    if direct_path.exists():
        return upgraded(direct_path)
    for split_name in SPLIT_NAMES:
        instance_path = (
            directory
            / SPLIT_DIRECTORIES[split_name]
            / f"{instance_name}.pkl"
        )
        if instance_path.exists():
            return upgraded(instance_path)

    legacy_directory = ROOT_DIR / "02_data" / "fjsp_instances"
    legacy_path = legacy_directory / f"{instance_name}.fjsp"
    if legacy_path.exists():
        return upgraded(legacy_path)
    raise FileNotFoundError(f"Generated instance not found: {instance_name}")


def generated_instance_names(instance_directory=None):
    directory = Path(instance_directory or INSTANCE_DIRECTORY)
    names = {
        path.stem
        for split_name in SPLIT_NAMES
        for path in (directory / SPLIT_DIRECTORIES[split_name]).glob("*.pkl")
    }
    names.update(path.stem for path in directory.glob("*.pkl"))
    return sorted(names)


def generated_instance_names_by_split(instance_directory=None):
    directory = Path(instance_directory or INSTANCE_DIRECTORY)
    return {
        split_name: sorted(
            path.stem
            for path in (
                directory / SPLIT_DIRECTORIES[split_name]
            ).glob("*.pkl")
        )
        for split_name in SPLIT_NAMES
    }


def configured_instance_names_by_split(
    specs,
    split_ratios=None,
    random_seed=42,
    instance_directory=None,
    processing_time_range=None,
    processing_time_deviation=None,
    machine_parameter_ranges=None,
    machine_profile_config=None,
    time_unit_minutes=None,
):
    """Select and validate exactly the instances requested by the config."""
    from helper.stochastic_fjsp import (
        INDEPENDENT_GENERATION_MODEL,
        PROFILE_GENERATION_MODEL,
        normalize_independent_machine_parameter_ranges,
        normalize_machine_profile_config,
    )

    expected_machine_ranges = (
        normalize_independent_machine_parameter_ranges(
            machine_parameter_ranges
        )
        if machine_parameter_ranges is not None else None
    )
    expected_profile_config = (
        normalize_machine_profile_config(machine_profile_config)
        if machine_profile_config is not None else None
    )

    specs = list(specs)
    if not specs:
        raise ValueError("Mindestens eine Instanzgröße muss konfiguriert sein.")

    instance_splits = {split_name: [] for split_name in SPLIT_NAMES}
    expected_specs = {}
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
        if (
            time_unit_minutes is not None
            and not math.isclose(
                float(getattr(instance, "time_unit_minutes", 1.0)),
                float(time_unit_minutes),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise ValueError(
                f"Gespeicherte Instanz {instance_name} verwendet eine andere "
                "Zeiteinheit. Setze workflow.create_instances einmal auf true."
            )
        expected_model = (
            PROFILE_GENERATION_MODEL
            if expected_profile_config is not None
            else INDEPENDENT_GENERATION_MODEL
        )
        if getattr(instance, "instance_generation_model", None) != expected_model:
            raise ValueError(
                f"Gespeicherte Instanz {instance_name} verwendet noch das "
                "falsche Generatormodell. Setze "
                "workflow.create_instances einmal auf true."
            )
        if (
            expected_profile_config is not None
            and getattr(instance, "machine_profile_config", None)
            != expected_profile_config
        ):
            raise ValueError(
                f"Gespeicherte Instanz {instance_name} verwendet andere "
                "Maschinenprofile. Setze workflow.create_instances einmal auf true."
            )
        if (
            expected_machine_ranges is not None
            and getattr(instance, "machine_parameter_ranges", None)
            != expected_machine_ranges
        ):
            raise ValueError(
                f"Gespeicherte Instanz {instance_name} verwendet andere "
                "Maschinenparameterbereiche. Setze "
                "workflow.create_instances einmal auf true."
            )
        if processing_time_range is not None:
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
        if (
            processing_time_deviation is not None
            and not math.isclose(
                float(instance.proctime_deviation),
                float(processing_time_deviation),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise ValueError(
                f"Gespeicherte Instanz {instance_name} verwendet eine "
                "andere Bearbeitungszeitabweichung. Setze "
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

    return {
        split_name: sorted(instance_splits[split_name])
        for split_name in SPLIT_NAMES
    }


def selected_instance_splits(instance_splits, generate_splits=None):
    """Return only the train/valid/test splits enabled in the config."""
    if generate_splits is None:
        return {
            split_name: list(names)
            for split_name, names in instance_splits.items()
        }
    if not isinstance(generate_splits, dict):
        raise ValueError("generate_splits muss ein JSON-Objekt sein.")

    unknown = set(generate_splits) - set(SPLIT_CONFIG_KEYS)
    if unknown:
        raise ValueError(
            "Unbekannte generate_splits-Einträge: "
            f"{sorted(unknown)}."
        )

    selected = {}
    for config_key, split_name in SPLIT_CONFIG_KEYS.items():
        raw_value = generate_splits.get(config_key, False)
        if isinstance(raw_value, str):
            enabled = raw_value.strip().lower() in {
                "1", "true", "yes", "on"
            }
        else:
            enabled = bool(raw_value)
        if enabled and split_name in instance_splits:
            selected[split_name] = list(instance_splits[split_name])

    if not selected:
        raise ValueError(
            "Mindestens ein Eintrag in generate_splits muss true sein."
        )
    return selected


if __name__ == "__main__":
    
    generate_instances(nb_instances =5, num_jobs=3, num_machines=3, operations_per_job_min=1, operations_per_job_max=2, num_operations=None)



       
