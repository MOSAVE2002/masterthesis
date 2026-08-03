import argparse
import importlib
import pickle
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

FJSPData = importlib.import_module("01_generator.instance_generator").FJSPData

# This file is used to safe literature instances as pickle data


def parse_text_instance(path: Path) -> tuple[int, int, list[list[list[tuple[int, int]]]], list[str]]:
    """
    Parse a text FJSP instance into a nested structure:
    jobs -> operations -> machine alternatives -> (machine, processing_time).

    Supported headers:
    - `n_jobs n_machines`
    - `n_jobs n_machines avg_options`
    """
    raw_lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not raw_lines:
        raise ValueError(f"Instance file is empty: {path}")

    header = raw_lines[0].split()
    if len(header) not in (2, 3):
        raise ValueError(
            "First line must contain `num_jobs num_machines` or "
            "`num_jobs num_machines avg_options`."
        )

    num_jobs = int(header[0])
    num_machines = int(header[1])
    job_lines = raw_lines[1:]

    if len(job_lines) != num_jobs:
        raise ValueError(
            f"Header says {num_jobs} jobs, but file contains {len(job_lines)} job lines."
        )

    jobs: list[list[list[tuple[int, int]]]] = []
    for job_idx, line in enumerate(job_lines):
        tokens = [int(token) for token in line.split()]
        if not tokens:
            raise ValueError(f"Empty job line for job {job_idx}.")

        idx = 0
        num_operations = tokens[idx]
        idx += 1
        operations: list[list[tuple[int, int]]] = []

        for op_idx in range(num_operations):
            if idx >= len(tokens):
                raise ValueError(f"Unexpected end of line in job {job_idx}, op {op_idx}.")
            num_options = tokens[idx]
            idx += 1
            alts: list[tuple[int, int]] = []
            for _ in range(num_options):
                if idx + 1 >= len(tokens):
                    raise ValueError(f"Missing machine/time pair in job {job_idx}, op {op_idx}.")
                machine = tokens[idx]
                duration = tokens[idx + 1]
                idx += 2
                alts.append((machine, duration))
            operations.append(alts)

        if idx != len(tokens):
            raise ValueError(
                f"Job line {job_idx} contains trailing values after parsing: {tokens[idx:]}"
            )
        jobs.append(operations)

    return num_jobs, num_machines, jobs, raw_lines


def build_fjsp_from_operations(
    nb_instance: int,
    num_jobs: int,
    num_machines: int,
    jobs: list[list[list[tuple[int, int]]]],
    source_lines: list[str],
    output_name: str | None = None,
) -> FJSPData:
    """Create an FJSPData object with the same attribute layout as instance_generator.py."""
    instance = FJSPData.__new__(FJSPData)

    nums_operation = [len(job_ops) for job_ops in jobs]
    num_operations = sum(nums_operation)
    nums_option: list[int] = []
    ope_machine: list[int] = []
    processing_time: list[int] = []

    for job_ops in jobs:
        for alts in job_ops:
            nums_option.append(len(alts))
            for machine, duration in alts:
                ope_machine.append(machine)
                processing_time.append(duration)

    instance.nb_instance = nb_instance
    instance.path = str(ROOT_DIR / "02_data" / "instances_text") + "/"
    instance.flag_same_operations = False
    instance.flag_save_file = False

    instance.num_jobs = num_jobs
    instance.num_machines = num_machines
    instance.ope_per_job_min = min(nums_operation) if nums_operation else 0
    instance.ope_per_job_max = max(nums_operation) if nums_operation else 0
    instance.machine_per_ope_min = min(nums_option) if nums_option else 0
    instance.machine_per_ope_max = max(nums_option) if nums_option else 0
    instance.processing_time_per_ope_min = min(processing_time) if processing_time else 0
    instance.processing_time_per_ope_max = max(processing_time) if processing_time else 0
    instance.proctime_deviation = 0.0
    instance._set_default_nonlinear_parameters()

    instance.nums_operation = nums_operation
    instance.num_operations = num_operations
    instance.nums_option = nums_option
    instance.nums_options = sum(nums_option)
    instance.ope_machine = ope_machine
    instance.processing_time = processing_time
    instance.processing_times_mean = []
    instance.num_ope_bias = [sum(nums_operation[:i]) for i in range(num_jobs)]
    instance.num_machine_bias = [sum(nums_option[:i]) for i in range(num_operations)]

    default_name = f"i{num_jobs}_k{num_machines}_{nb_instance}"
    instance.instance_name = output_name or default_name
    instance.lines = [line + "\n" for line in source_lines] + ["\n"]
    instance._build_operation_metadata()

    return instance


def convert_text_to_pickle(
    input_path: Path,
    output_path: Path | None = None,
    nb_instance: int = 1,
    instance_name: str | None = None,
) -> Path:
    num_jobs, num_machines, jobs, raw_lines = parse_text_instance(input_path)
    instance = build_fjsp_from_operations(
        nb_instance=nb_instance,
        num_jobs=num_jobs,
        num_machines=num_machines,
        jobs=jobs,
        source_lines=raw_lines,
        output_name=instance_name,
    )

    if output_path is None:
        output_path = ROOT_DIR / "02_data" / "fjsp_instances" / f"{instance.instance_name}.fjsp"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump(instance, f)
    return output_path


def convert_directory_to_pickles(
    input_dir: Path,
    output_dir: Path | None = None,
) -> list[Path]:
    """
    Convert all .txt files in a directory to .fjsp pickle files.

    Each pickle is stored under the text file stem, e.g. `k3.txt` -> `k3.fjsp`.
    This naming is the easiest way to launch literature instances later via
    `instance_name="k3"` in helper/start_solve_ins.py.
    """
    if output_dir is None:
        output_dir = ROOT_DIR / "02_data" / "fjsp_instances"

    output_dir.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    txt_files = sorted(input_dir.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in {input_dir}")

    for idx, txt_path in enumerate(txt_files, start=1):
        instance_name = txt_path.stem
        output_path = output_dir / f"{instance_name}.fjsp"
        created.append(
            convert_text_to_pickle(
                input_path=txt_path,
                output_path=output_path,
                nb_instance=idx,
                instance_name=instance_name,
            )
        )
    return created


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a text FJSP instance into the pickle format used by instance_generator.py."
    )
    parser.add_argument(
        "input_path",
        type=Path,
        nargs="?",
        default=ROOT_DIR / "02_data" / "fjsp_literature_instances",
        help="Path to a text FJSP instance or a directory of .txt instances.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Optional output .fjsp path for single-file mode.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT_DIR / "02_data" / "fjsp_instances",
        help="Output directory for directory mode. Defaults to 02_data/fjsp_instances",
    )
    parser.add_argument(
        "--instance-name",
        type=str,
        default=None,
        help="Optional instance name stored in the pickle object.",
    )
    parser.add_argument(
        "--instance-nb",
        type=int,
        default=1,
        help="Instance number used for the default name i<jobs>_k<machines>_<nb>.",
    )
    args = parser.parse_args()

    if args.input_path.is_dir():
        created = convert_directory_to_pickles(
            input_dir=args.input_path,
            output_dir=args.output_dir,
        )
        print(f"Converted {len(created)} text instances into: {args.output_dir}")
        for path in created:
            print(path)
    else:
        output_path = convert_text_to_pickle(
            input_path=args.input_path,
            output_path=args.output_path,
            nb_instance=args.instance_nb,
            instance_name=args.instance_name,
        )
        print(f"Pickle instance written to: {output_path}")


if __name__ == "__main__":
    main()
