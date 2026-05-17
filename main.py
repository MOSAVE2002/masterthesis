import json
from pathlib import Path

from helper.start_solve_ins import solve_instances_with_solver
from generator.instance_generator import generate_instances

ROOT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = ROOT_DIR / "config.json" #Pfad bitte auf Config Path einstellen


def _as_bool(value):
    return value if isinstance(value, bool) else str(value).lower() == "true"


def main():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    instance_cfg = config.get("instance_parameters", {})
    program_cfg = config.get("programm_settings", {})
    params = config.get("params", {})

    num_jobs_list = instance_cfg.get("num_jobs", [])
    num_machines_list = instance_cfg.get("num_machines", [])
    ops_per_job = instance_cfg.get("num_operations_per_job", [])
    nb_instances = int(instance_cfg.get("nb_instances", 1))
    solvers = program_cfg.get("solver", [])

    if _as_bool(program_cfg.get("create_ins", True)):
        for num_jobs in num_jobs_list:
            for num_machines in num_machines_list:
                generate_instances(
                    nb_instances=nb_instances,
                    num_jobs=num_jobs,
                    num_machines=num_machines,
                    operations_per_job_min=min(ops_per_job),
                    operations_per_job_max=max(ops_per_job),
                    num_operations=None,
                )

    if _as_bool(program_cfg.get("solve_ins", True)):
        for num_jobs in num_jobs_list:
            for num_machines in num_machines_list:
                for instance_nb in range(1, nb_instances + 1):
                    instance_name = f"i{num_jobs}_k{num_machines}_{instance_nb}"
                    for solver in solvers:
                        solve_instances_with_solver(instance_name=instance_name, solver=solver, **params)


if __name__ == "__main__":
    main()
