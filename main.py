import json
from pathlib import Path

from helper.start_solve_ins import solve_instances_with_solver
from generator.instance_generator import generate_instances

ROOT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = ROOT_DIR / "config.json" #Pfad bitte auf Config Path einstellen



# Bitte alles in die Config Datei eintragen

def main():

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    # Get instance parameters
    instance_cfg = config.get("instance_parameters", {})

    num_jobs_list = instance_cfg.get("num_jobs", [])
    num_machines_list = instance_cfg.get("num_machines", [])
    ops_per_job = instance_cfg.get("num_operations_per_job", [])
    nb_instances = int(instance_cfg.get("nb_instances", 1))
    instance_names = instance_cfg.get("instance_names", [])

    # Get programm parameters
    program_cfg = config.get("programm_settings", {})
    solvers = program_cfg.get("solver", [])

    # Get solver parameters
    gurobi_cfg = config.get("gurobi_params", {})
    scip_cfg = config.get("scip_params", {})
    random_fixed_cfg = config.get("random_fixed_y", {})
    fixed_y_cfg = {}
   
    
    # Create fixed Values
    if random_fixed_cfg:
        fixed_y_cfg = {
            "create_fixed_y": _as_bool(
                random_fixed_cfg.get(
                    "create_fixed_y",
                    random_fixed_cfg.get("enabled", True),
                )
            ),
            "amount_of_samples_per_instance": random_fixed_cfg.get(
                "amount_of_samples_per_instance", 1
            ),
            "random_seed": random_fixed_cfg.get("random_seed", 0),
            "save_assignments": _as_bool(
                random_fixed_cfg.get("save_assignments", True)
            ),
        }

    if _as_bool(program_cfg.get("create_ins", False)):
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

    if _as_bool(program_cfg.get("solve_ins", False)):
        # You can select specific instances to check model
        if instance_names:
            for instance_name in instance_names:
                for solver in solvers:
                    solver_specific_params = (
                        gurobi_cfg if solver.lower() == "gurobi" else scip_cfg
                    )
                    fixed_y_params = fixed_y_cfg if solver.lower() == "gurobi" else {}
                    solve_instances_with_solver(
                        instance_name=instance_name,
                        solver=solver,
                        **solver_specific_params,
                        **fixed_y_params,
                    )
        
        else:
            for num_jobs in num_jobs_list:
                for num_machines in num_machines_list:
                    for instance_nb in range(1, nb_instances + 1):
                        instance_name = f"i{num_jobs}_k{num_machines}_{instance_nb}"
                        for solver in solvers:
                            solver_specific_params = (
                                gurobi_cfg if solver.lower() == "gurobi" else scip_cfg
                            )
                            fixed_y_params = fixed_y_cfg if solver.lower() == "gurobi" else {}
                            solve_instances_with_solver(
                                instance_name=instance_name,
                                solver=solver,
                                **solver_specific_params,
                                **fixed_y_params,
                            )

def _as_bool(value):
    return value if isinstance(value, bool) else str(value).lower() == "true"


if __name__ == "__main__":
    main()
