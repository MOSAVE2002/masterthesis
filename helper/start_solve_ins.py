import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from Gurobi.solve_instances_fjsp import solveModel
from Algorithm.HA import load_instance, hybrid_ga_ts, PARAMS


def main(**kwargs):
    solver_kwargs = dict(kwargs)

    num_jobs     = solver_kwargs.pop("num_jobs",     20)
    num_machines = solver_kwargs.pop("num_machines", 10)
    instance_nb  = solver_kwargs.pop("instance_nb",  1)
    solver       = solver_kwargs.pop("solver","ha").lower()

    instance_name = f"i{num_jobs}_k{num_machines}_{instance_nb}"

    if solver == "gurobi":
        # Forward remaining kwargs as Gurobi parameters (TimeLimit, Threads, …)
        solveModel(instance_name=instance_name, **solver_kwargs)

    elif solver == "ha":
        import random

        seed = solver_kwargs.pop("seed", None)
        if seed is not None:
            random.seed(seed)

        # Collect HA parameters, fall back to defaults from PARAMS
        ha_kwargs = {
            "pop_size":     int(solver_kwargs.pop("pop_size",     PARAMS["pop_size"])),
            "max_gen":      int(solver_kwargs.pop("max_gen",      PARAMS["max_gen"])),
            "max_stagnant": int(solver_kwargs.pop("max_stagnant", PARAMS["max_stagnant"])),
            "pc":           float(solver_kwargs.pop("pc",         PARAMS["pc"])),
            "pm":           float(solver_kwargs.pop("pm",         PARAMS["pm"])),
            "pr":           float(solver_kwargs.pop("pr",         PARAMS["pr"])),
            "tabu_len":     int(solver_kwargs.pop("tabu_len",     PARAMS["tabu_len"])),
            "ts_iter_base": int(solver_kwargs.pop("ts_iter_base", PARAMS["ts_iter_base"])),
        }

        # Warn about unrecognised kwargs
        for key in solver_kwargs:
            print(f"Warning: Unknown argument '{key}' ignored for HA solver.")

        pt, _ = load_instance(instance_name)

        t0   = time.time()
        best = hybrid_ga_ts(pt, **ha_kwargs)
        elapsed = time.time() - t0

        print(f"\n=== HA Result ===")
        print(f"Instance : {instance_name}")
        print(f"Makespan : {best.fitness}")
        print(f"Time     : {elapsed:.2f} s")

    else:
        print(f"Error: Unknown solver '{solver}'. Use 'gurobi' or 'ha'.")


if __name__ == '__main__':
    main()
