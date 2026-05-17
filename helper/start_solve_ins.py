import os
import subprocess
import random
import sys
import tempfile
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from Gurobi.solve_instances_fjsp import solveModel
from Algorithm.HA import load_instance, hybrid_ga_ts, export_pt_to_text, PARAMS

HA_CPP_BINARY = ROOT_DIR / "Algorithm" / "ha_solver"


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def solve_instances_with_solver(**kwargs):
    solver_kwargs = dict(kwargs)

    instance_name = solver_kwargs.pop("instance_name", "lao7")
    num_jobs      = solver_kwargs.pop("num_jobs",     20)
    num_machines  = solver_kwargs.pop("num_machines", 10)
    instance_nb   = solver_kwargs.pop("instance_nb",  1)
    solver        = solver_kwargs.pop("solver", "ha_cpp").lower()

    if instance_name is None:
        instance_name = f"i{num_jobs}_k{num_machines}_{instance_nb}"

    if solver == "gurobi":
        solveModel(instance_name=instance_name, **solver_kwargs)

    elif solver == "ha_cpp":
        if not HA_CPP_BINARY.exists():
            print(f"Error: C++ binary not found at {HA_CPP_BINARY}")
            print("Run:  cd Algorithm && make")
            return

        pop_size     = solver_kwargs.pop("pop_size", None)
        max_gen      = solver_kwargs.pop("max_gen", None)
        max_stagnant = solver_kwargs.pop("max_stagnant", PARAMS["max_stagnant"])
        ts_iter_base = solver_kwargs.pop("ts_iter_base", None)
        seed         = solver_kwargs.pop("seed", None)
        paper_mode   = _as_bool(solver_kwargs.pop("paper_mode", False))
        fast_small   = _as_bool(solver_kwargs.pop("fast_small", False)) # not important

        for key in solver_kwargs:
            print(f"Warning: Unknown argument '{key}' ignored for solver.")

        pt, _ = load_instance(instance_name)

        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w") as f:
            tmp_path = f.name
        try:
            export_pt_to_text(pt, tmp_path)

            cmd = [
                str(HA_CPP_BINARY), tmp_path,
            ]
            if pop_size is not None:
                cmd += ["--pop-size", str(int(pop_size))]
            if max_gen is not None:
                cmd += ["--max-gen", str(int(max_gen))]
            if max_stagnant is not None:
                cmd += ["--stagnant", str(int(max_stagnant))]
            if ts_iter_base is not None:
                cmd += ["--ts-base", str(int(ts_iter_base))]
            if seed is not None:
                cmd += ["--seed", str(seed)]
            if paper_mode:
                cmd += ["--paper-mode"]
            if fast_small:
                cmd += ["--fast-small"]

            t0     = time.time()
            result = subprocess.run(cmd, capture_output=False, text=True)
            elapsed = time.time() - t0

            if result.returncode != 0:
                print(f"Error: ha_solver exited with code {result.returncode}")
            else:
                print(f"\n=== HA_CPP Result ===")
                print(f"Instance : {instance_name}")
                print(f"Time     : {elapsed:.2f} s")
        finally:
            os.unlink(tmp_path)

    # elif solver == "ha":
    #     seed = solver_kwargs.pop("seed", None)
    #     if seed is not None:
    #         random.seed(seed)

    #     ha_kwargs = {
    #         "pop_size":     int(solver_kwargs.pop("pop_size",     PARAMS["pop_size"])),
    #         "max_gen":      int(solver_kwargs.pop("max_gen",      PARAMS["max_gen"])),
    #         "max_stagnant": int(solver_kwargs.pop("max_stagnant", PARAMS["max_stagnant"])),
    #         "pc":           float(solver_kwargs.pop("pc",         PARAMS["pc"])),
    #         "pm":           float(solver_kwargs.pop("pm",         PARAMS["pm"])),
    #         "pr":           float(solver_kwargs.pop("pr",         PARAMS["pr"])),
    #         "tabu_len":     int(solver_kwargs.pop("tabu_len",     PARAMS["tabu_len"])),
    #         "ts_iter_base": int(solver_kwargs.pop("ts_iter_base", PARAMS["ts_iter_base"])),
    #     }
    #     _warn_unknown_args(solver_kwargs, "HA")

    #     pt, _ = load_instance(instance_name)
    #     t0    = time.time()
    #     best  = hybrid_ga_ts(pt, **ha_kwargs)
    #     elapsed = time.time() - t0

    #     print(f"\n=== HA Result ===")
    #     print(f"Instance : {instance_name}")
    #     print(f"Makespan : {best.fitness}")
    #     print(f"Time     : {elapsed:.2f} s")

    else:
        print(f"Error: Unknown solver '{solver}'. Use 'gurobi', 'ha', or 'ha_cpp'.")
