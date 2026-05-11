import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))
sys.path.append(str(ROOT_DIR / "Gurobi"))

from Gurobi.solve_instances_Fjsp import solveModel

def main(**kwargs):
    solver_kwargs = dict(kwargs)
    num_jobs = solver_kwargs.pop("num_jobs", 3)
    num_machines = solver_kwargs.pop("num_machines", 3)
    instance_nb = solver_kwargs.pop("instance_nb", 1)

    #solve_model_with_fixed_values = True

    instance_name = f"i{num_jobs}_k{num_machines}_{instance_nb}"
    solveModel(instance_name=instance_name, **solver_kwargs)


if __name__ == '__main__':
    main()



    
