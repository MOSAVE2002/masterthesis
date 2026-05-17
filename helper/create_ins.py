import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

import json
import sys
from pathlib import Path
from generator.instance_generator import generate_instances


def main():
    batch_size = 10 # batch size, wie binde ich das ein, macht das überhaupt Sinn?
    num_jobs = 30
    num_machines = 10
    nb_instances = 1
    operation_per_job_min = int(num_machines * 0.8)
    operation_per_job_max = int(num_machines * 1.2)
    #TODO Config Datei erstellen, am Ende nur darüber die Instanzen steuern
    # with open("config.json", "r") as f: # vielleicht später noch relevanter, wenn NN trainiert werden soll
    #     config = json.load(f)
    generate_instances(nb_instances=nb_instances,num_jobs=num_jobs, num_machines=num_machines, operations_per_job_min=operation_per_job_min, operations_per_job_max=operation_per_job_max, num_operations=None)

if __name__ == "__main__":
    main()
    print("Instances generated successfully.")
