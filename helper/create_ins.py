import json
from instance_generator import FJSPInstanceGenerator


def main():
    batch_size = 10 # batch size, wie binde ich das ein, macht das überhaupt Sinn?
    num_jobs = 5
    num_machines = 3
    operation_per_job_min = int(num_machines * 0.8)
    operation_per_job_max = int(num_machines * 1.2)
    with open("../config.json", "r") as f: # vielleicht später noch relevanter, wenn NN trainiert werden soll
        config = json.load(f)
    instances = FJSPInstanceGenerator(num_jobs=num_jobs, num_machines=num_machines, operations_per_job_min=operation_per_job_min, operations_per_job_max=operation_per_job_max, flag_save_file=True, path=config["instance_path"], flag_same_operations=True)

   

if __name__ == "__main__":
    main()
    print("Instances generated successfully.")