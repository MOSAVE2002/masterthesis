# Generate FJSP instances
import os
import random
import pickle
import numpy as np


class FJSPInstance:
    def __init__(
        self,
        num_jobs: int,
        num_machines: int,
        min_processing_time: int = 1,
        max_processing_time: int = 10,
        min_operations_per_job: int = 1,
        max_operations_per_job: int = 3,
        min_machines_per_operation: int = 1,
        max_machines_per_operation: int = 2,
        jobs=None,
        machine_ids=None,
        instance_number: int = 1,
        instance_name: str | None = None,
        seed: int | None = None
    ) -> None:

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.num_jobs = num_jobs
        self.num_machines = num_machines
        self.instance_number = instance_number

        self.instance_name = (
            instance_name
            if instance_name is not None
            else f"instance_J{num_jobs}_M{num_machines}_{instance_number}"
        )

        if machine_ids is None:
            self.machine_ids = [f"M_{i}" for i in range(num_machines)]
        else:
            if len(machine_ids) != num_machines:
                raise ValueError("machine_ids muss genau num_machines Einträge haben.")
            self.machine_ids = machine_ids

        if max_machines_per_operation > num_machines:
            raise ValueError("max_machines_per_operation darf nicht größer als num_machines sein.")

        if min_machines_per_operation < 1:
            raise ValueError("min_machines_per_operation muss mindestens 1 sein.")

        if min_machines_per_operation > max_machines_per_operation:
            raise ValueError("min_machines_per_operation darf nicht größer als max_machines_per_operation sein.")

        if min_operations_per_job > max_operations_per_job:
            raise ValueError("min_operations_per_job darf nicht größer als max_operations_per_job sein.")

        if min_processing_time > max_processing_time:
            raise ValueError("min_processing_time darf nicht größer als max_processing_time sein.")

        if jobs is not None:
            self.jobs = jobs
        else:
            self.jobs = self._generate_jobs(
                min_processing_time=min_processing_time,
                max_processing_time=max_processing_time,
                min_operations_per_job=min_operations_per_job,
                max_operations_per_job=max_operations_per_job,
                min_machines_per_operation=min_machines_per_operation,
                max_machines_per_operation=max_machines_per_operation,
            )

    def _generate_jobs(
        self,
        min_processing_time: int,
        max_processing_time: int,
        min_operations_per_job: int,
        max_operations_per_job: int,
        min_machines_per_operation: int,
        max_machines_per_operation: int,
    ):
        jobs = []

        for job_index in range(self.num_jobs):
            job_id = f"J_{job_index}"

            num_operations = random.randint(min_operations_per_job,max_operations_per_job)

            operations = []

            for operation_index in range(num_operations):
                operation_id = f"{job_id}_O_{operation_index}"

                num_possible_machines = random.randint(min_machines_per_operation,max_machines_per_operation)

                eligible_machine_ids = random.sample(self.machine_ids,num_possible_machines) # randomly select machines for this operation

                machine_options = []

                for machine_id in eligible_machine_ids:
                    machine_options.append({
                        "machine_id": machine_id,
                        "processing_time": random.randint(min_processing_time,max_processing_time
                        )
                    })

                operation = {
                    "operation_id": operation_id,
                    "operation_index": operation_index,
                    "machine_options": machine_options
                }

                operations.append(operation)

            job = {
                "job_id": job_id,
                "operations": operations
            }

            jobs.append(job)

        return jobs
               
    
def generate_instances(instance = FJSPInstance):
    print("____Generate Instance_____")
    data_directory = 'data/'
    # data directory not exist

    if not os.path.exists(data_directory):
        os.makedirs(data_directory, exist_ok=True)

    for i in range(1, instance.instance_number + 1):  # instance_number  
        with open(f'{data_directory}/{instance.instance_name}.fjsp', 'wb') as fh:
            pickle.dump(instance, fh)

# def create_samples(instance: FJSPInstance, num_samples: int = 10):
#     samples = []
#     for _ in range(num_samples):
#         sample = {
#             "instance_name": instance.instance_name,
#             "num_jobs": instance.num_jobs,
#             "num_machines": instance.num_machines,
#             "jobs": instance.jobs
#         }
#         samples.append(sample)
#     return samples



# Example usage
if __name__ == "__main__":
    num_jobs = 2
    num_machines = 2
    instance = FJSPInstance(num_jobs=num_jobs, num_machines=num_machines, instance_number=1)
    print("Generated FJSP Instance:")
    print(f"Instance Name: {instance.instance_name}")
    print(f"Number of Jobs: {instance.num_jobs}")
    print(f"Number of Machines: {instance.num_machines}")
    print(f"Jobs: {instance.jobs}")

    #generate_instances(instance)