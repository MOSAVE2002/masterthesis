#Instance Generator for flexible job shop scheduling problem
import random
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class Operation:
    job_id: int
    op_id: int
    processing_times: Dict[int, int]


@dataclass
class ScheduledOperation:
    job_id: int
    op_id: int
    machine_id: int
    start: int
    end: int


@dataclass
class FJSPInstance:
    jobs: List[List[Operation]]
    num_machines: int

class Settings:
    def __init__(
        self,
        num_jobs: int = 3,
        num_machines: int = 3,
        ops_per_job: int = 3,
        min_processing_time: int = 1,
        max_processing_time: int = 20,
        min_eligible_machines: int = 1,
        seed: Optional[int] = None,
    ):
        self.num_jobs = num_jobs
        self.num_machines = num_machines
        self.ops_per_job = ops_per_job
        self.min_processing_time = min_processing_time
        self.max_processing_time = max_processing_time
        self.min_eligible_machines = min_eligible_machines
        self.seed = seed


def generate_instance(
   settings: Settings
) -> FJSPInstance:
    if settings.seed is not None:
        random.seed(settings.seed)

    jobs = []

    for job_id in range(settings.num_jobs):
        job_operations = []

        for op_id in range(settings.ops_per_job):
            num_eligible = random.randint(settings.min_eligible_machines, settings.num_machines)
            eligible_machines = random.sample(range(settings.num_machines), num_eligible)

            processing_times = {
                machine_id: random.randint(settings.min_processing_time, settings.max_processing_time)
                for machine_id in eligible_machines
            }

            job_operations.append(
                Operation(
                    job_id=job_id,
                    op_id=op_id,
                    processing_times=processing_times,
                )
            )

        jobs.append(job_operations)

    return FJSPInstance(jobs=jobs, num_machines=settings.num_machines)


def print_instance(instance: FJSPInstance) -> None:
    print("FJSP-Instanz")
    print("------------")

    for job in instance.jobs:
        print(f"Job {job[0].job_id}:")
        for op in job:
            print(f"  Operation {op.op_id}: {op.processing_times}")
        print()