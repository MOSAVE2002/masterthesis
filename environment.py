# Stateclass
from typing import List, Tuple
from instance_model import FJSPInstance, ScheduledOperation

class FJSPEnvironment:
    def __init__(self, instance: FJSPInstance):
        self.instance = instance
        self.reset()

    def reset(self):
        self.num_jobs = len(self.instance.jobs)
        self.num_machines = self.instance.num_machines

        self.next_op_index = [0 for _ in range(self.num_jobs)]
        self.job_ready_time = [0 for _ in range(self.num_jobs)]
        self.machine_ready_time = [0 for _ in range(self.num_machines)]

        self.schedule: List[ScheduledOperation] = []

        return self.get_state()

    def get_state(self):
        return {
            "next_op_index": self.next_op_index.copy(),
            "job_ready_time": self.job_ready_time.copy(),
            "machine_ready_time": self.machine_ready_time.copy(),
            "schedule": self.schedule.copy(),
        }

    def get_available_actions(self) -> List[Tuple[int, int, int]]:
        actions = []

        for job_id in range(self.num_jobs):
            op_idx = self.next_op_index[job_id]

            if op_idx >= len(self.instance.jobs[job_id]):
                continue

            operation = self.instance.jobs[job_id][op_idx]

            for machine_id in operation.processing_times.keys():
                actions.append((job_id, operation.op_id, machine_id))

        return actions

    def step(self, action: Tuple[int, int, int]):
        job_id, op_id, machine_id = action

        expected_op_idx = self.next_op_index[job_id]

        if expected_op_idx >= len(self.instance.jobs[job_id]):
            raise ValueError(f"Job {job_id} ist bereits fertig.")

        operation = self.instance.jobs[job_id][expected_op_idx]

        if operation.op_id != op_id:
            raise ValueError(
                f"Ungültige Operation. Erwartet: {operation.op_id}, bekommen: {op_id}"
            )

        if machine_id not in operation.processing_times:
            raise ValueError(
                f"Operation {op_id} von Job {job_id} kann nicht auf Maschine {machine_id} laufen."
            )

        processing_time = operation.processing_times[machine_id]

        start_time = max(
            self.job_ready_time[job_id],
            self.machine_ready_time[machine_id],
        )

        end_time = start_time + processing_time

        scheduled_op = ScheduledOperation(
            job_id=job_id,
            op_id=op_id,
            machine_id=machine_id,
            start=start_time,
            end=end_time,
        )

        self.schedule.append(scheduled_op)

        self.job_ready_time[job_id] = end_time
        self.machine_ready_time[machine_id] = end_time
        self.next_op_index[job_id] += 1

        done = self.is_done()
        reward = -self.get_makespan() if done else 0

        return self.get_state(), reward, done

    def is_done(self) -> bool:
        for job_id in range(self.num_jobs):
            if self.next_op_index[job_id] < len(self.instance.jobs[job_id]):
                return False

        return True

    def get_makespan(self) -> int:
        if not self.schedule:
            return 0

        return max(op.end for op in self.schedule)




    