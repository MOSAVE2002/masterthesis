import random

from environment import FJSPEnvironment


def random_scheduler(env: FJSPEnvironment):
    env.reset()

    while not env.is_done():
        actions = env.get_available_actions()
        action = random.choice(actions)
        env.step(action)

    return env.schedule, env.get_makespan()


def shortest_processing_time_scheduler(env: FJSPEnvironment):
    env.reset()

    while not env.is_done():
        actions = env.get_available_actions()

        best_action = None
        best_processing_time = float("inf")

        for action in actions:
            job_id, op_id, machine_id = action
            op_idx = env.next_op_index[job_id]
            operation = env.instance.jobs[job_id][op_idx]
            processing_time = operation.processing_times[machine_id]

            if processing_time < best_processing_time:
                best_processing_time = processing_time
                best_action = action

        env.step(best_action)

    return env.schedule, env.get_makespan()


def earliest_start_time_scheduler(env: FJSPEnvironment):
    env.reset()

    while not env.is_done():
        actions = env.get_available_actions()

        best_action = None
        best_start_time = float("inf")
        best_end_time = float("inf")

        for action in actions:
            job_id, op_id, machine_id = action
            op_idx = env.next_op_index[job_id]
            operation = env.instance.jobs[job_id][op_idx]

            processing_time = operation.processing_times[machine_id]

            start_time = max(
                env.job_ready_time[job_id],
                env.machine_ready_time[machine_id],
            )
            end_time = start_time + processing_time

            if start_time < best_start_time:
                best_start_time = start_time
                best_end_time = end_time
                best_action = action
            elif start_time == best_start_time and end_time < best_end_time:
                best_end_time = end_time
                best_action = action

        env.step(best_action)

    return env.schedule, env.get_makespan()
