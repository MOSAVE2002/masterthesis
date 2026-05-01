from environment import FJSPEnvironment
from graph_converter import env_to_graph
from heuristic import (
    random_scheduler,
    shortest_processing_time_scheduler,
    earliest_start_time_scheduler,
)
from instance_model import generate_instance, print_instance
from visualize import print_schedule, plot_gantt,plot_graph


def main():
    
    instance = generate_instance(
        num_jobs=5,
        num_machines=3,
        ops_per_job=3,
        min_processing_time=1,
        max_processing_time=20,
        seed=42,
    )

    print_instance(instance)

    env = FJSPEnvironment(instance)

    graph = env_to_graph(env)
    plot_graph(graph, instance)

    print("Graph")
    print("-----")
    print("Node features x:", graph["x"].shape)
    print("Edge index:", graph["edge_index"].shape)
    print("Edge attributes:", graph["edge_attr"].shape)
    print("Action mask:", graph["action_mask"].shape)
    print("Actions:", graph["actions"])
    print()

    print("Random Scheduler")
    schedule, makespan = random_scheduler(env)
    print_schedule(schedule)
    print("Makespan:", makespan)
    print()

    print("Shortest Processing Time Scheduler")
    schedule, makespan = shortest_processing_time_scheduler(env)
    print_schedule(schedule)
    print("Makespan:", makespan)
    print()

    print("Earliest Start Time Scheduler")
    schedule, makespan = earliest_start_time_scheduler(env)
    print_schedule(schedule)
    print("Makespan:", makespan)

    plot_gantt(schedule, instance.num_machines)


if __name__ == "__main__":
    main()