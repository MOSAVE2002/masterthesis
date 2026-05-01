from typing import List

from instance_model import ScheduledOperation


def print_schedule(schedule: List[ScheduledOperation]) -> None:
    print("Schedule")
    print("--------")

    for op in sorted(schedule, key=lambda x: x.start):
        print(
            f"Job {op.job_id}, Op {op.op_id}, "
            f"Maschine {op.machine_id}: "
            f"{op.start} -> {op.end}"
        )


def plot_gantt(schedule: List[ScheduledOperation], num_machines: int) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4))

    for op in schedule:
        ax.barh(
            y=op.machine_id,
            width=op.end - op.start,
            left=op.start,
            height=0.4,
        )

        ax.text(
            x=op.start + (op.end - op.start) / 2,
            y=op.machine_id,
            s=f"J{op.job_id}-O{op.op_id}",
            va="center",
            ha="center",
            fontsize=8,
        )

    ax.set_xlabel("Zeit")
    ax.set_ylabel("Maschine")
    ax.set_yticks(range(num_machines))
    ax.set_yticklabels([f"M{m}" for m in range(num_machines)])
    ax.set_title("Gantt-Chart")
    ax.grid(True, axis="x", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.show()

def plot_graph(graph: dict, instance) -> None:
    """
    Visualisiert den Graphen aus graph_converter.env_to_graph(env).

    Operation-Knoten:
        O{job_id}.{op_id}

    Maschinen-Knoten:
        M{machine_id}

    Kantentypen:
        - precedence: Operation -> nächste Operation im Job
        - op_machine: Operation -> Maschine
        - machine_op: Maschine -> Operation
    """

    import matplotlib.pyplot as plt
    import networkx as nx

    edge_index = graph["edge_index"]
    edge_attr = graph["edge_attr"]

    num_jobs = len(instance.jobs)
    ops_per_job = len(instance.jobs[0])
    num_operation_nodes = graph["num_operation_nodes"]
    num_machines = instance.num_machines

    G = nx.DiGraph()

    labels = {}
    node_colors = []

    # -----------------------------
    # Operation-Knoten
    # -----------------------------
    for job_id in range(num_jobs):
        for op_id in range(ops_per_job):
            node_id = job_id * ops_per_job + op_id
            G.add_node(node_id)

            labels[node_id] = f"O{job_id}.{op_id}"
            node_colors.append("lightblue")

    # -----------------------------
    # Maschinen-Knoten
    # -----------------------------
    for machine_id in range(num_machines):
        node_id = num_operation_nodes + machine_id
        G.add_node(node_id)

        labels[node_id] = f"M{machine_id}"
        node_colors.append("lightgreen")

    # -----------------------------
    # Kanten
    # -----------------------------
    edge_labels = {}
    precedence_edges = []
    op_machine_edges = []
    machine_op_edges = []

    for edge_idx in range(edge_index.shape[1]):
        src = int(edge_index[0, edge_idx])
        dst = int(edge_index[1, edge_idx])

        attr = edge_attr[edge_idx].tolist()

        precedence_edge = attr[0] == 1.0
        operation_to_machine_edge = attr[1] == 1.0
        machine_to_operation_edge = attr[2] == 1.0
        processing_time = attr[3]

        G.add_edge(src, dst)

        if precedence_edge:
            precedence_edges.append((src, dst))
        elif operation_to_machine_edge:
            op_machine_edges.append((src, dst))
            edge_labels[(src, dst)] = str(int(processing_time))
        elif machine_to_operation_edge:
            machine_op_edges.append((src, dst))
            edge_labels[(src, dst)] = str(int(processing_time))

    # -----------------------------
    # Layout
    # -----------------------------
    pos = {}

    # Operationen links/rechts nach op_id, Jobs nach y-Achse
    for job_id in range(num_jobs):
        for op_id in range(ops_per_job):
            node_id = job_id * ops_per_job + op_id
            pos[node_id] = (op_id, -job_id)

    # Maschinen rechts daneben
    for machine_id in range(num_machines):
        node_id = num_operation_nodes + machine_id
        pos[node_id] = (ops_per_job + 1.5, -machine_id)

    # -----------------------------
    # Zeichnen
    # -----------------------------
    plt.figure(figsize=(12, 6))

    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=node_colors,
        node_size=900,
    )

    nx.draw_networkx_labels(
        G,
        pos,
        labels=labels,
        font_size=9,
    )

    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=precedence_edges,
        arrows=True,
        width=2,
        style="solid",
    )

    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=op_machine_edges,
        arrows=True,
        width=1,
        style="dashed",
        alpha=0.6,
    )

    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=machine_op_edges,
        arrows=True,
        width=1,
        style="dotted",
        alpha=0.4,
    )

    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels=edge_labels,
        font_size=7,
    )

    plt.title("FJSP Graph")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

