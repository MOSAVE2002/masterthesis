from typing import Dict, List, Tuple

import torch

from environment import FJSPEnvironment


def operation_node_id(job_id: int, op_id: int, ops_per_job: int) -> int:
    """
    Eindeutige Node-ID für eine Operation.

    Annahme:
    Jeder Job hat gleich viele Operationen.
    """
    return job_id * ops_per_job + op_id


def machine_node_id(machine_id: int, num_operation_nodes: int) -> int:
    """
    Maschinenknoten kommen nach allen Operation-Knoten.
    """
    return num_operation_nodes + machine_id


def env_to_graph(env: FJSPEnvironment) -> Dict:
    """
    Wandelt den aktuellen Environment-Zustand in einen Graphen um.

    Rückgabe:
    {
        "x": node_features,
        "edge_index": edge_index,
        "edge_attr": edge_features,
        "action_mask": action_mask,
        "actions": actions
    }

    Diese Struktur ist absichtlich nah an PyTorch Geometric:
    Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    """

    instance = env.instance

    num_jobs = len(instance.jobs)
    num_machines = instance.num_machines
    ops_per_job = len(instance.jobs[0])

    num_operation_nodes = num_jobs * ops_per_job
    num_machine_nodes = num_machines
    num_nodes = num_operation_nodes + num_machine_nodes

    node_features: List[List[float]] = []

    # --------------------------------------------------
    # 1. Operation-Knoten-Features
    # --------------------------------------------------
    for job_id, job in enumerate(instance.jobs):
        for op in job:
            op_id = op.op_id
            node_id = operation_node_id(job_id, op_id, ops_per_job)

            is_scheduled = 1.0 if op_id < env.next_op_index[job_id] else 0.0
            is_available = 1.0 if op_id == env.next_op_index[job_id] else 0.0

            # Früheste Startzeit des Jobs
            job_ready_time = float(env.job_ready_time[job_id])

            # Durchschnittliche Bearbeitungszeit über mögliche Maschinen
            avg_processing_time = (
                sum(op.processing_times.values()) / len(op.processing_times)
            )

            # Minimale Bearbeitungszeit über mögliche Maschinen
            min_processing_time = min(op.processing_times.values())

            # Anzahl möglicher Maschinen
            num_eligible_machines = len(op.processing_times)

            # Verbleibende Operationen im Job
            remaining_ops = len(job) - env.next_op_index[job_id]

            node_features.append(
                [
                    1.0,  # is_operation_node
                    0.0,  # is_machine_node
                    float(job_id),
                    float(op_id),
                    is_scheduled,
                    is_available,
                    job_ready_time,
                    float(avg_processing_time),
                    float(min_processing_time),
                    float(num_eligible_machines),
                    float(remaining_ops),
                ]
            )

    # --------------------------------------------------
    # 2. Maschinen-Knoten-Features
    # --------------------------------------------------
    for machine_id in range(num_machines):
        machine_ready_time = float(env.machine_ready_time[machine_id])

        # Wie viele aktuell verfügbare Operationen können auf dieser Maschine laufen?
        waiting_ops = 0
        for action in env.get_available_actions():
            _, _, action_machine_id = action
            if action_machine_id == machine_id:
                waiting_ops += 1

        node_features.append(
            [
                0.0,  # is_operation_node
                1.0,  # is_machine_node
                -1.0,  # job_id
                -1.0,  # op_id
                0.0,  # is_scheduled
                0.0,  # is_available
                machine_ready_time,
                0.0,  # avg_processing_time
                0.0,  # min_processing_time
                0.0,  # num_eligible_machines
                float(waiting_ops),
            ]
        )

    # --------------------------------------------------
    # 3. Kanten bauen
    # --------------------------------------------------
    edges: List[Tuple[int, int]] = []
    edge_features: List[List[float]] = []

    # Kanten-Typen:
    # [precedence_edge, operation_to_machine_edge, machine_to_operation_edge]

    # 3.1 Precedence-Kanten: Operation i -> Operation i+1 im gleichen Job
    for job_id, job in enumerate(instance.jobs):
        for op_idx in range(len(job) - 1):
            src = operation_node_id(job_id, op_idx, ops_per_job)
            dst = operation_node_id(job_id, op_idx + 1, ops_per_job)

            edges.append((src, dst))
            edge_features.append(
                [
                    1.0,  # precedence_edge
                    0.0,  # operation_to_machine_edge
                    0.0,  # machine_to_operation_edge
                    0.0,  # processing_time
                ]
            )

    # 3.2 Operation-Maschine-Kanten
    for job_id, job in enumerate(instance.jobs):
        for op in job:
            op_node = operation_node_id(job_id, op.op_id, ops_per_job)

            for machine_id, processing_time in op.processing_times.items():
                machine_node = machine_node_id(machine_id, num_operation_nodes)

                # Operation -> Maschine
                edges.append((op_node, machine_node))
                edge_features.append(
                    [
                        0.0,
                        1.0,
                        0.0,
                        float(processing_time),
                    ]
                )

                # Maschine -> Operation
                edges.append((machine_node, op_node))
                edge_features.append(
                    [
                        0.0,
                        0.0,
                        1.0,
                        float(processing_time),
                    ]
                )

    # --------------------------------------------------
    # 4. Action-Mask
    # --------------------------------------------------
    actions = env.get_available_actions()

    action_mask = []

    for action in actions:
        job_id, op_id, machine_id = action

        op_node = operation_node_id(job_id, op_id, ops_per_job)
        machine_node = machine_node_id(machine_id, num_operation_nodes)

        action_mask.append(
            [
                float(job_id),
                float(op_id),
                float(machine_id),
                float(op_node),
                float(machine_node),
            ]
        )

    # --------------------------------------------------
    # 5. Tensoren erzeugen
    # --------------------------------------------------
    x = torch.tensor(node_features, dtype=torch.float)

    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 4), dtype=torch.float)

    action_mask = torch.tensor(action_mask, dtype=torch.float)

    return {
        "x": x,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "action_mask": action_mask,
        "actions": actions,
        "num_operation_nodes": num_operation_nodes,
        "num_machine_nodes": num_machine_nodes,
        "num_nodes": num_nodes,
    }