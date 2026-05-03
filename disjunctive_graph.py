# Generate an disjunctive graph representation of the Flexible Job Shop Scheduling Problem (FJSP) instance
import os
import pickle
from instance_model import FJSPInstance
import networkx as nx

class DisjunctiveGraph(FJSPInstance):
    def __init__(self, instance: FJSPInstance) -> None:
        super().__init__(
            num_jobs=instance.num_jobs,
            num_machines=instance.num_machines,
            min_processing_time=1,
            max_processing_time=10,
            min_operations_per_job=1,
            max_operations_per_job=3,
            min_machines_per_operation=1,
            max_machines_per_operation=2,
            jobs=instance.jobs,
            machine_ids=instance.machine_ids,
            instance_number=instance.instance_number,
            instance_name=instance.instance_name
        )
        self.graph = self.create_disjunctive_graph()
    
    def create_disjunctive_graph(self):
        graph = nx.DiGraph()
        
        # Add nodes for each operation
        for job in self.jobs:
            for operation in job["operations"]:
                node_id = f"{job['job_id']}_{operation['operation_id']}"
                graph.add_node(node_id, processing_time=operation["processing_time"])
        
        # Add conjunctive edges (precedence constraints)
        for job in self.jobs:
            for i in range(len(job["operations"]) - 1):
                from_node = f"{job['job_id']}_{job['operations'][i]['operation_id']}"
                to_node = f"{job['job_id']}_{job['operations'][i + 1]['operation_id']}"
                graph.add_edge(from_node, to_node, type="conjunctive")

