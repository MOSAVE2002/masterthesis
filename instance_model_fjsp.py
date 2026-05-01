#Instance Generator for flexible job shop scheduling problem
import random
import pickle
import math
import os

class FjspData:

    def __init__(self, num_jobs, num_machines, processing_time_range, machine_options):
        self.num_jobs = num_jobs
        self.num_machines = num_machines
        self.processing_time_range = processing_time_range
        self.machine_options = machine_options
        self.jobs = self.generate_jobs()

def generate_instances():
    pass