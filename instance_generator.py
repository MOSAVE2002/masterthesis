import pickle
import random
import time 
import os
from pathlib import Path


class FJSPData:
    def __init__(
        self,
        nb_instance: int,
        num_jobs: int,
        num_machines: int,
        operations_per_job_min: int,
        operations_per_job_max: int,
        num_operations = None,
        path = '../FJSP_Simulation/data/instances_text/',
        flag_same_operations = False,
        flag_save_file = True):

        """
        Initializes the FJSPInstanceGenerator with the specified parameters.
        abbreviations:
            ope: operation
        explanations of argumets

        """
        self.nb_instance = nb_instance

        if num_operations is None:
            num_operations = []
        self.path = path
        
        # flags
        self.flag_same_operations = flag_same_operations # if True, all jobs will have the same number of operations, determined by the average of operations_per_job_min and operations_per_job_max. If False, the number of operations per job will be randomly generated within the specified range.
        self.flag_save_file = flag_save_file 

        # instance parameters
        self.num_jobs = num_jobs
        self.num_machines = num_machines

        # Operations per job parameters
        self.ope_per_job_min = operations_per_job_min
        self.ope_per_job_max = operations_per_job_max
        
        # Machine options per operation parameters
        self.nums_operation = num_operations
        self.machine_per_ope_min = 1 # at least one machine must be able to process each operation
        self.machine_per_ope_max = num_machines
        
        # Processing time parameters
        self.processing_time_per_ope_min = 1
        self.processing_time_per_ope_max = 10
        self.proctime_deviation = 0.2

         # Instance Name
         #TODO mit zfill() sieht besser aus, wenn ich nacher mti tausenden datein arbeite, sieht das übersichtlicher aus
        self.instance_name = f"i{self.num_jobs}_k{self.num_machines}_{self.nb_instance}"


        if not self.flag_same_operations:
            self.nums_operation = [random.randint(self.ope_per_job_min, self.ope_per_job_max) for _ in range(self.num_jobs)]
        self.num_operations = sum(self.nums_operation) # Amount of operations

        self.nums_option = [random.randint(1, self.num_machines) for _ in range(self.num_operations)]   # was macht das?
        self.nums_options = sum(self.nums_option)

        self.ope_machine = []
        for val in self.nums_option:
            self.ope_machine = self.ope_machine + sorted(random.sample(range(self.num_machines), val)) # flache Liste von möglichen Maschinen

        self.processing_time = []
        self.processing_times_mean = [random.randint(self.processing_time_per_ope_min, self.processing_time_per_ope_max) for _ in range(self.num_operations)]
        
        for i in range(len(self.nums_option)):
            low_bound = max(self.processing_time_per_ope_min, round(self.processing_times_mean[i] * (1 - self.proctime_deviation))) #
            high_bound = min(self.processing_time_per_ope_max, round(self.processing_times_mean[i] * (1 + self.proctime_deviation))) #
            process_time_ope = [random.randint(low_bound, high_bound) for _ in range(self.nums_option[i])]
            self.processing_time = self.processing_time + process_time_ope 

        self.num_ope_bias = [sum(self.nums_operation[0:i]) for i in range(self.num_jobs)] # kumulierte Summe als Liste
        self.num_machine_bias = [sum(self.nums_option[0:i]) for i in range(self.num_operations)] # kumulierte Summe als Liste

       

        line0 = '{0}\t{1}\t{2}\n'.format(self.num_jobs, self.num_machines, self.nums_options / self.num_operations)
        lines = []
        lines_doc = []
        lines.append(line0)
        #TODO: Weitere Informationen sammeln, wie Upper und Lower Bound? -> Literatur für Sampling?
        lines_doc.append('{0}\t{1}\t{2}\n'.format(self.num_jobs, self.num_machines, self.nums_options / self.num_operations))
    
        idx = self.nb_instance
        for i in range(self.num_jobs):
            flag = 0
            flag_time = 0
            flag_new_ope = 1
            idx_ope = -1 
            idx_machine = 0
            line = []
            option_max = sum(self.nums_option[self.num_ope_bias[i]:self.num_ope_bias[i]+self.nums_operation[i]]) # da sind alle 
            idx_option = 0
            while True:
                if flag == 0:
                    line.append(self.nums_operation[i])
                    flag += 1
                elif flag == flag_new_ope:
                    idx_ope += 1
                    idx_machine = 0
                    flag_new_ope += self.nums_option[self.num_ope_bias[i]+idx_ope] * 2 + 1
                    line.append(self.nums_option[self.num_ope_bias[i]+idx_ope])
                    flag += 1
                elif flag_time == 0:
                    line.append(self.ope_machine[self.num_machine_bias[self.num_ope_bias[i]+idx_ope] + idx_machine])
                    flag += 1
                    flag_time = 1
                else:
                    line.append(self.processing_time[self.num_machine_bias[self.num_ope_bias[i]+idx_ope] + idx_machine])
                    flag += 1
                    flag_time = 0
                    idx_option += 1
                    idx_machine+= 1
                if idx_option == option_max:
                    str_line = " ".join([str(val) for val in line])
                    lines.append(str_line + '\n')
                    lines_doc.append(str_line)
                    break
        lines.append('\n')

        self.lines = lines

        # Text file zum einfacheren Lesen lassen
        if self.flag_save_file:
            if not os.path.exists(self.path):
                os.makedirs(self.path)
            document = open(self.path + '{0}j_{1}m_{2}.fjs'.format(self.num_jobs, self.num_machines, str.zfill(str(idx),3)),'w')
            for i in range(len(lines_doc)):
                print(lines_doc[i], file=document)
            document.close()
        def __repr__(self):
            return f"FJSP({self.instance_name})"

def generate_instances(nb_instances, num_jobs, num_machines, operations_per_job_min, operations_per_job_max, num_operations):
    """
    Generate multiple instances 

    Parameters:

    Returns:
    - a pickle file with the data for each generated instance
    """
    project_root = Path(__file__).resolve().parent
    data_directory = project_root / "data" / "fsjp_instances"
    if not data_directory.exists():
        data_directory.mkdir(parents=True, exist_ok=True)

    # generate the number of instances as defined in nb_instances
    for instance_nb in range(1, nb_instances + 1):
        # generate an instance
        instance = FJSPData(nb_instance=instance_nb,num_jobs=num_jobs, num_machines=num_machines, operations_per_job_min=operations_per_job_min, operations_per_job_max=operations_per_job_max, num_operations=None)

        # save the generated instance to the created directory using pickle
        with open(data_directory / f"{instance.instance_name}.fjsp", "wb") as output_file:
            pickle.dump(instance, output_file)


if __name__ == "__main__":
    
    generate_instances(nb_instances =5, num_jobs=3, num_machines=3, operations_per_job_min=1, operations_per_job_max=2, num_operations=None)



       
