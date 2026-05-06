import random
import time 
import os


class FJSPInstanceGenerator:
    def __init__(
        self,
        num_jobs: int,
        num_machines: int,
        operations_per_job_min: int,
        operations_per_job_max: int,
        nums_ope = None,
        path = '../instances/',
        flag_same_operations = False,
        flag_save_file = False):

        """
        Initializes the FJSPInstanceGenerator with the specified parameters.
        abbreviations:
            ope: operation

        """

        if nums_ope is None:
            nums_ope = []
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
        self.numes_ope = nums_ope
        self.machine_per_ope_min = 1 # at least one machine must be able to process each operation
        self.machine_per_ope_max = num_machines
        
        # Processing time parameters
        self.processing_time_per_ope_min = 1
        self.processing_time_per_ope_max = 10
        self.proctime_deviation = 0.2


        if not self.flag_same_operations:
            self.nums_ope = [random.randint(self.ope_per_job_min, self.ope_per_job_max) for _ in range(self.num_jobs)]
        self.nums_operations = sum(self.nums_ope) # Amount of operations

        self.nums_option = [random.randint(1, self.num_machines) for _ in range(self.nums_operations)]   # was macht das?
        self.nums_options = sum(self.nums_option)

        self.ope_machine = []
        for val in self.nums_option:
            self.ope_machine = self.ope_machine + sorted(random.sample(range(self.num_machines), val)) # das ist nur eine flache Liste von möglichen Maschinen

        self.processing_time = []
        self.processing_times_mean = [random.randint(self.processing_time_per_ope_min, self.processing_time_per_ope_max) for _ in range(self.nums_operations)]
        
        for i in range(len(self.nums_option)):
            low_bound = max(self.processing_time_per_ope_min, round(self.processing_times_mean[i] * (1 - self.proctime_deviation))) #
            high_bound = min(self.processing_time_per_ope_max, round(self.processing_times_mean[i] * (1 + self.proctime_deviation))) #
            process_time_ope = [random.randint(low_bound, high_bound) for _ in range(self.nums_option[i])]
            self.processing_time = self.processing_time + process_time_ope 

        self.num_ope_bias = [sum(self.nums_option[0:i]) for i in range(self.num_jobs)] # kumulierte Summe als Liste
        self.num_machine_bias = [sum(self.nums_option[0:i]) for i in range(len(self.nums_option))]

        line0 = '{0}\t{1}\t{2}\n'.format(self.num_jobs, self.num_machines, self.nums_options / self.nums_operations)
        lines = []
        lines_doc = []
        lines.append(line0)
        lines_doc.append('{0}\t{1}\t{2}\n'.format(self.num_jobs, self.num_machines, self.nums_options / self.nums_operations))

        idx = 0
        for i in range(self.num_jobs):
            flag = 0
            flag_time = 0
            flag_new_ope = 1
            idx_ope = -1 
            idx_machine = 0
            line = []
            option_max = sum(self.nums_option[self.num_ope_bias[i]:self.num_ope_bias[i]+self.nums_ope[i]]) # da sind alle 
            idx_option = 0
            while True:
                if flag == 0:
                    line.append(self.nums_ope[i])
                    flag += 1
                elif flag == flag_new_ope:
                    idx_ope += 1
                    idx_ma = 0
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
                    idx_ma += 1
                if idx_option == option_max:
                    str_line = " ".join([str(val) for val in line])
                    lines.append(str_line + '\n')
                    lines_doc.append(str_line)
                    break
        lines.append('\n')


        if self.flag_save_file:
            if not os.path.exists(self.path):
                os.makedirs(self.path)
            document = open(self.path + '{0}j_{1}m_{2}.fjs'.format(self.num_jobs, self.num_machines, str.zfill(str(idx+4),3)),'a')
            for i in range(len(lines_doc)):
                print(lines_doc[i], file=document)
            document.close()

if __name__ == "__main__":
    num_jobs = 3
    num_machines = 2
    operations_per_job_min = 1
    operations_per_job_max = 10
    nums_ope = None
    path = '../FJSP_Simulation/instances/'
    flag_same_operations = False
    flag_save_file = True

    # Todo Anzahl an Instanzen hinzufügen mit idx


    generator = FJSPInstanceGenerator(
        num_jobs=num_jobs,
        num_machines=num_machines,
        operations_per_job_min=operations_per_job_min,
        operations_per_job_max=operations_per_job_max,
        nums_ope=nums_ope,
        path=path,
        flag_same_operations=flag_same_operations,
        flag_save_file=flag_save_file
    )

       

