import gurobipy as gp
from gurobipy import GRB

def build_fjsp(fjsp, instance):
    """
    Baut ein MIP-Modell für das Flexible Job Shop Scheduling Problem
    mit Fertigstellungszeiten.

    X[i,j,k] = 1 bedeutet:
        Operation i wird vor Operation j auf Maschine k bearbeitet.
    """
    
    model = fjsp

    n_jobs = instance.num_jobs
    n_machines = instance.num_machines
    machines = list(range(n_machines))

    #TODO im Instanzgenerator schon vorbereiten? Dann spare ich mir das berechnen beim Modell erstellen, bei Algoritm das Gleiche

    jobs = {} # set of all jobs
    eligible_machines = {} # set of 
    processing_times = {} 
    predecessors = {}
    job_end_operations = {}
    real_operations = []

    operation_id = 1

    for job_index in range(n_jobs):
        num_operations = instance.nums_operation[job_index]
        job_operations = []

        for local_operation_index in range(num_operations):
            current_operation = operation_id
            operation_id += 1

            global_op_idx = instance.num_ope_bias[job_index] + local_operation_index
            num_options = instance.nums_option[global_op_idx]
            machine_offset = instance.num_machine_bias[global_op_idx]

            eligible = []
            for option_idx in range(num_options):
                machine = instance.ope_machine[machine_offset + option_idx]
                processing_time = instance.processing_time[machine_offset + option_idx]
                eligible.append(machine)
                processing_times[current_operation, machine] = processing_time

            eligible_machines[current_operation] = eligible
            job_operations.append(current_operation)
            real_operations.append(current_operation)

            if local_operation_index == 0:
                predecessors[current_operation] = []
            else:
                predecessors[current_operation] = [job_operations[local_operation_index - 1]]

        jobs[job_index + 1] = job_operations
        job_end_operations[job_index + 1] = job_operations[-1]

    # Big-M
    H = sum(
        max(processing_times[i, k] for k in eligible_machines[i])
        for i in real_operations
    )

    # -------------------------
    # Variablen
    # -------------------------

    C = model.addVars(
        real_operations,
        lb=0.0,
        vtype=GRB.CONTINUOUS,
        name="C"
    )

    C_max = model.addVar(
        lb=0.0,
        vtype=GRB.CONTINUOUS,
        name="C_max"
    )

    Y_index = [
        (i, k)
        for i in real_operations
        for k in eligible_machines[i]
    ]

    Y = model.addVars(
        Y_index,
        vtype=GRB.BINARY,
        name="Y"
    )

    X_index = []

    for idx_i, i in enumerate(real_operations):
        for j in real_operations[idx_i + 1:]:
            common_machines = set(eligible_machines[i]) & set(eligible_machines[j])
            for k in common_machines:
                X_index.append((i, j, k))

    X = model.addVars(
        X_index,
        vtype=GRB.BINARY,
        name="X"
    )

    # -------------------------
    # Zielfunktion
    # -------------------------

    model.setObjective(C_max, GRB.MINIMIZE)

    # -------------------------
    # Nebenbedingungen
    # -------------------------

    # Jede Operation wird genau einer zulässigen Maschine zugewiesen
    for i in real_operations:
        model.addConstr(
            gp.quicksum(Y[i, k] for k in eligible_machines[i]) == 1,
            name=f"assignment[{i}]"
        )

    # Technologische Präzedenzrelationen
    for i in real_operations:
        for j in predecessors.get(i, []):
            processing_time_i = gp.quicksum(
                Y[i, k] * processing_times[i, k]
                for k in eligible_machines[i]
            )

            model.addConstr(
                C[i] >= C[j] + processing_time_i,
                name=f"precedence[{j}_before_{i}]"
            )

    # Keine Überlappung auf derselben Maschine
    for i, j, k in X_index:
        # Fall X[i,j,k] = 0: j liegt vor i
        model.addConstr(
            C[i] >= C[j] + processing_times[i, k]
            - H * (2 + X[i, j, k] - Y[i, k] - Y[j, k]),
            name=f"nonoverlap_j_before_i[{j}_{i}_{k}]"
        )

        # Fall X[i,j,k] = 1: i liegt vor j
        model.addConstr(
            C[j] >= C[i] + processing_times[j, k]
            - H * (3 - X[i, j, k] - Y[i, k] - Y[j, k]),
            name=f"nonoverlap_i_before_j[{i}_{j}_{k}]"
        )

    # Completion lower bound
    for i in real_operations:
        processing_time_i = gp.quicksum(
            Y[i, k] * processing_times[i, k]
            for k in eligible_machines[i]
        )
        model.addConstr(
            C[i] >= processing_time_i,
            name=f"completion_lb[{i}]"
        )

    # Makespan
    for u in jobs:
        end_operation = job_end_operations[u]
        model.addConstr(
            C_max >= C[end_operation],
            name=f"makespan[{u}]"
        )

    model.update()

    variables = {
        "C": C,
        "C_max": C_max,
        "Y": Y,
        "X": X,
        "H": H,
        "operations": real_operations,
        "real_operations": real_operations,
        "eligible_machines": eligible_machines,
        "processing_times" : processing_times,
        "machines": machines,
        "jobs": jobs,
        "X_index": X_index,
        "Y_index": Y_index,
    }

    return model, variables


def write_solution_file(model, variables, instance, filename="solution.txt"):
    """
    Schreibt die gefundene Lösung in eine Textdatei.
    """

    feasible_statuses = {
        GRB.OPTIMAL,
        GRB.TIME_LIMIT,
        GRB.SUBOPTIMAL,
        GRB.USER_OBJ_LIMIT,
    }

    if model.Status not in feasible_statuses or model.SolCount == 0:
        with open(filename, "w", encoding="utf-8") as file:
            file.write("Keine zulässige Lösung gefunden.\n")
            file.write(f"Gurobi-Status: {model.Status}\n")
        return


    C = variables["C"]
    C_max = variables["C_max"]
    Y = variables["Y"]
    real_operations = variables["real_operations"]
    eligible_machines = variables["eligible_machines"]
    processing_times = variables["processing_times"]
    machines = variables["machines"]
    rows = []

    for i in real_operations:

        for k in eligible_machines[i]:
            if Y[i, k].X > 0.5:
                assigned_machine = k
                processing_time = processing_times[i, k]
                break

        if assigned_machine is None:
            continue

        completion_time = C[i].X
        start_time = completion_time - processing_time

        rows.append({
            "operation": i,
            "machine": assigned_machine,
            "start": start_time,
            "completion": completion_time,
            "processing_time": processing_time,
        })

    rows.sort(key=lambda row: (row["machine"], row["start"], row["operation"]))

    with open(filename, "w", encoding="utf-8") as file:
        file.write("FJSP-Lösung\n")
        file.write("=" * 70 + "\n\n")

        file.write(f"Status: {model.Status}\n")
        file.write(f"Makespan: {C_max.X:.4f}\n")
        file.write(f"Zielfunktionswert: {model.ObjVal:.4f}\n")
        file.write(f"Big-M H: {variables['H']}\n\n")

        file.write("Operationen sortiert nach Maschine und Startzeit:\n")
        file.write("-" * 70 + "\n")
        file.write(
            f"{'Operation':>10} | {'Maschine':>8} | {'Start':>10} | "
            f"{'Ende':>10} | {'Dauer':>10}\n"
        )
        file.write("-" * 70 + "\n")

        for row in rows:
            file.write(
                f"{row['operation']:>10} | "
                f"{row['machine']:>8} | "
                f"{row['start']:>10.4f} | "
                f"{row['completion']:>10.4f} | "
                f"{row['processing_time']:>10.4f}\n"
            )

        file.write("\nMaschinenpläne:\n")
        file.write("=" * 70 + "\n")

        for k in machines:
            machine_rows = [row for row in rows if row["machine"] == k]

            file.write(f"\nMaschine {k}:\n")

            if not machine_rows:
                file.write("  Keine Operationen zugewiesen.\n")
                continue

            for row in machine_rows:
                file.write(
                    f"  Operation {row['operation']}: "
                    f"[{row['start']:.4f}, {row['completion']:.4f}] "
                    f"Dauer={row['processing_time']:.4f}\n"
                )


if __name__ == '__main__':
    pass
