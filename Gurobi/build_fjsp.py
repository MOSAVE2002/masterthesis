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

    n_machines = instance.num_machines
    machines = list(range(n_machines))
  

    # Randomgenerator, fixe Zuordnung

    # Big-M
    H = sum(
        max(instance.processing_times[i, k] for k in instance.eligible_machines[i])
        for i in instance.real_operations
    )

    # -------------------------
    # Variablen
    # -------------------------

    C = model.addVars(
        instance.real_operations,
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
        for i in instance.real_operations
        for k in instance.eligible_machines[i]
    ]

    Y = model.addVars(
        Y_index,
        vtype=GRB.BINARY,
        name="Y"
    )

    X_index = []

    for idx_i, i in enumerate(instance.real_operations):
        for j in instance.real_operations[idx_i + 1:]:
            common_machines = set(instance.eligible_machines[i]) & set(instance.eligible_machines[j])
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
    for i in instance.real_operations:
        model.addConstr(
            gp.quicksum(Y[i, k] for k in instance.eligible_machines[i]) == 1,
            name=f"assignment[{i}]"
        )

    # Technologische Präzedenzrelationen
    for i in instance.real_operations:
        for j in instance.predecessors.get(i, []):
            processing_time_i = gp.quicksum(
                Y[i, k] * instance.processing_times[i, k]
                for k in instance.eligible_machines[i]
            )

            model.addConstr(
                C[i] >= C[j] + processing_time_i,
                name=f"precedence[{j}_before_{i}]"
            )

    # Keine Überlappung auf derselben Maschine
    for i, j, k in X_index:
        # Fall X[i,j,k] = 0: j liegt vor i
        model.addConstr(
            C[i] >= C[j] + instance.processing_times[i, k]
            - H * (2 + X[i, j, k] - Y[i, k] - Y[j, k]),
            name=f"nonoverlap_j_before_i[{j}_{i}_{k}]"
        )

        # Fall X[i,j,k] = 1: i liegt vor j
        model.addConstr(
            C[j] >= C[i] + instance.processing_times[j, k]
            - H * (3 - X[i, j, k] - Y[i, k] - Y[j, k]),
            name=f"nonoverlap_i_before_j[{i}_{j}_{k}]"
        )

    # Completion lower bound
    for i in instance.real_operations:
        processing_time_i = gp.quicksum(
            Y[i, k] * instance.processing_times[i, k]
            for k in instance.eligible_machines[i]
        )
        model.addConstr(
            C[i] >= processing_time_i,
            name=f"completion_lb[{i}]"
        )

    # Makespan
    for u in instance.jobs:
        end_operation = instance.job_end_operations[u]
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
        "operations": instance.real_operations,
        "real_operations": instance.real_operations,
        "eligible_machines": instance.eligible_machines,
        "processing_times" : instance.processing_times,
        "machines": machines,
        "jobs": instance.jobs,
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

        # file.write("Operationen sortiert nach Maschine und Startzeit:\n")
        # file.write("-" * 70 + "\n")
        # file.write(
        #     f"{'Operation':>10} | {'Maschine':>8} | {'Start':>10} | "
        #     f"{'Ende':>10} | {'Dauer':>10}\n"
        # )
        # file.write("-" * 70 + "\n")

        # for row in rows:
        #     file.write(
        #         f"{row['operation']:>10} | "
        #         f"{row['machine']:>8} | "
        #         f"{row['start']:>10.4f} | "
        #         f"{row['completion']:>10.4f} | "
        #         f"{row['processing_time']:>10.4f}\n"
        #     )

        # file.write("\nMaschinenpläne:\n")
        # file.write("=" * 70 + "\n")

        # for k in machines:
        #     machine_rows = [row for row in rows if row["machine"] == k]

        #     file.write(f"\nMaschine {k}:\n")

        #     if not machine_rows:
        #         file.write("  Keine Operationen zugewiesen.\n")
        #         continue

        #     for row in machine_rows:
        #         file.write(
        #             f"  Operation {row['operation']}: "
        #             f"[{row['start']:.4f}, {row['completion']:.4f}] "
        #             f"Dauer={row['processing_time']:.4f}\n"
        #         )


if __name__ == '__main__':
    pass
