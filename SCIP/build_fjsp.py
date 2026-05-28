from pyscipopt import Model, quicksum


def build_fjsp(fjsp: Model, instance):
    """
    Baut ein SCIP-MIP-Modell für das Flexible Job Shop Scheduling Problem
    mit Fertigstellungszeiten.

    X[i,j,k] = 1 bedeutet:
        Operation i wird vor Operation j auf Maschine k bearbeitet.
    """

    model = fjsp

    n_machines = instance.num_machines
    machines = list(range(n_machines))

    H = sum(
        max(instance.processing_times[i, k] for k in instance.eligible_machines[i])
        for i in instance.real_operations
    )

    C = {
        i: model.addVar(vtype="C", lb=0.0, name=f"C[{i}]")
        for i in instance.real_operations
    }

    C_max = model.addVar(vtype="C", lb=0.0, name="C_max")

    Y_index = [
        (i, k)
        for i in instance.real_operations
        for k in instance.eligible_machines[i]
    ]

    Y = {
        (i, k): model.addVar(vtype="B", name=f"Y[{i},{k}]")
        for i, k in Y_index
    }

    X_index = []
    for idx_i, i in enumerate(instance.real_operations):
        for j in instance.real_operations[idx_i + 1:]:
            common_machines = set(instance.eligible_machines[i]) & set(instance.eligible_machines[j])
            for k in common_machines:
                X_index.append((i, j, k))

    X = {
        (i, j, k): model.addVar(vtype="B", name=f"X[{i},{j},{k}]")
        for i, j, k in X_index
    }

    model.setObjective(C_max, "minimize")

    for i in instance.real_operations:
        model.addCons(
            quicksum(Y[i, k] for k in instance.eligible_machines[i]) == 1,
            name=f"assignment[{i}]",
        )

    for i in instance.real_operations:
        processing_time_i = quicksum(
            Y[i, k] * instance.processing_times[i, k]
            for k in instance.eligible_machines[i]
        )
        for j in instance.predecessors.get(i, []):
            model.addCons(
                C[i] >= C[j] + processing_time_i,
                name=f"precedence[{j}_before_{i}]",
            )

    for i, j, k in X_index:
        model.addCons(
            C[i] >= C[j] + instance.processing_times[i, k]
            - H * (2 + X[i, j, k] - Y[i, k] - Y[j, k]),
            name=f"nonoverlap_j_before_i[{j}_{i}_{k}]",
        )

        model.addCons(
            C[j] >= C[i] + instance.processing_times[j, k]
            - H * (3 - X[i, j, k] - Y[i, k] - Y[j, k]),
            name=f"nonoverlap_i_before_j[{i}_{j}_{k}]",
        )

    for i in instance.real_operations:
        processing_time_i = quicksum(
            Y[i, k] * instance.processing_times[i, k]
            for k in instance.eligible_machines[i]
        )
        model.addCons(
            C[i] >= processing_time_i,
            name=f"completion_lb[{i}]",
        )

    for u in instance.jobs:
        end_operation = instance.job_end_operations[u]
        model.addCons(
            C_max >= C[end_operation],
            name=f"makespan[{u}]",
        )

    variables = {
        "C": C,
        "C_max": C_max,
        "Y": Y,
        "X": X,
        "H": H,
        "operations": instance.real_operations,
        "real_operations": instance.real_operations,
        "eligible_machines": instance.eligible_machines,
        "processing_times": instance.processing_times,
        "machines": machines,
        "jobs": instance.jobs,
        "X_index": X_index,
        "Y_index": Y_index,
    }

    return model, variables


def write_solution_file(model: Model, variables, instance, filename="solution.txt"):
    """
    Schreibt die gefundene SCIP-Lösung in eine Textdatei.
    """

    if model.getNSols() == 0:
        with open(filename, "w", encoding="utf-8") as file:
            file.write("Keine zulässige Lösung gefunden.\n")
            file.write(f"SCIP-Status: {model.getStatus()}\n")
        return

    solution = model.getBestSol()

    C = variables["C"]
    C_max = variables["C_max"]
    Y = variables["Y"]
    real_operations = variables["real_operations"]
    eligible_machines = variables["eligible_machines"]
    processing_times = variables["processing_times"]
    rows = []

    for i in real_operations:
        assigned_machine = None

        for k in eligible_machines[i]:
            if model.getSolVal(solution, Y[i, k]) > 0.5:
                assigned_machine = k
                processing_time = processing_times[i, k]
                break

        if assigned_machine is None:
            continue

        completion_time = model.getSolVal(solution, C[i])
        start_time = completion_time - processing_time

        rows.append(
            {
                "operation": i,
                "machine": assigned_machine,
                "start": start_time,
                "completion": completion_time,
                "processing_time": processing_time,
            }
        )

    rows.sort(key=lambda row: (row["machine"], row["start"], row["operation"]))

    with open(filename, "w", encoding="utf-8") as file:
        file.write("FJSP-Lösung\n")
        file.write("=" * 70 + "\n\n")
        file.write(f"Status: {model.getStatus()}\n")
        file.write(f"Makespan: {model.getSolVal(solution, C_max):.4f}\n")
        file.write(f"Zielfunktionswert: {model.getObjVal():.4f}\n")
        file.write(f"Big-M H: {variables['H']}\n\n")
