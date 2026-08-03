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

    # H dient als Makespan-Horizont und Big-M-Konstante.
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

    # Maschinenlast-Untergrenze: der Makespan ist mindestens so gross wie die
    # gesamte auf einer Maschine eingeplante Bearbeitungszeit. Diese gueltige
    # Ungleichung staerkt die LP-Relaxierung deutlich (bester Hebel im Benchmark).
    for k in machines:
        operations_on_machine = [
            i for i in instance.real_operations if k in instance.eligible_machines[i]
        ]
        if operations_on_machine:
            model.addCons(
                C_max >= quicksum(
                    instance.processing_times[i, k] * Y[i, k]
                    for i in operations_on_machine
                ),
                name=f"machine_load_lb[{k}]",
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

    makespan = ""
    solution = None
    if model.getNSols() > 0:
        solution = model.getBestSol()
        makespan = f"{float(model.getSolVal(solution, variables['C_max'])):.4f}"

    with open(filename, "w", encoding="utf-8") as file:
        file.write(f"Status: {model.getStatus()}\n")
        file.write(f"Makespan: {makespan}\n")
        file.write(f"Big M: {float(variables['H']):.4f}\n")
        file.write(f"Berechnungszeit: {float(model.getSolvingTime()):.4f}\n")

        if solution is not None:
            file.write("\nY values:\n")
            for i, k in variables["Y_index"]:
                y_value = int(round(model.getSolVal(solution, variables["Y"][i, k])))
                file.write(f"Y[{i},{k}] = {y_value}\n")
