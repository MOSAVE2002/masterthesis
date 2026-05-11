import gurobipy as gp
from gurobipy import GRB


def parse_standard_fjsp_lines(lines, one_based_operations=True):
    """
    Parst das Standard-FJSP-Instanzformat.

    Erwartetes Format:
        Erste Zeile:
            n_jobs n_machines avg_machines_per_operation

        Jede weitere Zeile beschreibt einen Job:
            n_operations
            n_eligible_machines_op_1 machine processing_time ...
            n_eligible_machines_op_2 machine processing_time ...
            ...

    Beispiel:
        2 3 2.0
        2 2 0 5 1 6 1 2 7
        1 3 0 4 1 5 2 6

    Rückgabe:
        instance-Dictionary für build_fjsp().
    """

    lines = lines.strip().splitlines()

    lines = [line.strip() for line in lines if line.strip()]

    header = lines[0].split()
    n_jobs = int(header[0])
    n_machines = int(header[1])

    machines = list(range(n_machines))

    jobs = {}
    eligible_machines = {}
    processing_times = {}
    predecessors = {}
    job_end_operations = {}
    real_operations = []

    operation_id = 1 if one_based_operations else 0

    if len(lines) - 1 != n_jobs:
        raise ValueError(
            f"Die Kopfzeile gibt {n_jobs} Jobs an, "
            f"aber es wurden {len(lines) - 1} Job-Zeilen gefunden."
        )

    for job_idx in range(1, n_jobs + 1):
        tokens = list(map(int, lines[job_idx].split()))

        pos = 0
        n_operations = tokens[pos]
        pos += 1

        job_operations = []

        for op_pos in range(n_operations):
            current_operation = operation_id
            operation_id += 1

            job_operations.append(current_operation)
            real_operations.append(current_operation)

            n_eligible = tokens[pos]
            pos += 1

            eligible = []

            for _ in range(n_eligible):
                machine = tokens[pos]
                processing_time = tokens[pos + 1]
                pos += 2

                if machine < 0 or machine >= n_machines:
                    raise ValueError(
                        f"Ungültige Maschine {machine} in Job {job_idx}, "
                        f"Operation {op_pos + 1}. Zulässig sind 0 bis {n_machines - 1}."
                    )

                eligible.append(machine)
                processing_times[current_operation, machine] = processing_time

            eligible_machines[current_operation] = eligible

        if pos != len(tokens):
            raise ValueError(
                f"Job-Zeile {job_idx} wurde nicht vollständig geparst. "
                f"Position {pos}, aber {len(tokens)} Tokens vorhanden."
            )

        jobs[job_idx] = job_operations

        # Lineare Standardpräzedenz innerhalb eines Jobs:
        # o_1 -> o_2 -> ... -> o_n
        for idx, op in enumerate(job_operations):
            if idx == 0:
                predecessors[op] = []
            else:
                predecessors[op] = [job_operations[idx - 1]]

        job_end_operations[job_idx] = job_operations[-1]

    instance = {
        "n_jobs": n_jobs,
        "n_machines": n_machines,
        "jobs": jobs,
        "machines": machines,
        "eligible_machines": eligible_machines,
        "processing_times": processing_times,
        "predecessors": predecessors,
        "job_end_operations": job_end_operations,
        "real_operations": real_operations,
    }

    return instance


def build_fjsp(fjsp, instance):
    """
    Baut ein MIP-Modell für das Flexible Job Shop Scheduling Problem
    mit Fertigstellungszeiten.

    X[i,j,k] = 1 bedeutet:
        Operation i wird vor Operation j auf Maschine k bearbeitet.
    """

    if fjsp is None:
        model = gp.Model("FJSP")
    else:
        model = fjsp

    jobs = list(instance["jobs"].keys())
    machines = list(instance["machines"])

    operations = sorted({
        operation
        for ops_of_job in instance["jobs"].values()
        for operation in ops_of_job
    })

    eligible_machines = instance["eligible_machines"]
    processing_times = instance["processing_times"]
    predecessors = instance.get("predecessors", {i: [] for i in operations})
    job_end_operations = instance["job_end_operations"]
    real_operations = list(instance.get("real_operations", operations))

    # Big-M
    H = sum(
        max(processing_times[i, k] for k in eligible_machines[i])
        for i in real_operations
    )

    # -------------------------
    # Variablen
    # -------------------------

    C = model.addVars(
        operations,
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
    for i in operations:
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
        "operations": operations,
        "real_operations": real_operations,
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
    machines = variables["machines"]

    eligible_machines = instance["eligible_machines"]
    processing_times = instance["processing_times"]

    rows = []

    for i in real_operations:
        assigned_machine = None
        processing_time = None

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