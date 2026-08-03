import gurobipy as gp
from gurobipy import GRB
from helper.gurobi_solution_writer import write_model_structure

STATUS_NAMES = {
    GRB.LOADED: "LOADED",
    GRB.OPTIMAL: "OPTIMAL",
    GRB.INFEASIBLE: "INFEASIBLE",
    GRB.INF_OR_UNBD: "INF_OR_UNBD",
    GRB.UNBOUNDED: "UNBOUNDED",
    GRB.CUTOFF: "CUTOFF",
    GRB.ITERATION_LIMIT: "ITERATION_LIMIT",
    GRB.NODE_LIMIT: "NODE_LIMIT",
    GRB.TIME_LIMIT: "TIME_LIMIT",
    GRB.SOLUTION_LIMIT: "SOLUTION_LIMIT",
    GRB.INTERRUPTED: "INTERRUPTED",
    GRB.NUMERIC: "NUMERIC",
    GRB.SUBOPTIMAL: "SUBOPTIMAL",
    GRB.INPROGRESS: "INPROGRESS",
    GRB.USER_OBJ_LIMIT: "USER_OBJ_LIMIT",
}

def build_fjsp(
    fjsp,
    instance,
    extra_duration_upper_bounds=None,
    add_time_constraints=True,
    total_extra_duration_upper_bound=None,
):
    """
    Baut ein MIP-Modell für das Flexible Job Shop Scheduling Problem
    mit Fertigstellungszeiten.

    X[i,j,k] = 1 bedeutet:
        Operation i wird vor Operation j auf Maschine k bearbeitet.

    Mit ``add_time_constraints=False`` werden nur die Variablen, die
    Maschinenzuordnung und die Makespan-Bedingungen erzeugt. Formulierungen
    mit eigenen effektiven Dauern können dadurch ihre stärkeren Zeitbedingungen
    ergänzen, ohne die schwächeren Basisbedingungen doppelt einzubauen.
    """
    
    model = fjsp

    n_machines = instance.num_machines
    machines = list(range(n_machines))
  

    # Randomgenerator, fixe Zuordnung

    # Big-M
    extra_duration_upper_bounds = extra_duration_upper_bounds or {}
    processing_horizon = sum(
        max(instance.processing_times[i, k] for k in instance.eligible_machines[i])
        for i in instance.real_operations
    )
    if total_extra_duration_upper_bound is None:
        extra_duration_horizon = sum(
            float(extra_duration_upper_bounds.get(i, 0.0))
            for i in instance.real_operations
        )
    else:
        extra_duration_horizon = float(total_extra_duration_upper_bound)
        if extra_duration_horizon < 0.0:
            raise ValueError(
                "total_extra_duration_upper_bound must be nonnegative."
            )
    H = processing_horizon + extra_duration_horizon

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

    if add_time_constraints:
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
        "base_time_constraints_enabled": bool(add_time_constraints),
    }

    return model, variables


def write_solution_file(model, variables, instance, filename="solution.txt"):
    """
    Schreibt die gefundene Lösung in eine Textdatei.
    """

    status = STATUS_NAMES.get(model.Status, str(model.Status))
    makespan = ""
    if model.SolCount > 0:
        makespan = f"{float(model.ObjVal):.4f}"

    def statistic(attribute):
        try:
            return float(getattr(model, attribute))
        except (AttributeError, TypeError, ValueError, gp.GurobiError):
            return None

    best_bound = statistic("ObjBound")
    mip_gap = statistic("MIPGap") if model.SolCount > 0 else None
    node_count = statistic("NodeCount")

    def formatted(value, digits):
        return "" if value is None else f"{value:.{digits}f}"

    with open(filename, "w", encoding="utf-8") as file:
        file.write(f"Status: {status}\n")
        file.write(f"Makespan: {makespan}\n")
        file.write(f"Big M: {float(variables['H']):.4f}\n")
        file.write(f"Berechnungszeit: {float(model.Runtime):.4f}\n")
        file.write(f"Best bound: {formatted(best_bound, 6)}\n")
        file.write(f"MIP gap: {formatted(mip_gap, 8)}\n")
        file.write(
            "MIP gap [%]: "
            f"{formatted(100.0 * mip_gap if mip_gap is not None else None, 6)}\n"
        )
        file.write(f"Explored nodes: {formatted(node_count, 0)}\n")
        file.write(f"Solution count: {int(model.SolCount)}\n")

        write_model_structure(file, model)

        if model.SolCount > 0:
            file.write("\nY values:\n")
            for i, k in variables["Y_index"]:
                y_value = int(round(variables["Y"][i, k].X))
                file.write(f"Y[{i},{k}] = {y_value}\n")


if __name__ == '__main__':
    pass
