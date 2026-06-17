import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

from pyscipopt import Model, quicksum
from pyscipopt_ml import add_predictor_constr


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from neuralnetwork.paths import METADATA_FILENAME, MODEL_DIR, MODEL_FILENAME

# Das trainierte Netz und seine Metadaten werden getrennt gespeichert:
# - .pt enthaelt die PyTorch-Gewichte.
# - .json enthaelt Feature-Reihenfolge und Trainingsinformationen.

def _build_sequential_from_state_dict(state_dict) -> nn.Sequential:
    """Rekonstruiert das nn.Sequential anhand der gespeicherten Gewichts-Shapes.

    So bleibt die SCIP-Einbettung unabhaengig von der konkreten Architektur:
    jede zusaetzliche/groessere Schicht im trainierten Netz wird automatisch
    uebernommen (ReLU zwischen aufeinanderfolgenden Linear-Schichten).
    """

    linear_shapes = []
    for key, weight in state_dict.items():
        if key.endswith(".weight") and weight.ndim == 2:
            prefix = key[: -len(".weight")]
            order = int(prefix.split(".")[-1]) if prefix.split(".")[-1].isdigit() else len(linear_shapes)
            out_features, in_features = weight.shape
            linear_shapes.append((order, in_features, out_features))

    if not linear_shapes:
        raise ValueError("Keine Linear-Schichten im state_dict gefunden.")

    linear_shapes.sort(key=lambda item: item[0])

    layers = []
    for idx, (_, in_features, out_features) in enumerate(linear_shapes):
        layers.append(nn.Linear(in_features, out_features))
        if idx < len(linear_shapes) - 1:
            layers.append(nn.ReLU())

    return nn.Sequential(*layers)


def _model_file_path(filename: str) -> Path:
    """Liefert eine Datei aus dem zentralen neuralnetwork/models-Ordner."""

    path = MODEL_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Could not find {filename}. Checked: {path}")

    return path


def _load_metadata():
    """Laedt Feature-Reihenfolge und Trainingsmetadaten."""

    metadata_path = _model_file_path(METADATA_FILENAME)
    with open(metadata_path, encoding="utf-8") as file:
        metadata = json.load(file)

    return metadata_path, metadata


def _load_predictor(input_size: int):
    """Laedt die PyTorch-Gewichte und gibt das nn.Sequential-Netz zurueck."""

    model_path = _model_file_path(MODEL_FILENAME)

    try:
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    except TypeError:
        state_dict = torch.load(model_path, map_location="cpu")

    # Architektur dynamisch aus den gespeicherten Gewichten ableiten, damit die
    # Einbettung jede trainierte Netzgroesse uebernimmt.
    network = _build_sequential_from_state_dict(state_dict)

    # Das Sequential erwartet Keys wie "0.weight"; der gespeicherte state_dict
    # nutzt den "network."-Praefix der Wrapper-Klasse. Praefix abstreifen.
    normalized_state_dict = {
        (key[len("network."):] if key.startswith("network.") else key): value
        for key, value in state_dict.items()
    }
    network.load_state_dict(normalized_state_dict)

    first_layer = next(layer for layer in network if isinstance(layer, nn.Linear))
    if first_layer.in_features != input_size:
        raise ValueError(
            f"Das gespeicherte Netz erwartet {first_layer.in_features} Inputs, "
            f"die Metadaten enthalten aber {input_size} Features."
        )

    # eval() deaktiviert Trainingseffekte. Fuer Linear/ReLU ist das hier
    # nicht kritisch, aber es ist die korrekte PyTorch-Inferenz-Einstellung.
    network.eval()

    # pyscipopt_ml hat einen Konverter fuer torch.nn.Sequential.
    return model_path, network


def build_fjsp(
    fjsp: Model,
    instance,
    formulation: str = "sos",
):
    """
    Baut das SCIP-FJSP-Modell und ergaenzt einen eingebetteten NN-Predictor.

    Eingaben:
    - fjsp: leeres oder vorbereitetes PySCIPOpt Model
    - instance: FJSP-Instanz mit Operationen, Maschinen, Zeiten und Vorgaengern

    Rueckgabe:
    - model: SCIP-Modell mit allen Scheduling- und ML-Constraints
    - variables: Dictionary mit wichtigen Variablen/Indexmengen fuer Auswertung

    Die Prediction-Ausgabe wird mit MAE + RMSE als Korridor eingebunden:
        predicted_makespan - (MAE + RMSE) <= C_max

    """

    model = fjsp

    n_machines = instance.num_machines
    machines = list(range(n_machines))

    setup_times = getattr(instance, "setup_times", {})
    max_setup_time = max(setup_times.values(), default=0)

    # Big-M-Konstante fuer die Nicht-Ueberlappungsbedingungen.
    # Sie ist eine grobe obere Schranke auf die maximale Fertigstellungszeit:
    # pro Operation die laengste erlaubte Bearbeitungszeit plus Setup-Puffer.
    H = sum(
        max(instance.processing_times[i, k] for k in instance.eligible_machines[i])
        for i in instance.real_operations
    ) + len(instance.real_operations) * max_setup_time

    # C[i] = Fertigstellungszeit von Operation i.
    C = {
        i: model.addVar(vtype="C", lb=0.0, name=f"C[{i}]")
        for i in instance.real_operations
    }

    # C_max = Makespan, also maximale Fertigstellungszeit aller Jobs.
    C_max = model.addVar(lb=0.0, vtype="C", name="C_max")

    # Y[i,k] = 1, wenn Operation i auf Maschine k ausgefuehrt wird.
    Y_index = [
        (i, k)
        for i in instance.real_operations
        for k in instance.eligible_machines[i]
    ]

    Y = {
        (i, k): model.addVar(vtype="B", name=f"Y[{i},{k}]")
        for i, k in Y_index
    }

    # X[i,j,k] kodiert die Reihenfolge zweier Operationen i und j auf Maschine k.
    # Die Variable wird nur fuer Paare angelegt, die beide auf k laufen koennen.
    X_index = []
    for idx_i, i in enumerate(instance.real_operations):
        for j in instance.real_operations[idx_i + 1:]:
            common_machines = set(instance.eligible_machines[i]) & set(
                instance.eligible_machines[j]
            )
            for k in common_machines:
                X_index.append((i, j, k))

    X = {
        (i, j, k): model.addVar(vtype="B", name=f"X[{i},{j},{k}]")
        for i, j, k in X_index
    }

    model.setObjective(C_max, "minimize")

    # Jede Operation muss genau einer ihrer erlaubten Maschinen zugeordnet werden.
    for i in instance.real_operations:
        model.addCons(
            quicksum(Y[i, k] for k in instance.eligible_machines[i]) == 1,
            name=f"assignment[{i}]",
        )

    # Technologische Reihenfolge innerhalb eines Jobs:
    # Wenn j Vorgaenger von i ist, darf i erst nach Fertigstellung von j
    # plus eigener Bearbeitungszeit fertig sein.
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

    # Nicht-Ueberlappung auf derselben Maschine:
    # Falls i und j beide Maschine k nutzen, erzwingt X eine der beiden
    # Reihenfolgen. Big-M schaltet die Bedingung aus, wenn mindestens eine
    # Operation nicht auf Maschine k liegt.
    for i, j, k in X_index:
        setup_j_before_i = setup_times.get((j, i, k), 0)
        setup_i_before_j = setup_times.get((i, j, k), 0)

        model.addCons(
            C[i] >= C[j] + setup_j_before_i + instance.processing_times[i, k]
            - H * (2 + X[i, j, k] - Y[i, k] - Y[j, k]),
            name=f"nonoverlap_j_before_i[{j}_{i}_{k}]",
        )

        model.addCons(
            C[j] >= C[i] + setup_i_before_j + instance.processing_times[j, k]
            - H * (3 - X[i, j, k] - Y[i, k] - Y[j, k]),
            name=f"nonoverlap_i_before_j[{i}_{j}_{k}]",
        )

    # Jede Operation kann nicht vor ihrer eigenen Bearbeitungszeit fertig sein.
    for i in instance.real_operations:
        processing_time_i = quicksum(
            Y[i, k] * instance.processing_times[i, k]
            for k in instance.eligible_machines[i]
        )
        model.addCons(
            C[i] >= processing_time_i,
            name=f"completion_lb[{i}]",
        )

    # C_max muss mindestens so gross sein wie die Endoperation jedes Jobs.
    for u in instance.jobs:
        end_operation = instance.job_end_operations[u]
        model.addCons(
            C_max >= C[end_operation],
            name=f"makespan[{u}]",
        )

    # Ab hier beginnt die ML-Einbettung.
    # ---------------------------------------------------------------------------------------------------
    # metadata["feature_columns"] bestimmt die exakte Reihenfolge der Features,
    # in der das Netz trainiert wurde. Diese Reihenfolge muss beim Einbetten
    # identisch sein, sonst interpretiert das Netz die Inputs falsch.
    metadata_path, metadata = _load_metadata()
    feature_columns = metadata["feature_columns"]
    missing_error_keys = [
        key
        for key in ("best_val_mse", "best_val_mae_makespan")
        if key not in metadata
    ]
    if missing_error_keys:
        raise ValueError(
            "Die ML-Metadaten enthalten nicht alle Fehlerwerte fuer die "
            f"Prediction-Schranke: {missing_error_keys}. Bitte das neuronale "
            "Netz neu trainieren."
        )
    prediction_mse = float(metadata["best_val_mse"])
    prediction_mae = float(metadata["best_val_mae_makespan"])
    prediction_rmse = math.sqrt(max(prediction_mse, 0.0))
    prediction_error_margin = prediction_mae + prediction_rmse

    model_path, predictor = _load_predictor(input_size=len(feature_columns))

    # Setup-Statistiken pro Maschine fuer die ML-Features.
    setup_times_by_machine = {machine: [] for machine in machines}
    for (_, _, machine), setup_time in setup_times.items():
        setup_times_by_machine[machine].append(setup_time)

    s_mean = {
        machine: (sum(values) / len(values)) if values else 0
        for machine, values in setup_times_by_machine.items()
    }

    # Feature-Variablen fuer das neuronale Netz.
    # Die Formeln spiegeln die CSV-Erzeugung in helper/start_solve_ins.py.
    feature_vars_by_name = {}
    for machine in machines:
        forced_count_lb = 0
        count_ub = 0
        machine_load_lb = -s_mean[machine]
        machine_load_ub = -s_mean[machine]
        for operation in instance.real_operations:
            eligible_machines = list(instance.eligible_machines[operation])
            if machine not in eligible_machines:
                continue

            count_ub += 1
            machine_load_coefficient = (
                instance.processing_times[operation, machine] + s_mean[machine]
            )
            machine_load_ub += machine_load_coefficient
            if len(eligible_machines) == 1:
                forced_count_lb += 1
                machine_load_lb += machine_load_coefficient

        count_feature_name = f"machine_{machine}_assigned_operation_count"
        count_var = model.addVar(
            vtype="C",
            lb=float(forced_count_lb),
            ub=float(count_ub),
            name=count_feature_name,
        )
        model.addCons(
            count_var
            == quicksum(
                Y[i, machine]
                for i in instance.real_operations
                if machine in instance.eligible_machines[i]
            ),
            name=f"{count_feature_name}_def",
        )
        feature_vars_by_name[count_feature_name] = count_var

        feature_name = f"machine_{machine}_load_mean_setup"
        feature_var = model.addVar(
            vtype="C",
            lb=float(machine_load_lb),
            ub=float(machine_load_ub),
            name=feature_name,
        )
        model.addCons(
            feature_var
            == -s_mean[machine]
            + quicksum(
                (instance.processing_times[i, machine] + s_mean[machine])
                * Y[i, machine]
                for i in instance.real_operations
                if machine in instance.eligible_machines[i]
            ),
            name=f"{feature_name}_def",
        )
        feature_vars_by_name[feature_name] = feature_var

    for job in sorted(instance.jobs):
        feature_name = f"job_{job}_path_mean_setup"
        job_path_lb = 0.0
        job_path_ub = 0.0
        job_path_terms = []
        for operation_idx, operation in enumerate(instance.jobs[job]):
            operation_coefficients = []
            for machine in instance.eligible_machines[operation]:
                setup_part = s_mean[machine] if operation_idx > 0 else 0.0
                coefficient = (
                    instance.processing_times[operation, machine] + setup_part
                )
                operation_coefficients.append(coefficient)
                job_path_terms.append(coefficient * Y[operation, machine])

            job_path_lb += min(operation_coefficients)
            job_path_ub += max(operation_coefficients)

        feature_var = model.addVar(
            vtype="C",
            lb=float(job_path_lb),
            ub=float(job_path_ub),
            name=feature_name,
        )

        model.addCons(
            feature_var == quicksum(job_path_terms),
            name=f"{feature_name}_def",
        )
        feature_vars_by_name[feature_name] = feature_var

    mean_time_lb = 0.0
    mean_time_ub = 0.0
    for operation in instance.real_operations:
        operation_coefficients = [
            instance.processing_times[operation, machine] + s_mean[machine]
            for machine in instance.eligible_machines[operation]
        ]
        mean_time_lb += min(operation_coefficients)
        mean_time_ub += max(operation_coefficients)

    mean_time_lb /= len(instance.real_operations)
    mean_time_ub /= len(instance.real_operations)

    mean_assigned_operation_time = model.addVar(
        vtype="C",
        lb=float(mean_time_lb),
        ub=float(mean_time_ub),
        name="mean_assigned_operation_time_mean_setup",
    )
    model.addCons(
        mean_assigned_operation_time
        == quicksum(
            (instance.processing_times[i, machine] + s_mean[machine]) * Y[i, machine]
            for i in instance.real_operations
            for machine in instance.eligible_machines[i]
        )
        / len(instance.real_operations),
        name="mean_assigned_operation_time_mean_setup_def",
    )
    feature_vars_by_name[
        "mean_assigned_operation_time_mean_setup"
    ] = mean_assigned_operation_time

    missing_features = [
        feature_name
        for feature_name in feature_columns
        if feature_name not in feature_vars_by_name
    ]
    if missing_features:
        raise ValueError(
            "Das SCIP-ML-Modell kann diese NN-Features noch nicht erzeugen: "
            f"{missing_features}. Bitte Feature-Set oder SCIP-Einbettung anpassen."
        )

    nn_input_vars = [feature_vars_by_name[feature_name] for feature_name in feature_columns]

    # Output-Variablen des eingebetteten Netzes.
    #
    # add_predictor_constr erzwingt:
    #   predicted_makespan == predictor(nn_input_vars)
    predicted_makespan = model.addVar(
        vtype="C",
        lb=-model.infinity(),
        ub=None,
        name="predicted_makespan",
    )
    predicted_makespan_minus_error = model.addVar(
        vtype="C",
        lb=None,
        ub=None,
        name="predicted_makespan_minus_error",
    )

    # Zentrale ML-Einbettung
    predictor_constraint = add_predictor_constr(
        model,
        predictor,
        np.array([nn_input_vars], dtype=object),
        np.array([[predicted_makespan]], dtype=object),
        unique_naming_prefix="makespan_nn_",
        formulation=formulation,
        output_type="regression",
    )

    model.addCons(
        predicted_makespan_minus_error
        == predicted_makespan - prediction_error_margin,
        name="predicted_makespan_minus_error_def",
    )

    # Nur untere Schranke: C_max >= pred - margin.
    prediction_lower_bound_constraint = model.addCons(
        C_max >= predicted_makespan_minus_error,
        name="prediction_lower_bound",
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
        "setup_times": setup_times,
        "machines": machines,
        "jobs": instance.jobs,
        "X_index": X_index,
        "Y_index": Y_index,
        "feature_vars": feature_vars_by_name,
        "ml_feature_columns": feature_columns,
        "nn_input_vars": nn_input_vars,
        "predicted_makespan": predicted_makespan,
        "predicted_makespan_minus_error": predicted_makespan_minus_error,
        "prediction_mse": prediction_mse,
        "prediction_mae": prediction_mae,
        "prediction_rmse": prediction_rmse,
        "prediction_error_margin": prediction_error_margin,
        "prediction_lower_bound_constraint": prediction_lower_bound_constraint,
        "predictor_constraint": predictor_constraint,
        "s_mean": s_mean,
        "ml_model_path": model_path,
        "ml_metadata_path": metadata_path,
        "ml_formulation": formulation,
    }

    return model, variables


def write_solution_file(model: Model, variables, _instance, filename="solution.txt"):
    """
    Schreibt die gefundene SCIP-Loesung inklusive ML-Lower-Bound-Information.
    """

    makespan = ""
    predicted_makespan = ""
    prediction_mse = variables["prediction_mse"]
    prediction_mae = variables["prediction_mae"]
    prediction_rmse = variables["prediction_rmse"]
    prediction_error_margin = variables["prediction_error_margin"]
    prediction_lower_bound = ""
    feature_values = {}
    solution = None

    if model.getNSols() > 0:
        solution = model.getBestSol()
        makespan = f"{float(model.getSolVal(solution, variables['C_max'])):.4f}"
        predicted_makespan = f"{float(model.getSolVal(solution, variables['predicted_makespan'])):.4f}"
        prediction_lower_bound = f"{float(model.getSolVal(solution, variables['predicted_makespan_minus_error'])):.4f}"
        for feature_name in variables["ml_feature_columns"]:
            feature_var = variables["feature_vars"][feature_name]
            feature_values[feature_name] = float(model.getSolVal(solution, feature_var))

    with open(filename, "w", encoding="utf-8") as file:
        file.write(f"Status: {model.getStatus()}\n")
        file.write(f"Makespan: {makespan}\n")
        file.write(f"Predicted makespan: {predicted_makespan}\n")
        file.write(f"Prediction MSE: {prediction_mse:.4f}\n")
        file.write(f"Prediction RMSE: {prediction_rmse:.4f}\n")
        file.write(f"Prediction MAE: {prediction_mae:.4f}\n")
        file.write(f"Prediction error margin: {prediction_error_margin:.4f}\n")
        file.write(f"Prediction lower bound: {prediction_lower_bound}\n")
        file.write(f"ML model: {variables['ml_model_path']}\n")
        file.write(f"ML metadata: {variables['ml_metadata_path']}\n")
        file.write(f"Big M: {float(variables['H']):.4f}\n")
        file.write(f"Berechnungszeit: {float(model.getSolvingTime()):.4f}\n")

        if solution is not None:
            file.write("\nML feature values:\n")
            for feature_name in variables["ml_feature_columns"]:
                file.write(f"{feature_name}: {feature_values[feature_name]:.4f}\n")

            file.write("\nY values:\n")
            for i, k in variables["Y_index"]:
                y_value = int(round(model.getSolVal(solution, variables["Y"][i, k])))
                file.write(f"Y[{i},{k}] = {y_value}\n")


def build_and_solve(
    instance_name: str = "i5_k5_1",
    instance_path: str | Path | None = None,
    solution_path: str | Path | None = None,
    formulation: str = "sos",
    time_limit: float | None = 300.0,
    write_solution: bool = True,
    verbose: bool = True,
):
    if instance_path is None:
        instance_path = ROOT_DIR / "data" / "fjsp_instances" / f"{instance_name}.fjsp"
    else:
        instance_path = Path(instance_path)

    if solution_path is None:
        solution_path = (
            ROOT_DIR
            / "data"
            / "fjsp_solutions"
            / f"solution_{instance_name}_scip_ml.txt"
        )
    else:
        solution_path = Path(solution_path)

    if not instance_path.exists():
        raise FileNotFoundError(f"Instance not found: {instance_path}")

    with open(instance_path, "rb") as file:
        instance = pickle.load(file)

    # SCIP Modell
    scip = Model(f"FJSP SCIP ML {instance_name}")
    if time_limit is not None and time_limit > 0:
        scip.setParam("limits/time", float(time_limit))
        if verbose:
            print(f"[info] SCIP time limit: {time_limit}s")

    model, variables = build_fjsp(scip, instance, formulation=formulation)

    if verbose:
        print(
            f"[info] Loese FJSP-ML: instance={instance_name}, "
            f"operations={len(instance.real_operations)}, "
            f"machines={instance.num_machines}, formulation={formulation}"
        )
        print(f"[info] ML model: {variables['ml_model_path']}")
        print(f"[info] ML metadata: {variables['ml_metadata_path']}")

    model.optimize()

    status = model.getStatus()
    print(f"\n=== SCIP Status: {status} ===")

    result = {"status": status, "instance": instance_name}
    if model.getNSols() > 0:
        solution = model.getBestSol()
        makespan = float(model.getSolVal(solution, variables["C_max"]))
        predicted_makespan = float(
            model.getSolVal(solution, variables["predicted_makespan"])
        )
        prediction_lower_bound = float(
            model.getSolVal(solution, variables["predicted_makespan_minus_error"])
        )
        prediction_mse = variables["prediction_mse"]
        prediction_mae = variables["prediction_mae"]
        prediction_rmse = variables["prediction_rmse"]
        prediction_error_margin = variables["prediction_error_margin"]
        feature_values = {
            feature_name: float(
                model.getSolVal(
                    solution,
                    variables["feature_vars"][feature_name],
                )
            )
            for feature_name in variables["ml_feature_columns"]
        }

        print(f"Objective / Makespan:           {makespan:.4f}")
        print(f"Predicted makespan:             {predicted_makespan:.4f}")
        print(f"Prediction MSE:                 {prediction_mse:.4f}")
        print(f"Prediction RMSE:                {prediction_rmse:.4f}")
        print(f"Prediction MAE:                 {prediction_mae:.4f}")
        print(f"Prediction error margin:        {prediction_error_margin:.4f}")
        print(f"Prediction lower bound:         {prediction_lower_bound:.4f}")
        print(f"ML input features:              {len(feature_values)}")
        for feature_name, feature_value in feature_values.items():
            print(f"  {feature_name}: {feature_value:.4f}")

        try:
            emb_err = float(np.max(variables["predictor_constraint"].get_error()))
            print(f"max embedding error (lib):      {emb_err:.2e}")
        except Exception:
            pass

        if write_solution:
            solution_path.parent.mkdir(parents=True, exist_ok=True)
            write_solution_file(model, variables, instance, solution_path)
            print(f"Solution written to:            {solution_path}")

        result.update(
            {
                "objective": makespan,
                "makespan": makespan,
                "predicted_makespan": predicted_makespan,
                "prediction_mse": prediction_mse,
                "prediction_mae": prediction_mae,
                "prediction_rmse": prediction_rmse,
                "prediction_error_margin": prediction_error_margin,
                "prediction_lower_bound": prediction_lower_bound,
                "feature_values": feature_values,
                "solution_path": str(solution_path) if write_solution else None,
                "ml_model_path": str(variables["ml_model_path"]),
                "ml_metadata_path": str(variables["ml_metadata_path"]),
            }
        )
    else:
        print("Keine zulaessige Loesung gefunden.")

    return result, model
