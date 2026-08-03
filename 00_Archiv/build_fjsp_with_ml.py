import importlib
import json
import math
import sys
from pathlib import Path

import gurobipy as gp
import torch
from gurobipy import GRB

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

_base_fjsp = importlib.import_module("03_Gurobi.build_fjsp")
STATUS_NAMES = _base_fjsp.STATUS_NAMES
build_base_fjsp = _base_fjsp.build_fjsp
_paths = importlib.import_module("00_Archiv.paths")
METADATA_FILENAME = _paths.METADATA_FILENAME
MODEL_DIR = _paths.MODEL_DIR
MODEL_FILENAME = _paths.MODEL_FILENAME


def _model_file_path(filename: str) -> Path:
    path = MODEL_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Could not find {filename}. Checked: {path}")

    return path


def _load_metadata():
    metadata_path = _model_file_path(METADATA_FILENAME)
    with open(metadata_path, encoding="utf-8") as file:
        metadata = json.load(file)

    return metadata_path, metadata


def _load_linear_layers():
    """Laedt die trainierten Gewichte als Listen fuer die Gurobi-Constraints."""
    model_path = _model_file_path(MODEL_FILENAME)

    try:
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    except TypeError:
        state_dict = torch.load(model_path, map_location="cpu")

    layers = []
    for key, weight in state_dict.items():
        if not key.endswith(".weight") or weight.ndim != 2:
            continue

        prefix = key[: -len(".weight")]
        bias_key = f"{prefix}.bias"
        if bias_key not in state_dict:
            raise ValueError(f"Missing bias tensor for layer '{prefix}'.")

        try:
            layer_order = int(prefix.split(".")[-1])
        except ValueError:
            layer_order = len(layers)

        layers.append(
            (
                layer_order,
                weight.detach().cpu().tolist(),
                state_dict[bias_key].detach().cpu().tolist(),
            )
        )

    if not layers:
        raise ValueError(f"No linear layers found in {model_path}.")

    layers.sort(key=lambda item: item[0])
    return model_path, [(W, b) for _, W, b in layers]


def _linear_expression(weights_row, bias, input_vars):
    expression = gp.LinExpr(float(bias))
    for coefficient, variable in zip(weights_row, input_vars):
        coefficient = float(coefficient)
        if coefficient:
            expression += coefficient * variable

    return expression


def _add_neural_network_constraints(model, layers, input_vars, predicted_makespan):
    current_vars = input_vars
    for layer_idx, (weights, biases) in enumerate(layers):
        is_last_layer = layer_idx == len(layers) - 1
        next_vars = []

        if len(weights) != len(biases):
            raise ValueError(
                f"Layer {layer_idx} has {len(weights)} weight rows but "
                f"{len(biases)} bias values."
            )

        for neuron_idx, (weights_row, bias) in enumerate(zip(weights, biases)):
            if len(weights_row) != len(current_vars):
                raise ValueError(
                    f"Layer {layer_idx}, neuron {neuron_idx} expects "
                    f"{len(weights_row)} inputs, got {len(current_vars)}."
                )

            linear_expr = _linear_expression(weights_row, bias, current_vars)
            if is_last_layer:
                if len(weights) != 1:
                    raise ValueError("The output layer must have exactly one neuron.")
                model.addConstr(
                    predicted_makespan == linear_expr,
                    name="nn_output_def",
                )
                next_vars.append(predicted_makespan)
            else:
                z_var = model.addVar(
                    lb=-GRB.INFINITY,
                    vtype=GRB.CONTINUOUS,
                    name=f"nn_layer_{layer_idx}_z[{neuron_idx}]",
                )
                relu_var = model.addVar(
                    lb=0.0,
                    vtype=GRB.CONTINUOUS,
                    name=f"nn_layer_{layer_idx}_relu[{neuron_idx}]",
                )
                model.addConstr(
                    z_var == linear_expr,
                    name=f"nn_layer_{layer_idx}_linear[{neuron_idx}]",
                )
                model.addGenConstrMax(
                    relu_var,
                    [z_var],
                    constant=0.0,
                    name=f"nn_layer_{layer_idx}_relu_def[{neuron_idx}]",
                )
                next_vars.append(relu_var)

        current_vars = next_vars

    return current_vars[0]


def _add_feature_variables(model, instance, variables):
    Y = variables["Y"]
    machines = variables["machines"]

    feature_vars_by_name = {}

    for machine in machines:
        forced_count_lb = 0
        count_ub = 0
        machine_load_lb = 0.0
        machine_load_ub = 0.0
        for operation in instance.real_operations:
            eligible_machines = list(instance.eligible_machines[operation])
            if machine not in eligible_machines:
                continue

            count_ub += 1
            machine_load_coefficient = instance.processing_times[operation, machine]
            machine_load_ub += machine_load_coefficient
            if len(eligible_machines) == 1:
                forced_count_lb += 1
                machine_load_lb += machine_load_coefficient

        count_feature_name = f"machine_{machine}_assigned_operation_count"
        count_var = model.addVar(
            lb=float(forced_count_lb),
            ub=float(count_ub),
            vtype=GRB.CONTINUOUS,
            name=count_feature_name,
        )
        model.addConstr(
            count_var
            == gp.quicksum(
                Y[i, machine]
                for i in instance.real_operations
                if machine in instance.eligible_machines[i]
            ),
            name=f"{count_feature_name}_def",
        )
        feature_vars_by_name[count_feature_name] = count_var

        feature_name = f"machine_{machine}_processing_load"
        feature_var = model.addVar(
            lb=float(machine_load_lb),
            ub=float(machine_load_ub),
            vtype=GRB.CONTINUOUS,
            name=feature_name,
        )
        model.addConstr(
            feature_var
            == gp.quicksum(
                instance.processing_times[i, machine] * Y[i, machine]
                for i in instance.real_operations
                if machine in instance.eligible_machines[i]
            ),
            name=f"{feature_name}_def",
        )
        feature_vars_by_name[feature_name] = feature_var

    for job in sorted(instance.jobs):
        feature_name = f"job_{job}_processing_path"
        job_path_lb = 0.0
        job_path_ub = 0.0
        job_path_terms = []
        for operation in instance.jobs[job]:
            operation_coefficients = []
            for machine in instance.eligible_machines[operation]:
                coefficient = instance.processing_times[operation, machine]
                operation_coefficients.append(coefficient)
                job_path_terms.append(coefficient * Y[operation, machine])

            job_path_lb += min(operation_coefficients)
            job_path_ub += max(operation_coefficients)

        feature_var = model.addVar(
            lb=float(job_path_lb),
            ub=float(job_path_ub),
            vtype=GRB.CONTINUOUS,
            name=feature_name,
        )

        model.addConstr(
            feature_var == gp.quicksum(job_path_terms),
            name=f"{feature_name}_def",
        )
        feature_vars_by_name[feature_name] = feature_var

    return feature_vars_by_name


def build_fjsp(fjsp, instance):
    """
    Baut das Gurobi-FJSP-Modell und ergaenzt einen eingebetteten NN-Predictor.

    Nur untere Schranke: C_max >= predicted_makespan - (MAE + RMSE).
    Eine harte obere Schranke wuerde das Modell unzulaessig machen, wenn die
    NN-Vorhersage ungenau ist, und wird daher weggelassen.

    """

    model, variables = build_base_fjsp(fjsp, instance)

    metadata_path, metadata = _load_metadata()
    feature_columns = metadata["feature_columns"]
    setup_features = [
        feature_name
        for feature_name in feature_columns
        if "setup" in feature_name.lower()
    ]
    if setup_features:
        raise ValueError(
            "Das gespeicherte neuronale Netz verwendet noch Setup-Zeit-Features. "
            "Bitte Fixed-Y-Daten ohne Setup-Zeiten neu erzeugen und das Netz "
            "anschliessend neu trainieren."
        )
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

    model_path, layers = _load_linear_layers()

    if int(metadata.get("input_size", len(feature_columns))) != len(feature_columns):
        raise ValueError("input_size in den ML-Metadaten passt nicht zur Featureliste.")
    first_layer_input_size = len(layers[0][0][0])
    if first_layer_input_size != len(feature_columns):
        raise ValueError(
            f"Das gespeicherte Netz erwartet {first_layer_input_size} Inputs, "
            f"die Metadaten enthalten aber {len(feature_columns)} Features."
        )

    feature_vars_by_name = _add_feature_variables(model, instance, variables)
    missing_features = [
        feature_name
        for feature_name in feature_columns
        if feature_name not in feature_vars_by_name
    ]
    if missing_features:
        raise ValueError(
            "Das Gurobi-ML-Modell kann diese NN-Features noch nicht erzeugen: "
            f"{missing_features}. Bitte Feature-Set oder Gurobi-Einbettung anpassen."
        )

    nn_input_vars = [
        feature_vars_by_name[feature_name]
        for feature_name in feature_columns
    ]

    predicted_makespan = model.addVar(
        lb=-GRB.INFINITY,
        vtype=GRB.CONTINUOUS,
        name="predicted_makespan",
    )
    _add_neural_network_constraints(model, layers, nn_input_vars, predicted_makespan)

    predicted_makespan_minus_error = model.addVar(
        lb=-GRB.INFINITY,
        vtype=GRB.CONTINUOUS,
        name="predicted_makespan_minus_error",
    )
    model.addConstr(
        predicted_makespan_minus_error
        == predicted_makespan - prediction_error_margin,
        name="predicted_makespan_minus_error_def",
    )

    # Nur untere Schranke verwenden.
    prediction_lower_bound_constraint = model.addConstr(
        variables["C_max"] >= predicted_makespan_minus_error,
        name="prediction_lower_bound",
    )

    variables.update(
        {
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
            "ml_model_path": model_path,
            "ml_metadata_path": metadata_path,
        }
    )

    model.update()
    return model, variables


def write_solution_file(model, variables, _instance, filename="solution.txt"):
    status = STATUS_NAMES.get(model.Status, str(model.Status))
    makespan = ""
    predicted_makespan = ""
    prediction_lower_bound = ""
    feature_values = {}

    if model.SolCount > 0:
        makespan = f"{float(variables['C_max'].X):.4f}"
        predicted_makespan = f"{float(variables['predicted_makespan'].X):.4f}"
        prediction_lower_bound = (
            f"{float(variables['predicted_makespan_minus_error'].X):.4f}"
        )
        for feature_name in variables["ml_feature_columns"]:
            feature_values[feature_name] = float(
                variables["feature_vars"][feature_name].X
            )

    with open(filename, "w", encoding="utf-8") as file:
        file.write(f"Status: {status}\n")
        file.write(f"Makespan: {makespan}\n")
        file.write(f"Predicted makespan: {predicted_makespan}\n")
        file.write(f"Prediction MSE: {variables['prediction_mse']:.4f}\n")
        file.write(f"Prediction RMSE: {variables['prediction_rmse']:.4f}\n")
        file.write(f"Prediction MAE: {variables['prediction_mae']:.4f}\n")
        file.write(
            f"Prediction error margin: {variables['prediction_error_margin']:.4f}\n"
        )
        file.write(f"Prediction lower bound: {prediction_lower_bound}\n")
        file.write(f"ML model: {variables['ml_model_path']}\n")
        file.write(f"ML metadata: {variables['ml_metadata_path']}\n")
        file.write(f"Big M: {float(variables['H']):.4f}\n")
        file.write(f"Berechnungszeit: {float(model.Runtime):.4f}\n")

        if model.SolCount > 0:
            file.write("\nML feature values:\n")
            for feature_name in variables["ml_feature_columns"]:
                file.write(f"{feature_name}: {feature_values[feature_name]:.4f}\n")

            file.write("\nY values:\n")
            for i, k in variables["Y_index"]:
                y_value = int(round(variables["Y"][i, k].X))
                file.write(f"Y[{i},{k}] = {y_value}\n")
