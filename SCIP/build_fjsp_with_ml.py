import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

from pyscipopt import Model, quicksum
from pyscipopt_ml.torch import add_sequential_constr

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from SCIP.build_fjsp import build_fjsp, write_solution_file


def _load_torch_sequential_model(model_path: str | Path) -> tuple[nn.Sequential, dict]:
    """
    Lade das gespeicherte PyTorch-Bundle aus `simpleNNTest.py` und rekonstruiere
    das reine `nn.Sequential`-Netz fuer das SCIP-Embedding.
    """

    bundle = torch.load(model_path, map_location="cpu", weights_only=False)

    n_inputs = int(bundle["n_inputs"])
    layer_sizes = tuple(int(size) for size in bundle["layer_sizes"])
    state_dict = bundle["model_state_dict"]

    layers: list[nn.Module] = []
    in_features = n_inputs

    for size in layer_sizes:
        layers.append(nn.Linear(in_features, size))
        layers.append(nn.ReLU())
        in_features = size

    layers.append(nn.Linear(in_features, 1))
    sequential_model = nn.Sequential(*layers)

    sequential_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("network."):
            sequential_state_dict[key[len("network."):]] = value
        else:
            sequential_state_dict[key] = value

    sequential_model.load_state_dict(sequential_state_dict)
    sequential_model.eval()

    return sequential_model, bundle


def _load_meta(meta_path: str | Path | None) -> dict:
    if meta_path is None:
        return {}

    with open(meta_path, "r", encoding="utf-8") as file:
        return json.load(file)


def _extract_operation_from_feature(feature_name: str) -> int:
    prefix = "y_op_"
    if not feature_name.startswith(prefix):
        raise ValueError(
            f"Unerwarteter Feature-Name '{feature_name}'. Erwartet wird das Format 'y_op_<id>'."
        )

    try:
        return int(feature_name[len(prefix):])
    except ValueError as exc:
        raise ValueError(
            f"Operation konnte nicht aus Feature '{feature_name}' gelesen werden."
        ) from exc


def build_fjsp_with_ml(
    fjsp: Model,
    instance,
    model_path: str | Path,
    meta_path: str | Path | None = None,
    formulation: str = "sos",
):
    """
    Baut ein SCIP-FJSP-Modell und erweitert es um ein trainiertes PyTorch-MLP,
    das den Makespan aus den Maschinenzuweisungen `Y` vorhersagt.

    `objective_mode`:
    - `predicted`: minimiere den vom MLP geschaetzten Makespan
    """

    model, variables = build_fjsp(fjsp, instance)
    sequential_model, bundle = _load_torch_sequential_model(model_path)
    meta = _load_meta(meta_path)
    # Das ist nicht die inpt order
    feature_columns = bundle.get("feature_columns") or meta.get("input_order")
    if not feature_columns:
        raise ValueError("Keine Feature-Reihenfolge gefunden. Modell oder Meta-Datei ist unvollstaendig.")

    expected_operations = set(instance.real_operations)
    feature_operations = {_extract_operation_from_feature(name) for name in feature_columns}
    if feature_operations != expected_operations:
        raise ValueError(
            "Die ML-Features passen nicht zur Instanz. "
            f"Features enthalten Operationen {sorted(feature_operations)}, "
            f"Instanz benoetigt {sorted(expected_operations)}."
        )

    if len(feature_columns) != bundle["n_inputs"]:
        raise ValueError(
            f"Das Modell erwartet {bundle['n_inputs']} Inputs, die Featureliste hat aber {len(feature_columns)}."
        )

    Y = variables["Y"]
    input_vars = np.empty((1, len(feature_columns)), dtype=object)
    machine_choice_vars = {}

    for col_idx, feature_name in enumerate(feature_columns):
        operation = _extract_operation_from_feature(feature_name)
        eligible_machines = list(instance.eligible_machines[operation])

        machine_var = model.addVar(
            vtype="C",
            lb=float(min(eligible_machines)),
            ub=float(max(eligible_machines)),
            name=f"ml_machine_choice[{operation}]",
        )
        machine_choice_vars[operation] = machine_var
        input_vars[0, col_idx] = machine_var

        model.addCons(
            machine_var
            == quicksum(machine * Y[operation, machine] for machine in eligible_machines),
            name=f"ml_machine_link[{operation}]",
        )

    target_min = float(meta.get("target_min", 0.0))
    target_max = float(meta.get("target_max", variables["H"]))
    ml_makespan = model.addVar(
        vtype="C",
        lb=min(0.0, target_min),
        ub=max(target_max, variables["H"]) + 1e-3,
        name="ml_makespan",
    )
    output_vars = np.array([[ml_makespan]], dtype=object)

    predictor_constr = add_sequential_constr(
        model,
        sequential_model,
        input_vars,
        output_vars,
        unique_naming_prefix="ml_",
        formulation=formulation,
        output_type="regression",
    )

    model.setObjective(variables["C_max"], "minimize")

    variables.update(
        {
            "ml_input_vars": input_vars,
            "ml_machine_choice_vars": machine_choice_vars,
            "ml_makespan": ml_makespan,
            "ml_feature_columns": feature_columns,
            "ml_predictor_constr": predictor_constr,
            "ml_model_path": str(model_path),
            "ml_meta_path": str(meta_path) if meta_path is not None else None,
        }
    )
    return model, variables


def write_solution_file_with_ml(
    model: Model,
    variables,
    instance,
    filename: str | Path = "solution.txt",
):
    """
    Schreibt die normale FJSP-Loesung und ergaenzt ML-spezifische Kennzahlen.
    """

    write_solution_file(model, variables, instance, filename)

    if model.getNSols() == 0:
        return

    solution = model.getBestSol()
    ml_makespan_var = variables.get("ml_makespan")
    objective_mode = variables.get("ml_objective_mode")

    if ml_makespan_var is None:
        return

    ml_value = model.getSolVal(solution, ml_makespan_var)
    exact_value = model.getSolVal(solution, variables["C_max"])

    with open(filename, "a", encoding="utf-8") as file:
        file.write("\n")
        file.write("ML-Erweiterung\n")
        file.write("=" * 70 + "\n")
        file.write(f"ML objective mode: {objective_mode}\n")
        file.write(f"Vorhergesagter Makespan: {ml_value:.4f}\n")
        file.write(f"Exakter Makespan: {exact_value:.4f}\n")
        file.write(f"Abweichung: {abs(exact_value - ml_value):.4f}\n")


if __name__ == "__main__":
    instance_name = "i5_k5_1"
    formulation = "sos"
    time_limit = 30

    instance_path = ROOT_DIR / "data" / "fjsp_instances" / f"{instance_name}.fjsp"
    model_path = ROOT_DIR / "NN Modell" / f"{instance_name}.pt"
    meta_path = ROOT_DIR / "NN Modell" / f"{instance_name}_meta.json"
    solution_path = (
        ROOT_DIR / "data" / "fjsp_solutions" / f"solution_{instance_name}_scip_ml.txt"
    )

    if not instance_path.exists():
        raise FileNotFoundError(f"Instanz nicht gefunden: {instance_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"ML-Modell nicht gefunden: {model_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Meta-Datei nicht gefunden: {meta_path}")

    with open(instance_path, "rb") as file:
        instance = pickle.load(file)

    scip = Model(f"FJSP ML Test {instance_name}")
    if time_limit is not None:
        scip.setParam("limits/time", float(time_limit))

    model, variables = build_fjsp_with_ml(
        scip,
        instance,
        model_path=model_path,
        meta_path=meta_path,
        formulation=formulation
    )

    model.optimize()

    print(f"SCIP-Status: {model.getStatus()}")
    print(f"Anzahl Loesungen: {model.getNSols()}")

    if model.getNSols() > 0:
        print(f"Zielfunktionswert: {model.getObjVal():.4f}")
        write_solution_file_with_ml(model, variables, instance, solution_path)
        print(f"Loesung gespeichert unter: {solution_path}")
    else:
        print("Keine zulaessige Loesung gefunden.")

    model.freeProb()
