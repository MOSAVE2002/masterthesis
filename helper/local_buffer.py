"""Expose one canonical repair-buffer label contract to GNN modules.

The constants and metadata function are re-exported from the simulation module
so dataset generation, training and solver embedding cannot silently diverge in
their target name, semantics or numerical label settings.
"""

import importlib


_simulation = importlib.import_module("05_Simulation.simulation")

TARGET_COLUMN = _simulation.TARGET_COLUMN
JOB_TARGET = _simulation.JOB_TARGET
LABEL_METHOD = _simulation.LABEL_METHOD
LABEL_SOURCE = _simulation.LABEL_SOURCE
label_config_dict = _simulation.label_config_dict

__all__ = [
    "JOB_TARGET",
    "LABEL_METHOD",
    "LABEL_SOURCE",
    "TARGET_COLUMN",
    "label_config_dict",
]
