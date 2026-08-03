import json
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT_DIR / "config.json"
if CONFIG_PATH.exists():
    with CONFIG_PATH.open(encoding="utf-8") as file:
        CONFIG = json.load(file)
else:
    CONFIG = {}
CONSTRAINT_CONFIG = CONFIG.get("constraint", {})

CONSTRAINT_WEIBULL = "weibull"
VALID_CONSTRAINT_TYPES = {CONSTRAINT_WEIBULL}

TARGET_COLUMNS = {
    CONSTRAINT_WEIBULL: "total_failure_delay",
}


def validate_constraint_type(value):
    value = str(value).strip().lower()
    if value not in VALID_CONSTRAINT_TYPES:
        raise ValueError(
            f"constraint_type must be 'weibull', got {value!r}."
        )
    return value


def target_column(constraint_type):
    return TARGET_COLUMNS[validate_constraint_type(constraint_type)]


def model_stem(constraint_type, seed):
    target = target_column(constraint_type)
    return f"fjsp_gnn_{target}_{int(seed)}"


def configured_constraint_type():
    return validate_constraint_type(
        CONSTRAINT_CONFIG.get("type", CONSTRAINT_WEIBULL)
    )

