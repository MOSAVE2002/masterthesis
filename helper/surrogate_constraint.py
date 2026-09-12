import json
from pathlib import Path

from helper.local_buffer import TARGET_COLUMN as LOCAL_BUFFER_TARGET


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
    CONSTRAINT_WEIBULL: LOCAL_BUFFER_TARGET,
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


def configured_constraint_type():
    return validate_constraint_type(
        CONSTRAINT_CONFIG.get("type", CONSTRAINT_WEIBULL)
    )
