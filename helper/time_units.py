"""Abstract scheduling time units; solver wall-clock times remain seconds."""

import math
from collections.abc import Mapping


TIME_UNIT = "ZE"


def normalize_time_unit(source=TIME_UNIT, *, require_metadata=False):
    """Read ZE or legacy annotations without rescaling any model values.

    Older artifacts stored a minutes-per-ZE annotation, but their numerical
    times and rates were already expressed in ZE and inverse ZE.
    """
    if isinstance(source, str):
        unit = source
    else:
        values = source if isinstance(source, Mapping) else vars(source)
        unit = values.get("time_unit")
        if unit is None:
            legacy = values.get("time_unit_minutes")
            if legacy is not None:
                legacy = float(legacy)
                if not math.isfinite(legacy) or legacy <= 0:
                    raise ValueError("Invalid legacy time-unit annotation.")
            elif require_metadata:
                raise ValueError("Missing time_unit in artifact metadata.")
            unit = TIME_UNIT
    if unit != TIME_UNIT:
        raise ValueError(f"time_unit must be 'ZE', got {unit!r}.")
    return TIME_UNIT
