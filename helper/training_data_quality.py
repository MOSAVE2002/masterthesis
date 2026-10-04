"""Maintain durable progress and quality metadata for GNN dataset generation.

Generation summaries are updated after individual instances and written by an
atomic temporary-file replacement so interrupted long-running jobs retain a
consistent description of completed, skipped and jittered samples.
"""

import json
from datetime import datetime, timezone


def utc_timestamp():
    """Return the current timezone-aware UTC timestamp in ISO 8601 format.

    The explicit timezone keeps summaries comparable across execution hosts.
    """
    return datetime.now(timezone.utc).isoformat()


def write_generation_summary(path, summary):
    """Update aggregate counters and atomically persist a generation summary.

    Args:
        path: Destination JSON path.
        summary: Mutable summary containing per-split progress structures.

    Side Effects:
        Adds timestamps and totals to ``summary`` and replaces the destination
        through a temporary file in the same directory.
    """
    summary["updated_at_utc"] = utc_timestamp()
    split_values = list(summary["splits"].values())
    successful = sum(
        len(values["successful_instances"]) for values in split_values
    )
    skipped = sum(
        len(values["skipped_instances"]) for values in split_values
    )
    summary["totals"] = {
        "configured_instances": sum(
            values["configured_count"] for values in split_values
        ),
        "processed_instances": successful + skipped,
        "successful_instances": successful,
        "jittered_successful_instances": sum(
            bool(item.get("training_parameter_jitter_applied", False))
            for values in split_values
            for item in values["successful_instances"]
        ),
        "skipped_instances": skipped,
        "written_graphs": sum(
            values["written_graphs"] for values in split_values
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary_path.replace(path)
