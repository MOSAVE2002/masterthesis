"""Architecture naming for the active reliability-surrogate models."""

from itertools import product
from pathlib import Path


GRAPH_FIXED = "fixed_candidate"
CONV_LINEAR = "linear"
CONV_SAGE = "sage"
CONV_JOB = "job"
MESSAGE_PASSING_CONVOLUTIONS = {CONV_SAGE, CONV_JOB}
POOL_ADD = "global_add"
VALID_LAYER_COUNTS = {1, 2, 3}


def normalize_convolution(value):
    convolution = str(value).strip().lower()
    if convolution not in {CONV_LINEAR, CONV_SAGE, CONV_JOB}:
        raise ValueError(
            "convolution must be 'linear', 'sage' or 'job'."
        )
    return convolution


def normalize_pooling(value):
    pooling = str(value).strip().lower()
    if pooling != POOL_ADD:
        raise ValueError("pooling must be 'global_add'.")
    return pooling


def normalize_aggregation(value, convolution):
    convolution = normalize_convolution(convolution)
    if convolution == CONV_LINEAR:
        if value not in (None, "", "none"):
            raise ValueError("linear requires aggregation='none'.")
        return "none"
    aggregation = str(value or "sum").strip().lower()
    if aggregation != "sum":
        raise ValueError(
            f"{convolution} requires aggregation='sum'."
        )
    return aggregation


def validate_architecture(graph_mode, convolution, aggregation, pooling):
    graph_mode = str(graph_mode).strip().lower()
    if graph_mode != GRAPH_FIXED:
        raise ValueError("graph_mode must be 'fixed_candidate'.")
    convolution = normalize_convolution(convolution)
    return {
        "graph_mode": graph_mode,
        "convolution": convolution,
        "aggregation": normalize_aggregation(aggregation, convolution),
        "pooling": normalize_pooling(pooling),
    }


def _positive_integer_options(value, name, allowed=None):
    values = value if isinstance(value, (list, tuple)) else [value]
    result = []
    for item in values:
        if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
            raise ValueError(f"{name} must contain positive integers.")
        if allowed is not None and item not in allowed:
            choices = ", ".join(map(str, sorted(allowed)))
            raise ValueError(f"{name} must be one of {choices}.")
        if item not in result:
            result.append(item)
    if not result:
        raise ValueError(f"{name} must not be empty.")
    return result


def expand_architecture_variants(raw, defaults=None):
    raw = dict(raw or {})
    defaults = dict(defaults or {})
    architecture = validate_architecture(
        raw.get("graph_mode", defaults.get("graph_mode", GRAPH_FIXED)),
        raw.get("convolution", defaults.get("convolution", CONV_SAGE)),
        raw.get("aggregation", defaults.get("aggregation")),
        raw.get("pooling", defaults.get("pooling", POOL_ADD)),
    )
    layers = raw.get("layers", defaults.get("layers"))
    hidden = raw.get("hidden_channels", defaults.get("hidden_channels"))
    if layers is None or hidden is None:
        raise ValueError("layers and hidden_channels are required.")
    return [
        {**architecture, "layers": layer, "hidden_channels": width}
        for layer, width in product(
            _positive_integer_options(
                layers, "layers", allowed=VALID_LAYER_COUNTS
            ),
            _positive_integer_options(hidden, "hidden_channels"),
        )
    ]


def _explicit_size(layers, hidden_channels):
    if (layers is None) != (hidden_channels is None):
        raise ValueError(
            "layers and hidden_channels must either both be set or omitted."
        )
    if layers is None:
        return None
    layer = _positive_integer_options(
        layers, "layers", allowed=VALID_LAYER_COUNTS
    )
    hidden = _positive_integer_options(hidden_channels, "hidden_channels")
    if len(layer) != 1 or len(hidden) != 1:
        raise ValueError("A path requires one explicit model size.")
    return layer[0], hidden[0]


def architecture_stem(
    graph_mode,
    convolution,
    aggregation,
    pooling,
    target,
    seed,
    layers=None,
    hidden_channels=None,
):
    architecture = validate_architecture(
        graph_mode, convolution, aggregation, pooling
    )
    stem = (
        f"fjsp_gnn_{architecture['graph_mode']}_"
        f"{architecture['convolution']}_{architecture['aggregation']}_"
        f"{architecture['pooling']}"
    )
    size = _explicit_size(layers, hidden_channels)
    if size:
        stem += f"_layers{size[0]}_hidden{size[1]}"
    return f"{stem}_{target}_seed{int(seed)}"


def architecture_slug(
    graph_mode,
    convolution,
    aggregation,
    pooling,
    layers=None,
    hidden_channels=None,
):
    architecture = validate_architecture(
        graph_mode, convolution, aggregation, pooling
    )
    slug = (
        f"{architecture['convolution']}_{architecture['aggregation']}_"
        f"{architecture['pooling']}"
    )
    size = _explicit_size(layers, hidden_channels)
    if size:
        slug += f"_layers{size[0]}_hidden{size[1]}"
    return slug


def architecture_model_dir(
    root,
    graph_mode,
    convolution,
    aggregation,
    pooling,
    layers=None,
    hidden_channels=None,
):
    architecture = validate_architecture(
        graph_mode, convolution, aggregation, pooling
    )
    size = _explicit_size(layers, hidden_channels)
    root = Path(root)
    if size is None:
        return root / architecture_slug(**architecture)
    family = (
        f"{architecture['convolution']}_{architecture['aggregation']}_"
        f"{architecture['pooling']}"
    )
    return root / family / f"hidden{size[1]}_layers{size[0]}"
