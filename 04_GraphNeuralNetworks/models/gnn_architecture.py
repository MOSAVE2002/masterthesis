"""Canonical GNN architecture names and compatibility validation."""

from itertools import product

GRAPH_FIXED = "fixed_candidate"
CONV_LINEAR = "linear"
CONV_GCN = "gcn"
CONV_GINE = "gine"
CONV_MPNN = "mpnn"
CONV_SAGE = "sage"
MESSAGE_PASSING_CONVOLUTIONS = {
    CONV_GCN,
    CONV_GINE,
    CONV_MPNN,
    CONV_SAGE,
}
SUM_ONLY_CONVOLUTIONS = {CONV_GINE, CONV_MPNN}

POOL_ADD = "global_add"
VALID_LAYER_COUNTS = {1, 2, 3}


def normalize_convolution(value):
    normalized = str(value).strip()
    if normalized not in {
        CONV_LINEAR,
        CONV_GCN,
        CONV_GINE,
        CONV_MPNN,
        CONV_SAGE,
    }:
        raise ValueError(
            "convolution must be linear, gcn, gine, mpnn, or sage."
        )
    return normalized


def normalize_pooling(value):
    normalized = str(value).strip()
    if normalized != POOL_ADD:
        raise ValueError("pooling must be global_add.")
    return normalized


def normalize_aggregation(value, convolution):
    convolution = normalize_convolution(convolution)
    if convolution == CONV_LINEAR:
        if value not in (None, "", "none"):
            raise ValueError(
                "Linear requires aggregation='none'."
            )
        return "none"
    normalized = str(value or "sum").strip().lower()
    if normalized not in {"mean", "sum"}:
        raise ValueError("aggregation must be mean or sum.")
    if convolution in SUM_ONLY_CONVOLUTIONS and normalized != "sum":
        raise ValueError(f"{convolution.upper()} requires aggregation='sum'.")
    return normalized


def validate_architecture(graph_mode, convolution, aggregation, pooling):
    graph_mode = str(graph_mode).strip().lower()
    if graph_mode != GRAPH_FIXED:
        raise ValueError("graph_mode must be fixed_candidate.")
    convolution = normalize_convolution(convolution)
    aggregation = normalize_aggregation(aggregation, convolution)
    pooling = normalize_pooling(pooling)
    return {
        "graph_mode": graph_mode,
        "convolution": convolution,
        "aggregation": aggregation,
        "pooling": pooling,
    }


def _positive_integer_options(value, name, allowed=None):
    values = value if isinstance(value, (list, tuple)) else [value]
    if not values:
        raise ValueError(f"{name} must not be empty.")
    result = []
    for item in values:
        if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
            raise ValueError(
                f"{name} must be a positive integer or a non-empty list "
                "of positive integers."
            )
        if allowed is not None and item not in allowed:
            allowed_text = ", ".join(str(option) for option in sorted(allowed))
            raise ValueError(f"{name} must be one of {allowed_text}.")
        if item not in result:
            result.append(item)
    return result


def expand_architecture_variants(raw, defaults=None):
    """Expand one config entry into explicit layer/hidden-size variants."""
    raw = dict(raw or {})
    defaults = dict(defaults or {})
    architecture = validate_architecture(
        raw.get(
            "graph_mode",
            defaults.get("graph_mode", GRAPH_FIXED),
        ),
        raw.get(
            "convolution",
            defaults.get("convolution", CONV_SAGE),
        ),
        raw.get("aggregation", defaults.get("aggregation")),
        raw.get("pooling", defaults.get("pooling", POOL_ADD)),
    )
    layers = raw.get("layers", defaults.get("layers"))
    hidden_channels = raw.get(
        "hidden_channels", defaults.get("hidden_channels")
    )
    if layers is None or hidden_channels is None:
        raise ValueError(
            "Every GNN combination must define layers and "
            "hidden_channels."
        )
    layer_options = _positive_integer_options(
        layers,
        "layers",
        allowed=VALID_LAYER_COUNTS,
    )
    hidden_options = _positive_integer_options(
        hidden_channels,
        "hidden_channels",
    )
    return [
        {
            **architecture,
            "layers": layers,
            "hidden_channels": hidden_channels,
        }
        for layers, hidden_channels in product(
            layer_options, hidden_options
        )
    ]


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
    arch = validate_architecture(graph_mode, convolution, aggregation, pooling)
    stem = (
        f"fjsp_gnn_{arch['graph_mode']}_{arch['convolution']}_"
        f"{arch['aggregation']}_{arch['pooling']}"
    )
    if (layers is None) != (hidden_channels is None):
        raise ValueError(
            "layers and hidden_channels must either both be set or both "
            "be omitted."
        )
    if layers is not None:
        layer_value = _positive_integer_options(
            layers, "layers", allowed=VALID_LAYER_COUNTS
        )
        hidden_value = _positive_integer_options(
            hidden_channels, "hidden_channels"
        )
        if len(layer_value) != 1 or len(hidden_value) != 1:
            raise ValueError(
                "A model name requires one explicit layers and "
                "hidden_channels value."
            )
        stem += (
            f"_layers{layer_value[0]}_hidden{hidden_value[0]}"
        )
    return f"{stem}_{target}_seed{int(seed)}"


def architecture_slug(
    graph_mode,
    convolution,
    aggregation,
    pooling,
    layers=None,
    hidden_channels=None,
):
    arch = validate_architecture(graph_mode, convolution, aggregation, pooling)
    slug = (
        f"{arch['convolution']}_{arch['aggregation']}_{arch['pooling']}"
    )
    if (layers is None) != (hidden_channels is None):
        raise ValueError(
            "layers and hidden_channels must either both be set or both "
            "be omitted."
        )
    if layers is not None:
        layer_value = _positive_integer_options(
            layers, "layers", allowed=VALID_LAYER_COUNTS
        )
        hidden_value = _positive_integer_options(
            hidden_channels, "hidden_channels"
        )
        if len(layer_value) != 1 or len(hidden_value) != 1:
            raise ValueError(
                "A model directory requires one explicit layers and "
                "hidden_channels value."
            )
        slug += (
            f"_layers{layer_value[0]}_hidden{hidden_value[0]}"
        )
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
    arch = validate_architecture(graph_mode, convolution, aggregation, pooling)
    if layers is None and hidden_channels is None:
        return root / architecture_slug(**arch)
    if layers is None or hidden_channels is None:
        raise ValueError(
            "layers and hidden_channels must either both be set or both "
            "be omitted."
        )
    layer_value = _positive_integer_options(
        layers, "layers", allowed=VALID_LAYER_COUNTS
    )
    hidden_value = _positive_integer_options(
        hidden_channels, "hidden_channels"
    )
    if len(layer_value) != 1 or len(hidden_value) != 1:
        raise ValueError(
            "A trained-model directory requires one layers and one "
            "hidden_channels value."
        )
    model_directory = (
        f"{arch['convolution']}_{arch['aggregation']}_{arch['pooling']}"
    )
    size_directory = (
        f"hidden{hidden_value[0]}_layers{layer_value[0]}"
    )
    return root / model_directory / size_directory
