"""Validate and name the supported repair-buffer GNN architectures.

The module centralizes architecture identifiers, valid layer counts, artifact
filenames and model-directory layouts so training and solver discovery apply
the same naming contract.
"""

from pathlib import Path


GRAPH_FIXED = "fixed_candidate"
CONV_LINEAR = "linear"
CONV_SAGE = "sage"
CONV_JOB = "job"
MESSAGE_PASSING_CONVOLUTIONS = {CONV_SAGE, CONV_JOB}
POOL_ADD = "global_add"
VALID_LAYER_COUNTS = {1, 2, 3}


def normalize_convolution(value):
    """Normalize and validate a configured convolution identifier.

    Returns:
        One of ``linear``, ``sage`` or ``job`` in lowercase.

    Raises:
        ValueError: If the identifier is unsupported.
    """
    convolution = str(value).strip().lower()
    if convolution not in {CONV_LINEAR, CONV_SAGE, CONV_JOB}:
        raise ValueError(
            "convolution must be 'linear', 'sage' or 'job'."
        )
    return convolution


def validate_architecture(convolution):
    """Return the fixed architecture fields for one convolution type.

    The graph mode and pooling are shared by all variants, while aggregation is
    disabled for the linear baseline and set to summation for message passing.
    """
    convolution = normalize_convolution(convolution)
    return {
        "graph_mode": GRAPH_FIXED,
        "convolution": convolution,
        "aggregation": "none" if convolution == CONV_LINEAR else "sum",
        "pooling": POOL_ADD,
    }


def _positive_integer(value, name, allowed=None):
    """Validate a strictly positive integer and an optional allowed set.

    Returns:
        The unchanged validated integer.

    Raises:
        ValueError: If booleans, nonintegers or unsupported values are passed.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    if allowed is not None and value not in allowed:
        choices = ", ".join(map(str, sorted(allowed)))
        raise ValueError(f"{name} must be one of {choices}.")
    return value


def architecture_from_config(raw):
    """Normalize one architecture combination from the project configuration.

    Returns:
        Complete architecture dictionary containing fixed graph choices, layer
        count and hidden width.
    """
    raw = dict(raw or {})
    return {
        **validate_architecture(raw["convolution"]),
        "layers": _positive_integer(
            raw["layers"], "layers", allowed=VALID_LAYER_COUNTS
        ),
        "hidden_channels": _positive_integer(
            raw["hidden_channels"], "hidden_channels"
        ),
    }


def architecture_stem(
    convolution,
    target,
    seed,
    layers,
    hidden_channels,
):
    """Build the deterministic filename stem of one trained model.

    The stem encodes graph mode, convolution, aggregation, pooling, depth,
    width, prediction target and training seed.
    """
    architecture = validate_architecture(convolution)
    layer = _positive_integer(
        layers, "layers", allowed=VALID_LAYER_COUNTS
    )
    width = _positive_integer(hidden_channels, "hidden_channels")
    return (
        f"fjsp_gnn_{architecture['graph_mode']}_"
        f"{architecture['convolution']}_{architecture['aggregation']}_"
        f"{architecture['pooling']}_layers{layer}_hidden{width}_"
        f"{target}_seed{int(seed)}"
    )


def architecture_slug(
    convolution,
    layers,
    hidden_channels,
):
    """Build a compact architecture identifier without target or seed.

    Returns:
        Slug containing convolution, aggregation, pooling, depth and width.
    """
    architecture = validate_architecture(convolution)
    layer = _positive_integer(
        layers, "layers", allowed=VALID_LAYER_COUNTS
    )
    width = _positive_integer(hidden_channels, "hidden_channels")
    return (
        f"{architecture['convolution']}_{architecture['aggregation']}_"
        f"{architecture['pooling']}_layers{layer}_hidden{width}"
    )


def architecture_model_dir(
    root,
    convolution,
    layers,
    hidden_channels,
):
    """Return the canonical artifact directory for one architecture.

    Args:
        root: Root directory containing all trained models.
        convolution: Active convolution identifier.
        layers: Number of hidden layers.
        hidden_channels: Width of every hidden layer.

    Returns:
        Architecture-specific :class:`pathlib.Path` below ``root``.
    """
    architecture = validate_architecture(convolution)
    layer = _positive_integer(
        layers, "layers", allowed=VALID_LAYER_COUNTS
    )
    width = _positive_integer(hidden_channels, "hidden_channels")
    family = (
        f"{architecture['convolution']}_{architecture['aggregation']}_"
        f"{architecture['pooling']}"
    )
    return Path(root) / family / f"hidden{width}_layers{layer}"
