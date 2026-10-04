"""Create reproducible visualizations of solved FJSP schedules and graphs.

The plotting helpers reconstruct incumbent timings, assignments and machine
sequences and render headless Gantt charts, disjunctive graphs, candidate graphs
or machine-operation layouts for diagnostics and thesis figures.
"""

import os
import tempfile
from itertools import pairwise
from pathlib import Path


DEFAULT_OUTPUT_DIRECTORY = Path("plots/fjsp_solution_plots")
SUPPORTED_GRAPH_STYLES = {
    "disjunctive",
    "disjunctive_solution",
    "machine_operation",
}


def _value(item):
    """Convert a numeric object, Gurobi variable or expression to ``float``.

    Plot construction assumes an incumbent and therefore reads active values.
    """
    if hasattr(item, "X"):
        return float(item.X)
    if hasattr(item, "getValue"):
        return float(item.getValue())
    return float(item)


def _prepare_plotting(filename):
    """Initialize headless Matplotlib output and the destination path.

    Cache directories are placed below the system temporary directory to avoid
    polluting the repository and the noninteractive ``Agg`` backend is forced.

    Returns:
        Destination path, ``matplotlib.pyplot`` and the ``Line2D`` class.
    """
    cache_root = Path(tempfile.gettempdir()) / "fjsp_plot_cache"
    matplotlib_cache = cache_root / "matplotlib"
    xdg_cache = cache_root / "xdg"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    xdg_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache))

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    return filename, plt, Line2D


def _operation_metadata(instance):
    """Build text labels, mathematical labels and job membership by operation.

    Returns:
        Three dictionaries keyed by operation identifier.
    """
    labels, math_labels, operation_jobs = {}, {}, {}
    for job, operations in sorted(instance.jobs.items()):
        for local_index, operation in enumerate(operations, start=1):
            labels[operation] = f"J{job}O{local_index}"
            math_labels[operation] = rf"$O_{{{job}{local_index}}}$"
            operation_jobs[operation] = job
    return labels, math_labels, operation_jobs


def _selected_machines(variables, instance):
    """Extract the incumbent machine assignment for every real operation.

    The eligible machine with the largest assignment value is selected.
    """
    return {
        operation: max(
            instance.eligible_machines[operation],
            key=lambda machine: _value(variables["Y"][operation, machine]),
        )
        for operation in instance.real_operations
    }


def _schedule(variables, instance):
    """Reconstruct plot-ready operation rows from an incumbent solution.

    Returns:
        Schedule-row dictionaries and the selected-machine mapping.
    """
    _labels, math_labels, operation_jobs = _operation_metadata(instance)
    selected = _selected_machines(variables, instance)
    starts = variables.get("S")
    rows = []
    for operation in instance.real_operations:
        machine = selected[operation]
        duration = float(instance.processing_times[operation, machine])
        completion = _value(variables["C"][operation])
        start = (
            _value(starts[operation])
            if starts is not None and operation in starts
            else completion - duration
        )
        rows.append({
            "operation": operation,
            "label": math_labels[operation],
            "job": operation_jobs[operation],
            "machine": machine,
            "start": max(0.0, start),
            "completion": completion,
            "duration": duration,
        })
    return rows, selected


def _machine_sequences(schedule_rows, machines):
    """Sort assigned operations into chronological sequences per machine.

    Completion time and operation ID provide deterministic tie breaking.
    """
    sequences = {machine: [] for machine in machines}
    for row in schedule_rows:
        sequences.setdefault(row["machine"], []).append(row["operation"])
    row_by_operation = {row["operation"]: row for row in schedule_rows}
    for operations in sequences.values():
        operations.sort(key=lambda operation: (
            row_by_operation[operation]["start"],
            row_by_operation[operation]["completion"],
            operation,
        ))
    return sequences


def _makespan(variables, schedule_rows):
    """Return the explicit makespan or derive it from operation completions.

    The fallback supports formulations that do not expose ``C_max`` directly.
    """
    if variables.get("C_max") is not None:
        return _value(variables["C_max"])
    return max((row["completion"] for row in schedule_rows), default=0.0)


def _palette(machines):
    """Assign deterministic reusable colors to machine identifiers.

    Colors repeat only when the machine count exceeds the fixed palette.
    """
    colors = [
        "#d7191c", "#2c7bb6", "#f2c500", "#1a9641", "#984ea3",
        "#ff7f00", "#66c2a5", "#a6761d", "#e7298a", "#7570b3",
    ]
    return {
        machine: colors[index % len(colors)]
        for index, machine in enumerate(machines)
    }


def plot_solution_schedule(variables, instance, filename, title=None):
    """Write a machine-based Gantt chart for the incumbent solution.

    Jobs determine bar colors, machine IDs define rows and the makespan is
    highlighted by a vertical reference line.

    Returns:
        Path of the written PNG image.
    """
    filename, plt, Line2D = _prepare_plotting(filename)
    from matplotlib.patches import Patch

    schedule_rows, _selected = _schedule(variables, instance)
    machines = sorted(variables.get("machines", range(instance.num_machines)))
    jobs = sorted(instance.jobs)
    makespan = _makespan(variables, schedule_rows)
    job_palette = [
        "#4E79A7", "#F28E2B", "#59A14F", "#E15759", "#B07AA1",
        "#76B7B2", "#EDC948", "#FF9DA7", "#9C755F", "#BAB0AC",
    ]
    job_colors = {
        job: job_palette[index % len(job_palette)]
        for index, job in enumerate(jobs)
    }

    figure_width = max(10.0, min(18.0, 8.0 + 0.12 * makespan))
    figure_height = max(4.2, min(12.0, 1.0 + 0.8 * len(machines)))
    figure, axis = plt.subplots(figsize=(figure_width, figure_height))
    for row in sorted(
        schedule_rows,
        key=lambda item: (item["machine"], item["start"], item["operation"]),
    ):
        axis.barh(
            row["machine"],
            row["duration"],
            left=row["start"],
            height=0.62,
            color=job_colors[row["job"]],
            edgecolor="black",
            linewidth=1.2,
            zorder=3,
        )
        axis.text(
            row["start"] + row["duration"] / 2.0,
            row["machine"],
            row["label"],
            ha="center",
            va="center",
            fontsize=10,
            clip_on=True,
            zorder=4,
        )

    axis.axvline(
        makespan,
        color="red",
        linestyle=(0, (7, 5)),
        linewidth=2.0,
        zorder=2,
    )
    axis.annotate(
        rf"$C_{{\max}}={makespan:g}$",
        xy=(makespan, machines[-1] if machines else 0),
        xytext=(-8, 16),
        textcoords="offset points",
        ha="right",
        va="bottom",
        color="red",
        fontsize=12,
    )
    axis.set_xlabel("Zeit [ZE]")
    axis.set_ylabel("Ressourcen")
    axis.set_yticks(machines)
    axis.set_yticklabels([f"Maschine {machine + 1}" for machine in machines])
    axis.set_xlim(0.0, max(makespan * 1.04, 1.0))
    if machines:
        axis.set_ylim(min(machines) - 0.6, max(machines) + 0.6)
    axis.grid(axis="x", color="#d0d0d0", linewidth=0.8, alpha=0.7, zorder=0)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.set_title(title or "Produktionsprogrammplanung")

    legend_handles = [
        Patch(
            facecolor=job_colors[job],
            edgecolor="black",
            label=f"Job {job}",
        )
        for job in jobs
    ]
    legend_handles.append(Line2D(
        [], [], linestyle="none", marker="",
        label=r"$O_{ij}$ = Operation $j$ von Job $i$",
    ))
    axis.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=min(6, len(legend_handles)),
        frameon=False,
    )
    figure.tight_layout()
    figure.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return filename


def _job_layout(instance):
    """Position operations by job row and technological sequence column.

    Returns:
        Jobs, maximum chain length, node positions and artificial start/end
        positions for disjunctive graph rendering.
    """
    jobs = sorted(instance.jobs)
    max_job_length = max((len(instance.jobs[job]) for job in jobs), default=1)
    positions = {}
    for row_index, job in enumerate(jobs):
        y_position = len(jobs) - row_index
        for column_index, operation in enumerate(instance.jobs[job], start=1):
            positions[operation] = (column_index, y_position)
    middle_y = (len(jobs) + 1) / 2.0
    return jobs, max_job_length, positions, (0, middle_y), (
        max_job_length + 1, middle_y
    )


def _draw_arrow(
    axis,
    source_position,
    target_position,
    *,
    color,
    width,
    alpha=1.0,
    radius=0.0,
    linestyle="-",
    zorder=2,
):
    """Draw one styled directed edge between two graph positions.

    Curvature, opacity, line style and draw order are forwarded to Matplotlib.
    """
    axis.annotate(
        "",
        xy=target_position,
        xytext=source_position,
        arrowprops={
            "arrowstyle": "-|>",
            "color": color,
            "lw": width,
            "alpha": alpha,
            "linestyle": linestyle,
            "shrinkA": 23,
            "shrinkB": 23,
            "mutation_scale": 14,
            "connectionstyle": f"arc3,rad={radius}",
        },
        zorder=zorder,
    )


def _draw_node(axis, position, label, *, edgecolor="black", size=1700):
    """Draw one circular graph node and its centered mathematical label.

    The edge color can encode the selected machine while the fill stays white.
    """
    axis.scatter(
        [position[0]], [position[1]], s=size, marker="o",
        facecolor="white", edgecolor=edgecolor, linewidth=1.8, zorder=4,
    )
    axis.text(
        position[0], position[1], label,
        ha="center", va="center", fontsize=11, zorder=5,
    )


def _draw_job_precedence(axis, instance, positions, start_position, end_position):
    """Draw fixed technological chains including artificial start and end.

    Every job is rendered as a solid directed path through its operations.
    """
    for operations in instance.jobs.values():
        if not operations:
            continue
        _draw_arrow(
            axis, start_position, positions[operations[0]],
            color="black", width=1.7, zorder=3,
        )
        for source, target in pairwise(operations):
            _draw_arrow(
                axis, positions[source], positions[target],
                color="black", width=1.7, zorder=3,
            )
        _draw_arrow(
            axis, positions[operations[-1]], end_position,
            color="black", width=1.7, zorder=3,
        )


def _edge_radius(source_position, target_position):
    """Choose a deterministic arc radius that separates crossing edges.

    Same-row arcs bend upward; cross-row direction determines the curvature sign.
    """
    if abs(source_position[1] - target_position[1]) < 1e-9:
        return 0.22
    return 0.28 if source_position[1] <= target_position[1] else -0.28


def _plot_disjunctive_solution_graph(
    model, variables, instance, filename, title, plt, Line2D
):
    """Render selected job and machine precedence in a job-oriented layout.

    Returns:
        Path of the written solution-graph image.
    """
    schedule_rows, selected = _schedule(variables, instance)
    machines = sorted(variables.get("machines", range(instance.num_machines)))
    sequences = _machine_sequences(schedule_rows, machines)
    _labels, math_labels, _operation_jobs = _operation_metadata(instance)
    jobs, max_length, positions, start_position, end_position = _job_layout(
        instance
    )
    machine_colors = _palette(machines)
    figure, axis = plt.subplots(figsize=(
        max(8.5, min(18.0, 1.75 * (max_length + 2))),
        max(4.8, min(14.0, 1.2 * len(jobs) + 2.2)),
    ))

    _draw_job_precedence(
        axis, instance, positions, start_position, end_position
    )
    for machine, operations in sequences.items():
        for source, target in pairwise(operations):
            _draw_arrow(
                axis,
                positions[source],
                positions[target],
                color=machine_colors[machine],
                width=1.8,
                alpha=0.95,
                radius=_edge_radius(positions[source], positions[target]),
            )

    _draw_node(axis, start_position, "Start", size=1800)
    _draw_node(axis, end_position, "End", size=1800)
    for operation, position in positions.items():
        _draw_node(
            axis,
            position,
            math_labels[operation],
            edgecolor=machine_colors[selected[operation]],
        )

    makespan = _makespan(variables, schedule_rows)
    axis.set_title(
        f"{title or 'Optimaler disjunktiver Lösungsgraph'}\n"
        f"Zielfunktionswert {_value(model.ObjVal):.2f} | "
        f"$C_{{\\max}}$ {makespan:.2f}",
        fontsize=12,
    )
    axis.set_xlim(-0.55, max_length + 1.55)
    axis.set_ylim(0.35, len(jobs) + 0.65)
    axis.axis("off")
    legend_handles = [
        Line2D([0], [0], color="black", lw=1.7, label="Job-Reihenfolge")
    ]
    legend_handles.extend(
        Line2D(
            [0], [0], color=machine_colors[machine], lw=1.8,
            label=f"M{machine + 1}",
        )
        for machine in machines
    )
    axis.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
    )
    figure.tight_layout()
    figure.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return filename


def plot_candidate_graph(variables, instance, filename, title=None):
    """Plot all pairwise machine-order candidates after machine assignment.

    Dashed colored arcs show every still-relevant machine conflict, while solid
    black arcs retain the fixed technological job order.

    Returns:
        Path of the written candidate-graph image.
    """
    filename, plt, Line2D = _prepare_plotting(filename)
    schedule_rows, selected = _schedule(variables, instance)
    machines = sorted(variables.get("machines", range(instance.num_machines)))
    sequences = _machine_sequences(schedule_rows, machines)
    order_positions = {
        (operation, machine): index
        for machine, operations in sequences.items()
        for index, operation in enumerate(operations)
    }
    _labels, math_labels, _operation_jobs = _operation_metadata(instance)
    jobs, max_length, positions, start_position, end_position = _job_layout(
        instance
    )
    machine_colors = _palette(machines)
    figure, axis = plt.subplots(figsize=(
        max(8.5, min(18.0, 1.75 * (max_length + 2))),
        max(4.8, min(14.0, 1.2 * len(jobs) + 2.2)),
    ))

    edge_count = 0
    operations = list(instance.real_operations)
    for left_index, operation_i in enumerate(operations):
        for operation_j in operations[left_index + 1:]:
            machine = selected[operation_i]
            if selected[operation_j] != machine:
                continue
            source, target = operation_i, operation_j
            if order_positions[source, machine] > order_positions[target, machine]:
                source, target = target, source
            _draw_arrow(
                axis,
                positions[source],
                positions[target],
                color=machine_colors[machine],
                width=1.25,
                alpha=0.5,
                radius=_edge_radius(positions[source], positions[target]),
                linestyle="--",
                zorder=1,
            )
            edge_count += 1

    _draw_job_precedence(
        axis, instance, positions, start_position, end_position
    )
    _draw_node(axis, start_position, "Start", size=1800)
    _draw_node(axis, end_position, "End", size=1800)
    for operation, position in positions.items():
        _draw_node(axis, position, math_labels[operation])

    axis.set_title(
        f"{title or 'Kandidatengraph nach Maschinenzuordnung'}\n"
        f"{edge_count} mögliche Maschinenreihenfolge-Kanten",
        fontsize=12,
    )
    axis.set_xlim(-0.55, max_length + 1.55)
    axis.set_ylim(0.35, len(jobs) + 0.65)
    axis.axis("off")
    legend_handles = [
        Line2D([0], [0], color="black", lw=1.7, label="Job-Reihenfolge")
    ]
    legend_handles.extend(
        Line2D(
            [0], [0], color=machine_colors[machine], lw=1.25, ls="--",
            label=f"M{machine + 1}",
        )
        for machine in machines
    )
    axis.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
    )
    figure.tight_layout()
    figure.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return filename


def _plot_machine_operation_graph(
    model, variables, instance, filename, title, plt, Line2D
):
    """Render operations along their selected machine sequences.

    Machine rows emphasize resource order; dashed cross-row edges retain job
    precedence. The figure title reports objective and makespan.
    """
    schedule_rows, _selected = _schedule(variables, instance)
    machines = sorted(variables.get("machines", range(instance.num_machines)))
    sequences = _machine_sequences(schedule_rows, machines)
    labels, _math_labels, operation_jobs = _operation_metadata(instance)
    jobs = sorted(instance.jobs)
    machine_colors = _palette(machines)
    job_colormap = plt.get_cmap("tab20", max(len(jobs), 1))
    job_colors = {
        job: job_colormap(index) for index, job in enumerate(jobs)
    }
    positions, machine_positions = {}, {}
    for row_index, machine in enumerate(machines):
        y_position = len(machines) - row_index
        machine_positions[machine] = (0, y_position)
        for column_index, operation in enumerate(sequences[machine], start=1):
            positions[operation] = (column_index, y_position)
    max_length = max((len(value) for value in sequences.values()), default=1)
    figure, axis = plt.subplots(figsize=(
        max(8.0, min(20.0, 1.8 * max_length + 4.0)),
        max(4.5, min(18.0, 1.1 * len(machines) + 2.0)),
    ))

    for machine, operations in sequences.items():
        for source, target in pairwise(operations):
            _draw_arrow(
                axis, positions[source], positions[target],
                color=machine_colors[machine], width=2.0,
            )
    for operations in instance.jobs.values():
        for source, target in pairwise(operations):
            _draw_arrow(
                axis, positions[source], positions[target],
                color="0.45", width=1.1, alpha=0.65,
                radius=0.2, linestyle="--",
            )
    for machine, position in machine_positions.items():
        axis.scatter(
            [position[0]], [position[1]], s=1450, marker="s",
            color=machine_colors[machine], edgecolor="black",
            linewidth=1.2, zorder=4,
        )
        axis.text(
            position[0], position[1], f"M{machine + 1}",
            ha="center", va="center", fontsize=9, fontweight="bold", zorder=5,
        )
    for operation, position in positions.items():
        axis.scatter(
            [position[0]], [position[1]], s=1250, marker="o",
            color=job_colors[operation_jobs[operation]], edgecolor="black",
            linewidth=1.1, zorder=4,
        )
        axis.text(
            position[0], position[1], labels[operation],
            ha="center", va="center", fontsize=8, fontweight="bold", zorder=5,
        )

    makespan = _makespan(variables, schedule_rows)
    axis.set_title(
        f"{title or 'Maschinen-Operationsgraph'}\n"
        f"Zielfunktionswert {_value(model.ObjVal):.2f} | "
        f"$C_{{\\max}}$ {makespan:.2f}",
        fontsize=12,
    )
    axis.set_xlabel("Position in der Maschinenreihenfolge")
    axis.set_ylabel("Maschine")
    axis.set_yticks([machine_positions[machine][1] for machine in machines])
    axis.set_yticklabels([f"Maschine {machine + 1}" for machine in machines])
    axis.set_xlim(-0.6, max_length + 0.6)
    axis.set_ylim(0.5, len(machines) + 0.5)
    axis.grid(axis="x", alpha=0.2)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(
        handles=[
            Line2D([0], [0], color="0.45", lw=1.1, ls="--", label="Job-Reihenfolge"),
            *[
                Line2D(
                    [0], [0], color=machine_colors[machine], lw=2.0,
                    label=f"Reihenfolge M{machine + 1}",
                )
                for machine in machines
            ],
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=min(4, len(machines) + 1),
        frameon=False,
    )
    figure.tight_layout()
    figure.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return filename


def plot_solution_graph(
    model,
    variables,
    instance,
    filename,
    title=None,
    style="disjunctive",
):
    """Write the incumbent graph using one supported layout.

    Args:
        style: ``disjunctive``, ``disjunctive_solution`` or
            ``machine_operation``.

    Returns:
        Path of the written PNG image.
    """
    if int(model.SolCount) <= 0:
        raise ValueError("Cannot plot a solution graph without a solution.")
    normalized_style = str(style).strip().lower()
    if normalized_style not in SUPPORTED_GRAPH_STYLES:
        raise ValueError(
            f"Unknown plot_solution_graph_style {style!r}; expected one of "
            f"{sorted(SUPPORTED_GRAPH_STYLES)}."
        )
    filename, plt, Line2D = _prepare_plotting(filename)
    if normalized_style in {"disjunctive", "disjunctive_solution"}:
        return _plot_disjunctive_solution_graph(
            model, variables, instance, filename, title, plt, Line2D
        )
    return _plot_machine_operation_graph(
        model, variables, instance, filename, title, plt, Line2D
    )


def write_solution_plots(
    model,
    variables,
    instance,
    *,
    instance_name,
    solver,
    output_directory=DEFAULT_OUTPUT_DIRECTORY,
    plot_solution_schedule_enabled=False,
    plot_solution_graph_enabled=False,
    plot_candidate_graph_enabled=False,
    graph_style="disjunctive",
):
    """Create all enabled solution plots for one solved model.

    GNN filenames include architecture metadata to keep variants separate. No
    files are created when the model has no incumbent.

    Returns:
        Mapping from plot type to generated image path.
    """
    if int(model.SolCount) <= 0:
        return {}
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    solver_slug = str(solver)
    if solver_slug == "gurobi_gnn" and variables.get("gnn_metadata"):
        metadata = variables["gnn_metadata"]
        solver_slug += (
            f"_layers{int(metadata['num_graphsage_layers'])}"
            f"_hidden{int(metadata['hidden_channels'])}"
            f"_seed{int(metadata['seed'])}"
        )
    slug = f"{instance_name}_{solver_slug}"
    paths = {}
    if plot_solution_schedule_enabled:
        paths["schedule"] = plot_solution_schedule(
            variables,
            instance,
            output_directory / f"schedule_{slug}.png",
        )
    if plot_solution_graph_enabled:
        paths["solution_graph"] = plot_solution_graph(
            model,
            variables,
            instance,
            output_directory / f"graph_{slug}.png",
            style=graph_style,
        )
    if plot_candidate_graph_enabled or plot_solution_graph_enabled:
        paths["candidate_graph"] = plot_candidate_graph(
            variables,
            instance,
            output_directory / f"graph_{slug}_candidates.png",
        )
    return paths
