"""
viz_metrics.py
--------------
Comparative visualizations for electoral system performance metrics.

Primary functions
-----------------
plot_metric_bar(metrics_list, metric, ...)
    Horizontal bar chart: all systems ranked by a single metric.

plot_grouped_metrics(metrics_list, ...)
    Grouped bar chart: all systems × all metrics side by side.

plot_scenario_heatmap(scenario_summaries, metric, ...)
    Heatmap: systems (rows) × scenarios (columns), color = metric value.

plot_radar(metrics_list, ...)
    Radar / spider chart: each system as a polygon across all metrics.

plot_monte_carlo_distributions(mc_results, metric, ...)
    Box plots showing metric distribution across Monte Carlo trials.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec

from electoral_sim.metrics import ElectionMetrics

FIGURE_DPI = 150
SAVE_DPI = 300

# Metric display config: (display_name, better_direction)
# better_direction: "low" means lower is better, "high" means higher is better
METRIC_CONFIG: dict[str, tuple[str, str]] = {
    "distance_to_median":   ("Distance to Geometric Median", "low"),
    "distance_to_mean":     ("Distance to Mean Preference",  "low"),
    "majority_satisfaction":("Majority Satisfaction",         "high"),
    "worst_case_distance":  ("Worst-Case Voter Distance",    "low"),
    "mean_voter_distance":  ("Mean Voter Distance",           "low"),
    "gini_distance":        ("Gini (Distance Inequality)",   "low"),
}

ALL_METRICS = list(METRIC_CONFIG.keys())

# Color palette for systems — consistent across all plots
_SYSTEM_COLORS: dict[str, str] = {}
_PALETTE = plt.cm.tab10.colors


def _system_color(system_name: str, all_names: list[str]) -> str:
    for i, n in enumerate(sorted(set(all_names))):
        _SYSTEM_COLORS.setdefault(n, _PALETTE[i % len(_PALETTE)])
    return _SYSTEM_COLORS[system_name]


def _short_name(name: str) -> str:
    """Abbreviate long system names for axis labels."""
    replacements = {
        "Plurality (FPTP)": "Plurality",
        "Instant Runoff (IRV)": "IRV",
        "Condorcet (Schulze)": "Condorcet",
        "Mixed Member Proportional (MMP)": "MMP",
    }
    # Handle dynamic PR name like "Party-List PR (D'Hondt, 100 seats)"
    if name.startswith("Party-List PR"):
        return "PR (D'Hondt)"
    return replacements.get(name, name)


def _save_figure(fig: plt.Figure, base_path: str) -> None:
    from pathlib import Path
    base = Path(base_path)
    base.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(base.with_suffix(f".{ext}"), dpi=SAVE_DPI, bbox_inches="tight")


# ── 1. Single-metric bar chart ────────────────────────────────────────────────

def plot_metric_bar(
    metrics_list: list[ElectionMetrics],
    metric: str = "distance_to_median",
    title: str | None = None,
    figsize: tuple[float, float] = (7, 4.5),
    ax: plt.Axes | None = None,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Horizontal bar chart ranking all systems by a single metric.
    Bars are colored by system; the best-performing bar is annotated.

    Parameters
    ----------
    metrics_list : list[ElectionMetrics]
        One ElectionMetrics per system (from a single scenario run).
    metric : str
        Key from METRIC_CONFIG.
    """
    assert metric in METRIC_CONFIG, f"Unknown metric: {metric}. Choose from {ALL_METRICS}"
    display_name, direction = METRIC_CONFIG[metric]

    names = [m.system_name for m in metrics_list]
    values = np.array([getattr(m, metric) for m in metrics_list])

    # Sort: best first
    order = values.argsort() if direction == "low" else values.argsort()[::-1]
    names_sorted = [_short_name(names[i]) for i in order]
    values_sorted = values[order]
    colors = [_system_color(names[i], names) for i in order]

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=figsize, dpi=FIGURE_DPI)
    else:
        fig = ax.get_figure()

    bars = ax.barh(names_sorted, values_sorted, color=colors,
                   edgecolor="white", linewidth=0.6, height=0.65)

    # Annotate values on bars
    for bar, val in zip(bars, values_sorted):
        ax.text(val + max(values_sorted) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", ha="left", fontsize=8.5)

    # Star best bar
    best_bar = bars[0]
    best_bar.set_edgecolor("black")
    best_bar.set_linewidth(1.5)

    ax.set_xlabel(display_name, fontsize=10)
    ax.set_title(title or f"Systems Ranked by {display_name}", fontsize=11, fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    ax.tick_params(labelsize=9)
    ax.invert_yaxis()  # best at top
    ax.grid(axis="x", alpha=0.3, linewidth=0.6)
    ax.set_axisbelow(True)

    better_label = "← lower is better" if direction == "low" else "→ higher is better"
    ax.text(0.99, -0.08, better_label, transform=ax.transAxes,
            ha="right", va="top", fontsize=8, color="gray", style="italic")

    if standalone:
        fig.tight_layout()
        if save_path:
            _save_figure(fig, save_path)

    return fig


# ── 2. Grouped bar chart (all metrics) ───────────────────────────────────────

def plot_grouped_metrics(
    metrics_list: list[ElectionMetrics],
    metrics_to_show: list[str] | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (13, 5.5),
    normalize: bool = True,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Grouped bar chart: systems on x-axis, one bar group per metric.
    Values are optionally min-max normalized within each metric so all
    metrics are on the same [0,1] scale for visual comparison.

    Parameters
    ----------
    normalize : bool
        If True, normalize each metric to [0,1] within this set of systems.
        If False, plot raw values (metrics will be on different scales).
    """
    metrics_to_show = metrics_to_show or ALL_METRICS
    n_systems = len(metrics_list)
    n_metrics = len(metrics_to_show)

    names = [_short_name(m.system_name) for m in metrics_list]
    all_names = [m.system_name for m in metrics_list]

    # Build value matrix: (n_systems, n_metrics)
    raw = np.array([[getattr(m, met) for met in metrics_to_show] for m in metrics_list])

    if normalize:
        lo = raw.min(axis=0, keepdims=True)
        hi = raw.max(axis=0, keepdims=True)
        denom = np.where(hi - lo < 1e-10, 1.0, hi - lo)
        display_vals = (raw - lo) / denom
        # For "high is better" metrics, invert so bar height = goodness
        for j, met in enumerate(metrics_to_show):
            if METRIC_CONFIG[met][1] == "high":
                display_vals[:, j] = 1.0 - display_vals[:, j]
        ylabel = "Normalized score (lower bar = better)"
    else:
        display_vals = raw
        ylabel = "Raw metric value"

    x = np.arange(n_systems)
    width = 0.75 / n_metrics

    fig, ax = plt.subplots(figsize=figsize, dpi=FIGURE_DPI)

    metric_colors = plt.cm.Set2(np.linspace(0, 1, n_metrics))

    for j, (met, color) in enumerate(zip(metrics_to_show, metric_colors)):
        offset = (j - n_metrics / 2 + 0.5) * width
        ax.bar(x + offset, display_vals[:, j], width=width * 0.92,
               label=METRIC_CONFIG[met][0], color=color, edgecolor="white", linewidth=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=22, ha="right", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title or "Electoral System Performance — All Metrics", 
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right", framealpha=0.85, ncol=2)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", labelsize=8)

    if normalize:
        ax.axhline(0.5, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)

    fig.tight_layout()
    if save_path:
        _save_figure(fig, save_path)

    return fig


# ── 3. Scenario × system heatmap ─────────────────────────────────────────────

def plot_scenario_heatmap(
    scenario_summaries: dict[str, list[ElectionMetrics]],
    metric: str = "distance_to_median",
    title: str | None = None,
    figsize: tuple[float, float] = (10, 5),
    annotate: bool = True,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Heatmap with scenarios on one axis and electoral systems on the other.
    Cell color = metric value. Useful for the cross-scenario comparison notebook.

    Parameters
    ----------
    scenario_summaries : dict[str, list[ElectionMetrics]]
        Keys are scenario names; values are metrics_list from run_simulation().
    metric : str
        Which metric to visualize.
    annotate : bool
        If True, print numeric values in each cell.
    """
    assert metric in METRIC_CONFIG

    display_name, direction = METRIC_CONFIG[metric]
    scenario_names = list(scenario_summaries.keys())
    # Get system names from first scenario
    system_names = [_short_name(m.system_name) for m in next(iter(scenario_summaries.values()))]

    # Build matrix: (n_systems, n_scenarios)
    n_sys = len(system_names)
    n_scen = len(scenario_names)
    matrix = np.zeros((n_sys, n_scen))

    for j, scen in enumerate(scenario_names):
        for i, m in enumerate(scenario_summaries[scen]):
            matrix[i, j] = getattr(m, metric)

    # Color: green = good (low for "low" metrics, high for "high" metrics)
    cmap = "RdYlGn_r" if direction == "low" else "RdYlGn"

    fig, ax = plt.subplots(figsize=figsize, dpi=FIGURE_DPI)
    im = ax.imshow(matrix, cmap=cmap, aspect="auto",
                   vmin=matrix.min(), vmax=matrix.max())

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label(display_name, fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    # Axis labels
    ax.set_xticks(range(n_scen))
    ax.set_xticklabels(scenario_names, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(n_sys))
    ax.set_yticklabels(system_names, fontsize=9)
    ax.set_title(title or f"{display_name} — Systems × Scenarios",
                 fontsize=12, fontweight="bold", pad=10)

    # Cell annotations
    if annotate:
        for i in range(n_sys):
            for j in range(n_scen):
                val = matrix[i, j]
                # Choose text color for contrast
                normed = (val - matrix.min()) / max(matrix.max() - matrix.min(), 1e-10)
                txt_color = "white" if (normed < 0.25 or normed > 0.75) else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=8, color=txt_color, fontweight="bold")

    # Best-per-scenario marker
    best_fn = np.argmin if direction == "low" else np.argmax
    for j in range(n_scen):
        best_i = best_fn(matrix[:, j])
        ax.add_patch(plt.Rectangle(
            (j - 0.5, best_i - 0.5), 1, 1,
            fill=False, edgecolor="black", linewidth=2.0, zorder=5
        ))

    fig.tight_layout()
    if save_path:
        _save_figure(fig, save_path)

    return fig


# ── 4. Radar / spider chart ───────────────────────────────────────────────────

def plot_radar(
    metrics_list: list[ElectionMetrics],
    metrics_to_show: list[str] | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (7, 7),
    save_path: str | None = None,
) -> plt.Figure:
    """
    Radar (spider) chart: each system is a colored polygon.
    All metrics are normalized to [0,1] and oriented so that
    outward = better (so lower-is-better metrics are inverted).

    Parameters
    ----------
    metrics_to_show : list[str], optional
        Subset of metrics to include. Defaults to all 6.
    """
    metrics_to_show = metrics_to_show or ALL_METRICS
    n_metrics = len(metrics_to_show)
    names = [m.system_name for m in metrics_list]

    labels = [METRIC_CONFIG[m][0] for m in metrics_to_show]
    # Shorten long labels for radar
    labels = [l.replace(" (Distance Inequality)", "\n(Inequality)")
               .replace("Geometric Median", "Geo. Median")
               .replace("Worst-Case Voter Distance", "Worst-Case\nDistance")
               for l in labels]

    raw = np.array([[getattr(m, met) for met in metrics_to_show] for m in metrics_list])

    # Normalize to [0,1], orient so outward = better
    lo = raw.min(axis=0)
    hi = raw.max(axis=0)
    denom = np.where(hi - lo < 1e-10, 1.0, hi - lo)
    normed = (raw - lo) / denom

    for j, met in enumerate(metrics_to_show):
        if METRIC_CONFIG[met][1] == "low":
            normed[:, j] = 1.0 - normed[:, j]  # invert: lower raw = further out

    # Angles for each metric axis
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # close polygon

    fig, ax = plt.subplots(figsize=figsize, dpi=FIGURE_DPI,
                           subplot_kw=dict(polar=True))

    palette = plt.cm.tab10.colors
    handles = []

    for i, (sys_name, vals) in enumerate(zip(names, normed)):
        color = palette[i % len(palette)]
        vals_closed = np.append(vals, vals[0])
        ax.plot(angles, vals_closed, color=color, linewidth=1.8, linestyle="solid")
        ax.fill(angles, vals_closed, color=color, alpha=0.12)
        handles.append(mpatches.Patch(color=color, label=_short_name(sys_name)))

    # Axis labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.0"], fontsize=7, color="gray")
    ax.set_ylim(0, 1)

    # Grid styling
    ax.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.spines["polar"].set_visible(False)

    ax.set_title(title or "Electoral System Profiles", fontsize=12,
                 fontweight="bold", pad=20)

    ax.legend(handles=handles, loc="upper right",
              bbox_to_anchor=(1.32, 1.15), fontsize=8.5, framealpha=0.9)

    fig.tight_layout()
    if save_path:
        _save_figure(fig, save_path)

    return fig


# ── 5. Monte Carlo distribution plots ────────────────────────────────────────

def plot_monte_carlo_distributions(
    mc_results: dict[str, list[ElectionMetrics]],
    metric: str = "distance_to_median",
    title: str | None = None,
    figsize: tuple[float, float] = (10, 5),
    save_path: str | None = None,
) -> plt.Figure:
    """
    Box plots showing the distribution of a metric across Monte Carlo trials,
    one box per electoral system. Useful for stability / variance analysis.

    Parameters
    ----------
    mc_results : dict[str, list[ElectionMetrics]]
        Output of run_monte_carlo() — keys are system names.
    metric : str
    """
    assert metric in METRIC_CONFIG
    display_name, direction = METRIC_CONFIG[metric]

    system_names = list(mc_results.keys())
    data = [np.array([getattr(m, metric) for m in mc_results[sn]])
            for sn in system_names]
    short_names = [_short_name(n) for n in system_names]

    # Sort by median
    medians = [np.median(d) for d in data]
    order = np.argsort(medians) if direction == "low" else np.argsort(medians)[::-1]
    data_sorted = [data[i] for i in order]
    names_sorted = [short_names[i] for i in order]
    colors_sorted = [_system_color(system_names[i], system_names) for i in order]

    fig, ax = plt.subplots(figsize=figsize, dpi=FIGURE_DPI)

    bp = ax.boxplot(
        data_sorted,
        vert=True,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=2),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        flierprops=dict(marker="o", markersize=3, alpha=0.4),
        widths=0.55,
    )

    for patch, color in zip(bp["boxes"], colors_sorted):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax.set_xticks(range(1, len(names_sorted) + 1))
    ax.set_xticklabels(names_sorted, rotation=22, ha="right", fontsize=9)
    ax.set_ylabel(display_name, fontsize=10)
    ax.set_title(title or f"Monte Carlo Distribution: {display_name}",
                 fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)

    better_label = "↓ lower is better" if direction == "low" else "↑ higher is better"
    ax.text(0.01, 0.97, better_label, transform=ax.transAxes,
            va="top", fontsize=8.5, color="gray", style="italic")

    # Annotate median values above each box
    for i, (d, name) in enumerate(zip(data_sorted, names_sorted)):
        ax.text(i + 1, np.percentile(d, 75) + (d.max() - d.min()) * 0.03,
                f"{np.median(d):.3f}", ha="center", va="bottom",
                fontsize=7.5, color="black")

    fig.tight_layout()
    if save_path:
        _save_figure(fig, save_path)

    return fig
