"""
viz_primaries.py
----------------
Visualizations specific to primary election analysis.

plot_primary_spatial(electorate, candidates, parties, result, ...)
    Spatial plot showing party membership regions, primary nominees,
    general election winner, and reference points.

plot_primary_comparison(results_dict, metric, ...)
    Bar chart comparing general election outcomes across different
    primary configurations (primary type × primary system).

plot_primary_vs_baseline(two_party_results, ...)
    Side-by-side comparison of outcome with and without primaries,
    across multiple scenarios.

plot_nominee_positions(scenarios_results, ...)
    Scatter showing where nominees land relative to voter clusters
    and geometric median, across scenarios and configurations.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.colors import to_rgba
from scipy.stats import gaussian_kde

from electoral_sim.electorate import Electorate
from electoral_sim.candidates import CandidateSet
from electoral_sim.primaries import (
    PartySpec, PrimaryResult, TwoPartyGeneralResult, PrimaryType
)
from electoral_sim.utils.viz_electorate import (
    _kde_contours, _plot_reference_points, _style_spatial_ax,
    MEAN_COLOR, MEDIAN_COLOR, FIGURE_DPI, SAVE_DPI,
)

# Party colors — consistent across all plots
LEFT_COLOR  = "#4361ee"   # blue
RIGHT_COLOR = "#e63946"   # red
NOMINEE_EDGE = "black"
WINNER_COLOR = "#2dc653"  # green

PARTY_COLORS = {
    "Left Party":  LEFT_COLOR,
    "Right Party": RIGHT_COLOR,
}


def _party_color(party_name: str) -> str:
    return PARTY_COLORS.get(party_name, "#888888")


def _save_figure(fig, base_path):
    from pathlib import Path
    base = Path(base_path)
    base.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(base.with_suffix(f".{ext}"), dpi=SAVE_DPI, bbox_inches="tight")


# ── 1. Spatial overview of one primary scenario ───────────────────────────────

def plot_primary_spatial(
    electorate: Electorate,
    candidates: CandidateSet,
    parties: list[PartySpec],
    result: TwoPartyGeneralResult,
    title: str | None = None,
    show_membership_shading: bool = True,
    figsize: tuple[float, float] = (7, 6.5),
    ax: plt.Axes | None = None,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Spatial plot of a two-party primary scenario.

    Shows:
    - Voter KDE contours (full electorate)
    - Party membership shading (left = blue tint, right = red tint)
    - All candidates (colored by party, faded if eliminated in primary)
    - Nominees (large colored stars)
    - General election winner (green outlined star)
    - Geometric median and mean reference points
    - Distance annotations
    """
    assert electorate.n_dims == 2

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=figsize, dpi=FIGURE_DPI)
    else:
        fig = ax.get_figure()

    # ── Party membership shading ──────────────────────────────────────────────
    if show_membership_shading:
        from electoral_sim.primaries import assign_party_membership
        memberships = assign_party_membership(
            electorate, candidates, parties,
            result.primary_results[0].primary_type,
        )
        for party in parties:
            mask  = memberships[party.name]
            prefs = electorate.preferences[mask]
            if len(prefs) > 10:
                color = _party_color(party.name)
                kde   = gaussian_kde(prefs.T, bw_method="scott")
                xi    = np.linspace(0, 1, 80)
                yi    = np.linspace(0, 1, 80)
                Xi, Yi = np.meshgrid(xi, yi)
                Zi = kde(np.vstack([Xi.ravel(), Yi.ravel()])).reshape(Xi.shape)
                ax.contourf(Xi, Yi, Zi, levels=4,
                            colors=[color], alpha=0.12)

    # ── Full-electorate KDE outline ───────────────────────────────────────────
    _kde_contours(ax, electorate.preferences, levels=5, cmap="Greys", alpha=0.35)

    # ── All candidates ────────────────────────────────────────────────────────
    nominee_global_indices = {pr.nominee_index for pr in result.primary_results}
    handles = []

    for party in parties:
        color = _party_color(party.name)
        for idx in party.candidate_indices:
            pos      = candidates.positions[idx]
            is_nom   = idx in nominee_global_indices
            marker   = "★" if is_nom else "o"
            size     = 280 if is_nom else 90
            alpha    = 1.0 if is_nom else 0.45
            edgecol  = "black" if is_nom else color
            lw       = 1.8 if is_nom else 0.6
            zorder   = 7 if is_nom else 4

            ax.scatter(pos[0], pos[1], s=size, c=[color], marker="*" if is_nom else "o",
                       alpha=alpha, edgecolors=edgecol, linewidths=lw, zorder=zorder)
            ax.annotate(candidates.labels[idx], xy=(pos[0], pos[1]),
                        xytext=(5, 5), textcoords="offset points",
                        fontsize=7, color=color,
                        fontweight="bold" if is_nom else "normal",
                        zorder=8)

    # ── Legend entries for parties ────────────────────────────────────────────
    for party in parties:
        c = _party_color(party.name)
        pr = next(p for p in result.primary_results if p.party_name == party.name)
        nom_label = candidates.labels[pr.nominee_index]
        handles.append(mpatches.Patch(
            color=c,
            label=f"{party.name} (nominee: {nom_label}, "
                  f"{pr.primary_system_name}, "
                  f"n={pr.n_primary_voters:,})"
        ))

    # ── General winner marker ─────────────────────────────────────────────────
    winner_pos = result.general_result.outcome_position
    ax.scatter(winner_pos[0], winner_pos[1], s=380, c=WINNER_COLOR,
               marker="*", zorder=10, edgecolors="black", linewidths=2.0)
    handles.append(mpatches.Patch(color=WINNER_COLOR,
                                   label=f"General winner  d(median)={result.general_metrics.distance_to_median:.4f}"))

    # ── Reference points ─────────────────────────────────────────────────────
    ref_handles = _plot_reference_points(ax, electorate, show_mean=True, show_median=True)
    handles += ref_handles

    # ── Baseline comparison annotation ───────────────────────────────────────
    d_primary  = result.general_metrics.distance_to_median
    d_baseline = result.baseline_metrics.distance_to_median
    delta      = d_primary - d_baseline
    delta_str  = f"{delta:+.4f}"
    color_str  = "#c0392b" if delta > 0 else "#27ae60"
    ax.text(0.98, 0.03,
            f"Primary: d={d_primary:.4f}\nBaseline: d={d_baseline:.4f}\nΔ={delta_str}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
            color=color_str,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.88))

    # ── Divergence annotation ─────────────────────────────────────────────────
    ax.text(0.02, 0.03,
            f"Nominee divergence: {result.primary_divergence:.4f}",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.88))

    _style_spatial_ax(ax, title or "Two-Party Primary", electorate.dim_names)
    ax.legend(handles=handles, loc="upper left", fontsize=7.2,
               framealpha=0.88, borderpad=0.6)

    if standalone:
        fig.tight_layout()
        if save_path:
            _save_figure(fig, save_path)

    return fig


# ── 2. Primary type × primary system comparison bar ──────────────────────────

def plot_primary_comparison(
    results: dict[str, TwoPartyGeneralResult],
    metric: str = "distance_to_median",
    title: str | None = None,
    include_baseline: bool = True,
    figsize: tuple[float, float] = (9, 4.5),
    ax: plt.Axes | None = None,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Bar chart comparing general election outcome across different primary
    configurations for a single scenario.

    Parameters
    ----------
    results : dict[str, TwoPartyGeneralResult]
        Keys are configuration labels (e.g. "Closed / Plurality",
        "Open / IRV"). Values are TwoPartyGeneralResult instances.
    metric : str
        Which metric to plot from general_metrics.
    include_baseline : bool
        If True, adds a "No Primary (baseline)" bar at the end.
    """
    metric_labels = {
        "distance_to_median": "Distance to Geometric Median",
        "distance_to_mean":   "Distance to Mean Preference",
        "majority_satisfaction": "Majority Satisfaction",
        "worst_case_distance": "Worst-Case Voter Distance",
    }
    display = metric_labels.get(metric, metric)
    direction = "high" if metric == "majority_satisfaction" else "low"

    labels = list(results.keys())
    values = [getattr(r.general_metrics, metric) for r in results.values()]

    if include_baseline:
        first_result = next(iter(results.values()))
        labels.append("No Primary (baseline)")
        values.append(getattr(first_result.baseline_metrics, metric))

    # Color bars: green if best, else steel blue, baseline gray
    best_val = min(values) if direction == "low" else max(values)
    colors = []
    for i, (lab, val) in enumerate(zip(labels, values)):
        if lab == "No Primary (baseline)":
            colors.append("#aaaaaa")
        elif val == best_val:
            colors.append("#2dc653")
        else:
            colors.append("#4895ef")

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=figsize, dpi=FIGURE_DPI)
    else:
        fig = ax.get_figure()

    bars = ax.barh(labels, values, color=colors, edgecolor="white",
                   linewidth=0.6, height=0.6)

    for bar, val in zip(bars, values):
        ax.text(val + max(values) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", ha="left", fontsize=9)

    ax.set_xlabel(display, fontsize=10)
    ax.set_title(title or f"Primary Configuration Comparison — {display}",
                 fontsize=11, fontweight="bold")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)

    better = "← lower is better" if direction == "low" else "→ higher is better"
    ax.text(0.99, -0.10, better, transform=ax.transAxes,
            ha="right", fontsize=8, color="gray", style="italic")

    if standalone:
        fig.tight_layout()
        if save_path:
            _save_figure(fig, save_path)

    return fig


# ── 3. Primary vs baseline across scenarios ───────────────────────────────────

def plot_primary_vs_baseline(
    scenario_results: dict[str, dict[str, TwoPartyGeneralResult]],
    metric: str = "distance_to_median",
    title: str | None = None,
    figsize: tuple[float, float] = (11, 5),
    save_path: str | None = None,
) -> plt.Figure:
    """
    Grouped bar chart: for each scenario, show outcome with each primary
    configuration alongside the no-primary baseline.

    Parameters
    ----------
    scenario_results : dict[str, dict[str, TwoPartyGeneralResult]]
        Outer key: scenario name.
        Inner key: configuration label.
    """
    metric_labels = {
        "distance_to_median": "Distance to Geometric Median",
        "distance_to_mean":   "Distance to Mean Preference",
        "majority_satisfaction": "Majority Satisfaction",
        "worst_case_distance": "Worst-Case Voter Distance",
    }
    display   = metric_labels.get(metric, metric)
    direction = "high" if metric == "majority_satisfaction" else "low"

    scenario_names = list(scenario_results.keys())
    # Collect all config keys (same across scenarios)
    config_keys = list(next(iter(scenario_results.values())).keys())
    all_keys    = config_keys + ["Baseline"]

    n_scenarios = len(scenario_names)
    n_configs   = len(all_keys)
    x           = np.arange(n_scenarios)
    width       = 0.75 / n_configs

    fig, ax = plt.subplots(figsize=figsize, dpi=FIGURE_DPI)
    palette = plt.cm.Set2(np.linspace(0, 1, n_configs))

    for j, (key, color) in enumerate(zip(all_keys, palette)):
        vals = []
        for scen in scenario_names:
            res_dict = scenario_results[scen]
            if key == "Baseline":
                first = next(iter(res_dict.values()))
                vals.append(getattr(first.baseline_metrics, metric))
            else:
                vals.append(getattr(res_dict[key].general_metrics, metric))
        offset = (j - n_configs / 2 + 0.5) * width
        ax.bar(x + offset, vals, width=width * 0.9, label=key,
               color=color, edgecolor="white", linewidth=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels(scenario_names, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel(display, fontsize=10)
    ax.set_title(title or f"Primary vs Baseline — {display}",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right", framealpha=0.85)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)

    better = "↓ lower is better" if direction == "low" else "↑ higher is better"
    ax.text(0.01, 0.97, better, transform=ax.transAxes,
            va="top", fontsize=8.5, color="gray", style="italic")

    fig.tight_layout()
    if save_path:
        _save_figure(fig, save_path)

    return fig


# ── 4. Nominee position scatter ───────────────────────────────────────────────

def plot_nominee_positions(
    scenario_results: dict[str, dict[str, TwoPartyGeneralResult]],
    electorates: dict[str, Electorate],
    candidates_map: dict[str, CandidateSet],
    figsize_per_panel: tuple[float, float] = (5.5, 5.0),
    n_cols: int = 2,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Grid of spatial plots — one panel per scenario — showing where nominees
    land under each primary configuration, relative to voter density and the
    geometric median.

    Parameters
    ----------
    scenario_results : dict[str, dict[str, TwoPartyGeneralResult]]
    electorates : dict[str, Electorate]
    candidates_map : dict[str, CandidateSet]
    """
    scenario_names = list(scenario_results.keys())
    n = len(scenario_names)
    n_cols = min(n_cols, n)
    n_rows = int(np.ceil(n / n_cols))

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_per_panel[0] * n_cols, figsize_per_panel[1] * n_rows),
        dpi=FIGURE_DPI,
    )
    axes_flat = np.array(axes).flatten()

    config_keys = list(next(iter(scenario_results.values())).keys())
    config_colors = plt.cm.tab10(np.linspace(0, 0.6, len(config_keys)))

    for ax, scen_name in zip(axes_flat, scenario_names):
        e = electorates[scen_name]
        c = candidates_map[scen_name]
        res_dict = scenario_results[scen_name]

        _kde_contours(ax, e.preferences, levels=5, cmap="Greys", alpha=0.35)
        _plot_reference_points(ax, e, show_mean=True, show_median=True)

        handles = []
        for (config_key, color) in zip(config_keys, config_colors):
            result = res_dict[config_key]
            for pr in result.primary_results:
                pos   = pr.nominee_position
                pcol  = _party_color(pr.party_name)
                # Outline color = config color
                ax.scatter(pos[0], pos[1], s=160, c=[pcol],
                           marker="*", zorder=6,
                           edgecolors=color, linewidths=2.0)
            handles.append(mpatches.Patch(color=color, label=config_key))

        # Baseline nominee (no-primary winner)
        first_result = next(iter(res_dict.values()))
        base_pos = first_result.baseline_metrics  # used for annotation only
        baseline_winner = first_result.general_result  # NOTE: baseline re-computed below

        _style_spatial_ax(ax, scen_name, e.dim_names)
        ax.legend(handles=handles, loc="upper left", fontsize=7, framealpha=0.85)

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    # Shared legend for party colors
    left_patch  = mpatches.Patch(color=LEFT_COLOR,  label="Left Party nominee")
    right_patch = mpatches.Patch(color=RIGHT_COLOR, label="Right Party nominee")
    mean_patch  = mpatches.Patch(color=MEAN_COLOR,  label="Mean preference")
    med_patch   = mpatches.Patch(color=MEDIAN_COLOR, label="Geometric median")
    fig.legend(handles=[left_patch, right_patch, mean_patch, med_patch],
               loc="lower center", ncol=4, fontsize=9,
               framealpha=0.9, bbox_to_anchor=(0.5, 0.0))

    fig.suptitle("Nominee Positions by Scenario and Primary Configuration",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()

    if save_path:
        _save_figure(fig, save_path)

    return fig
