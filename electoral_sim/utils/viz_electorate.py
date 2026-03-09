"""
viz_electorate.py
-----------------
Spatial visualizations of electorates, candidate positions, and election outcomes
in the 2D preference space [0,1]^2.

Primary functions
-----------------
plot_electorate(electorate, candidates, ...)
    Voter density (KDE contours) + candidate positions + mean/median markers.

plot_election_result(electorate, candidates, result, ...)
    Same as above but highlights the winning outcome and system name.

plot_all_systems_spatial(electorate, candidates, results, ...)
    Grid of subplots, one per electoral system, all on the same spatial axes.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgba
from scipy.stats import gaussian_kde

from electoral_sim.electorate import Electorate
from electoral_sim.candidates import CandidateSet
from electoral_sim.types import ElectionResult


# ── Shared style constants ────────────────────────────────────────────────────

CANDIDATE_CMAP = plt.cm.tab10
WINNER_MARKER = "*"
WINNER_SIZE = 380
CANDIDATE_SIZE = 160
MEAN_COLOR = "#e63946"       # red
MEDIAN_COLOR = "#2a9d8f"     # teal
OUTCOME_COLOR = "#f4a261"    # orange

FIGURE_DPI = 150
SAVE_DPI = 300


# ── Internal helpers ──────────────────────────────────────────────────────────

def _kde_contours(
    ax: plt.Axes,
    preferences: np.ndarray,
    n_grid: int = 120,
    levels: int = 8,
    cmap: str = "Blues",
    alpha: float = 0.55,
) -> None:
    """
    Draw KDE density as an imshow heatmap with contour lines overlaid.
    Uses imshow rather than contourf for matplotlib 3.9+ compatibility
    (contourf changed how QuadContourSet registers with axes in 3.9+).
    """
    x, y = preferences[:, 0], preferences[:, 1]
    kde = gaussian_kde(np.vstack([x, y]), bw_method="scott")
    xi = np.linspace(0, 1, n_grid)
    yi = np.linspace(0, 1, n_grid)
    Xi, Yi = np.meshgrid(xi, yi)
    Zi = kde(np.vstack([Xi.ravel(), Yi.ravel()])).reshape(Xi.shape)
    # imshow: origin="lower" aligns (0,0) at bottom-left
    ax.imshow(
        Zi,
        origin="lower",
        extent=[0, 1, 0, 1],
        aspect="auto",
        cmap=cmap,
        alpha=alpha,
        interpolation="bilinear",
    )
    ax.contour(Xi, Yi, Zi, levels=levels, colors="white", linewidths=0.4, alpha=0.5)


def _plot_candidates(
    ax: plt.Axes,
    candidates: CandidateSet,
    highlight_indices: list[int] | None = None,
    seat_shares: dict[int, float] | None = None,
) -> list:
    """
    Plot candidate positions as colored markers.
    Winners are drawn larger with a star marker; others as circles.
    Returns list of legend handles.
    """
    handles = []
    highlight_indices = highlight_indices or []
    seat_shares = seat_shares or {}

    for i, (pos, label) in enumerate(zip(candidates.positions, candidates.labels)):
        color = CANDIDATE_CMAP(i / max(candidates.n_candidates - 1, 1))
        is_winner = i in highlight_indices

        marker = WINNER_MARKER if is_winner else "o"
        size = WINNER_SIZE if is_winner else CANDIDATE_SIZE
        zorder = 6 if is_winner else 5
        edgecolor = "black" if is_winner else "white"
        linewidth = 1.5 if is_winner else 0.8

        ax.scatter(pos[0], pos[1], s=size, c=[color], marker=marker,
                   zorder=zorder, edgecolors=edgecolor, linewidths=linewidth)

        # Label offset — push text slightly up and right
        ax.annotate(
            label,
            xy=(pos[0], pos[1]),
            xytext=(5, 6),
            textcoords="offset points",
            fontsize=7.5,
            color="black",
            fontweight="bold" if is_winner else "normal",
            zorder=7,
        )

        share_str = f" ({seat_shares[i]:.0%})" if i in seat_shares else ""
        patch = mpatches.Patch(color=color, label=f"{label}{share_str}")
        handles.append(patch)

    return handles


def _plot_reference_points(
    ax: plt.Axes,
    electorate: Electorate,
    show_mean: bool = True,
    show_median: bool = True,
) -> list:
    """Draw mean and geometric median markers. Returns legend handles."""
    handles = []
    if show_mean:
        m = electorate.mean()
        ax.scatter(m[0], m[1], s=130, c=MEAN_COLOR, marker="D",
                   zorder=8, edgecolors="white", linewidths=1.2)
        handles.append(mpatches.Patch(color=MEAN_COLOR, label="Mean preference"))

    if show_median:
        gm = electorate.geometric_median()
        ax.scatter(gm[0], gm[1], s=130, c=MEDIAN_COLOR, marker="P",
                   zorder=8, edgecolors="white", linewidths=1.2)
        handles.append(mpatches.Patch(color=MEDIAN_COLOR, label="Geometric median"))

    return handles


def _plot_pr_outcomes(
    ax: plt.Axes,
    result: ElectionResult,
) -> list:
    """Draw both PR outcome points: centroid (✕) and median legislator (★ outlined)."""
    handles = []

    # Centroid — orange X
    c = result.centroid_position
    ax.scatter(c[0], c[1], s=200, color=OUTCOME_COLOR, marker="X",
               zorder=9, edgecolors="black", linewidths=1.0)
    handles.append(mpatches.Patch(color=OUTCOME_COLOR, label="PR centroid (reference)"))

    # Median legislator — purple diamond
    ml = result.median_legislator_position
    ax.scatter(ml[0], ml[1], s=220, color="#9b5de5", marker="D",
               zorder=10, edgecolors="black", linewidths=1.2)
    handles.append(mpatches.Patch(color="#9b5de5", label="Median legislator (outcome)"))

    # Connect them with a dashed line to show the gap
    ax.plot([c[0], ml[0]], [c[1], ml[1]],
            color="gray", linewidth=1.0, linestyle="--", zorder=8, alpha=0.7)

    return handles


def _style_spatial_ax(
    ax: plt.Axes,
    title: str,
    dim_names: list[str],
) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel(dim_names[0], fontsize=9)
    ax.set_ylabel(dim_names[1], fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
    ax.tick_params(labelsize=8)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2, linewidth=0.5)


# ── Public API ────────────────────────────────────────────────────────────────

def plot_electorate(
    electorate: Electorate,
    candidates: CandidateSet,
    title: str = "Electorate & Candidates",
    show_mean: bool = True,
    show_median: bool = True,
    figsize: tuple[float, float] = (6, 5.5),
    ax: plt.Axes | None = None,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Plot voter density (KDE contours), candidate positions, and
    mean/median reference points in the 2D preference space.

    Parameters
    ----------
    electorate : Electorate
    candidates : CandidateSet
    title : str
    show_mean : bool
    show_median : bool
    figsize : tuple
    ax : optional pre-existing Axes (for embedding in larger figures)
    save_path : str, optional. If given, saves PNG + PDF.

    Returns
    -------
    matplotlib Figure
    """
    assert electorate.n_dims == 2, "Spatial plots require 2D electorates"

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=figsize, dpi=FIGURE_DPI)
    else:
        fig = ax.get_figure()

    _kde_contours(ax, electorate.preferences)
    cand_handles = _plot_candidates(ax, candidates)
    ref_handles = _plot_reference_points(ax, electorate, show_mean, show_median)
    _style_spatial_ax(ax, title, electorate.dim_names)

    all_handles = cand_handles + ref_handles
    ax.legend(handles=all_handles, loc="upper left", fontsize=7.5,
               framealpha=0.85, borderpad=0.6)

    if standalone:
        fig.tight_layout()
        if save_path:
            _save_figure(fig, save_path)

    return fig


def plot_election_result(
    electorate: Electorate,
    candidates: CandidateSet,
    result: ElectionResult,
    show_mean: bool = True,
    show_median: bool = True,
    show_outcome_centroid: bool = True,
    figsize: tuple[float, float] = (6, 5.5),
    ax: plt.Axes | None = None,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Plot an election result spatially: voter KDE, all candidates
    (winners highlighted with ★), mean/median markers, and for PR
    systems the weighted-centroid outcome.

    Parameters
    ----------
    show_outcome_centroid : bool
        If True, draws the weighted-centroid outcome point (useful for
        PR systems where outcome != any single candidate position).
    """
    assert electorate.n_dims == 2, "Spatial plots require 2D electorates"

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=figsize, dpi=FIGURE_DPI)
    else:
        fig = ax.get_figure()

    _kde_contours(ax, electorate.preferences)

    cand_handles = _plot_candidates(
        ax, candidates,
        highlight_indices=result.winner_indices,
        seat_shares=result.seat_shares if result.is_pr else {},
    )

    ref_handles = _plot_reference_points(ax, electorate, show_mean, show_median)
    outcome_handles = []

    if result.is_pr and show_outcome_centroid:
        outcome_handles = _plot_pr_outcomes(ax, result)

    _style_spatial_ax(ax, result.system_name, electorate.dim_names)

    all_handles = cand_handles + ref_handles + outcome_handles
    ax.legend(handles=all_handles, loc="upper left", fontsize=7.0,
               framealpha=0.85, borderpad=0.6)

    # Annotate distances in corner
    geo_median = electorate.geometric_median()
    d_median = np.linalg.norm(result.outcome_position - geo_median)
    ax.text(0.98, 0.03,
            f"d(outcome, median) = {d_median:.3f}",
            transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    if standalone:
        fig.tight_layout()
        if save_path:
            _save_figure(fig, save_path)

    return fig


def plot_all_systems_spatial(
    electorate: Electorate,
    candidates: CandidateSet,
    results: list[ElectionResult],
    n_cols: int = 3,
    show_mean: bool = True,
    show_median: bool = True,
    suptitle: str | None = None,
    figsize_per_panel: tuple[float, float] = (4.2, 4.0),
    save_path: str | None = None,
) -> plt.Figure:
    """
    Grid of spatial subplots — one panel per electoral system.
    All panels share the same voter KDE and candidate positions;
    each highlights its own winner(s).

    Parameters
    ----------
    results : list[ElectionResult]
        One result per system. Ordering determines panel layout.
    n_cols : int
        Number of columns in the subplot grid.
    """
    assert electorate.n_dims == 2, "Spatial plots require 2D electorates"

    n = len(results)
    n_cols = min(n_cols, n)
    n_rows = int(np.ceil(n / n_cols))

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_per_panel[0] * n_cols, figsize_per_panel[1] * n_rows),
        dpi=FIGURE_DPI,
    )
    axes_flat = np.array(axes).flatten()

    for i, (result, ax) in enumerate(zip(results, axes_flat)):
        _kde_contours(ax, electorate.preferences, levels=6, alpha=0.45)
        _plot_candidates(
            ax, candidates,
            highlight_indices=result.winner_indices,
            seat_shares=result.seat_shares if result.is_pr else {},
        )
        _plot_reference_points(ax, electorate, show_mean, show_median)

        if result.is_pr:
            _plot_pr_outcomes(ax, result)

        geo_median = electorate.geometric_median()
        d_median = np.linalg.norm(result.outcome_position - geo_median)
        if result.is_pr:
            d_centroid = np.linalg.norm(result.centroid_position - geo_median)
            dist_label = f"d(med.leg)={d_median:.3f}\nd(centroid)={d_centroid:.3f}"
        else:
            dist_label = f"d={d_median:.3f}"

        _style_spatial_ax(ax, result.system_name, electorate.dim_names)
        ax.text(0.98, 0.03,
                dist_label,
                transform=ax.transAxes,
                ha="right", va="bottom", fontsize=7.5,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85))

    # Hide unused panels
    for ax in axes_flat[n:]:
        ax.set_visible(False)

    # Shared legend
    mean_patch    = mpatches.Patch(color=MEAN_COLOR,    label="Mean preference")
    median_patch  = mpatches.Patch(color=MEDIAN_COLOR,  label="Geometric median")
    centroid_patch = mpatches.Patch(color=OUTCOME_COLOR, label="PR centroid (reference)")
    medleg_patch  = mpatches.Patch(color="#9b5de5",     label="Median legislator (PR outcome)")
    fig.legend(
        handles=[mean_patch, median_patch, centroid_patch, medleg_patch],
        loc="lower center",
        ncol=4,
        fontsize=9,
        framealpha=0.9,
        bbox_to_anchor=(0.5, 0.0),
    )

    sup = suptitle or "Electoral Systems — Spatial Comparison"
    fig.suptitle(sup, fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()

    if save_path:
        _save_figure(fig, save_path)

    return fig


# ── Save helper ───────────────────────────────────────────────────────────────

def _save_figure(fig: plt.Figure, base_path: str) -> None:
    """Save figure as PNG, PDF, and SVG."""
    from pathlib import Path
    base = Path(base_path)
    base.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(base.with_suffix(f".{ext}"), dpi=SAVE_DPI, bbox_inches="tight")
