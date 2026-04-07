"""Latin-hypercube helpers for cross-group welfare notebooks."""
from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import qmc

from electoral_sim.ballots import BallotProfile
from electoral_sim.metrics import compute_group_metrics, compute_metrics
from notebooks.helpers.cross_group_welfare import (
    SLATE_ORDER,
    SYSTEM_ORDER,
    SYSTEM_SPECS,
    build_candidate_slate,
)


MAJORITY_SHARE_RANGE = (0.60, 0.99)
LOW_OVERLAP_MEANS = (
    np.array([0.24, 0.56]),
    np.array([0.76, 0.44]),
)
HIGH_OVERLAP_MEANS = (
    np.array([0.43, 0.52]),
    np.array([0.57, 0.48]),
)
TIGHT_COV = np.array([[0.007, 0.001], [0.001, 0.009]])
BROAD_COV = np.array([[0.018, 0.002], [0.002, 0.020]])
APPROVAL_THRESHOLD = 0.65


def lhs_helper_overview() -> pd.DataFrame:
    """Compact summary of the sampled design used by the notebook."""
    return pd.DataFrame(
        {
            "dimension": [
                "Majority share",
                "Overlap scale",
                "Dispersion scale",
                "Candidate slate",
                "Electoral systems",
            ],
            "values": [
                "[0.60, 0.99]",
                "0 = low overlap, 1 = high overlap",
                "0 = tight groups, 1 = broad groups",
                ", ".join(SLATE_ORDER),
                ", ".join(SYSTEM_ORDER),
            ],
        }
    )


def sample_latin_hypercube_design(
    n_cases: int = 36,
    seed: int = 20260406,
) -> pd.DataFrame:
    """
    Sample a compact design over continuous scenario parameters.

    The first three dimensions are used as continuous values and the fourth is
    binned into the three hand-built candidate slates.
    """
    sampler = qmc.LatinHypercube(d=4, seed=seed)
    raw = sampler.random(n=n_cases)
    majority_share = MAJORITY_SHARE_RANGE[0] + raw[:, 0] * (
        MAJORITY_SHARE_RANGE[1] - MAJORITY_SHARE_RANGE[0]
    )
    overlap_scale = raw[:, 1]
    dispersion_scale = raw[:, 2]
    slate_index = np.minimum((raw[:, 3] * len(SLATE_ORDER)).astype(int), len(SLATE_ORDER) - 1)
    slates = [SLATE_ORDER[idx] for idx in slate_index]

    df = pd.DataFrame(
        {
            "case_id": np.arange(n_cases, dtype=int),
            "majority_share": majority_share,
            "minority_share": 1.0 - majority_share,
            "overlap_scale": overlap_scale,
            "dispersion_scale": dispersion_scale,
            "candidate_slate": slates,
        }
    )
    df["share_label"] = [
        f"{int(round(100 * maj))}:{100 - int(round(100 * maj))}" for maj in df["majority_share"]
    ]
    return df.sort_values(["majority_share", "overlap_scale", "dispersion_scale"]).reset_index(drop=True)


def _means_from_overlap(overlap_scale: float) -> tuple[np.ndarray, np.ndarray]:
    majority_mean = LOW_OVERLAP_MEANS[0] + overlap_scale * (HIGH_OVERLAP_MEANS[0] - LOW_OVERLAP_MEANS[0])
    minority_mean = LOW_OVERLAP_MEANS[1] + overlap_scale * (HIGH_OVERLAP_MEANS[1] - LOW_OVERLAP_MEANS[1])
    return majority_mean, minority_mean


def _cov_from_dispersion(dispersion_scale: float) -> np.ndarray:
    return TIGHT_COV + dispersion_scale * (BROAD_COV - TIGHT_COV)


def _build_sampled_electorate(
    row: pd.Series | dict,
    seed: int,
    n_voters: int = 1800,
    dim_names: list[str] | None = None,
):
    from electoral_sim.electorate import gaussian_mixture_electorate

    majority_mean, minority_mean = _means_from_overlap(float(row["overlap_scale"]))
    cov = _cov_from_dispersion(float(row["dispersion_scale"]))
    rng = np.random.default_rng(seed)
    return gaussian_mixture_electorate(
        n_voters,
        [
            {
                "weight": float(row["majority_share"]),
                "mean": majority_mean,
                "cov": cov,
                "group": "Majority group",
            },
            {
                "weight": float(row["minority_share"]),
                "mean": minority_mean,
                "cov": cov,
                "group": "Minority group",
            },
        ],
        rng=rng,
        dim_names=dim_names or ["economic", "social"],
    )


def _group_scatter(ax, electorate, candidates, title: str, show_legend: bool = False) -> None:
    colors = {"Majority group": "#2a9d8f", "Minority group": "#e76f51"}
    for group_id, group_name in electorate.group_names.items():
        mask = electorate.group_ids == group_id
        ax.scatter(
            electorate.preferences[mask, 0],
            electorate.preferences[mask, 1],
            s=10,
            alpha=0.22,
            color=colors[group_name],
            rasterized=True,
            label=group_name,
        )
    ax.scatter(
        candidates.positions[:, 0],
        candidates.positions[:, 1],
        s=68,
        marker="X",
        color="black",
        linewidths=0.8,
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("economic", fontsize=8)
    ax.set_ylabel("social", fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.18, linewidth=0.5)
    if show_legend:
        ax.legend(loc="upper left", fontsize=7, framealpha=0.9)


def plot_lhs_parameter_coverage(design_df: pd.DataFrame):
    """Show how the Latin-hypercube points cover the continuous design space."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), dpi=150)
    sns.scatterplot(
        data=design_df,
        x="majority_share",
        y="overlap_scale",
        hue="candidate_slate",
        style="candidate_slate",
        s=80,
        ax=axes[0],
    )
    axes[0].set_title("Coverage of majority share and overlap", fontsize=10)
    axes[0].set_xlabel("Majority share")
    axes[0].set_ylabel("Overlap scale")

    sns.scatterplot(
        data=design_df,
        x="majority_share",
        y="dispersion_scale",
        hue="candidate_slate",
        style="candidate_slate",
        s=80,
        legend=False,
        ax=axes[1],
    )
    axes[1].set_title("Coverage of majority share and dispersion", fontsize=10)
    axes[1].set_xlabel("Majority share")
    axes[1].set_ylabel("Dispersion scale")
    fig.tight_layout()
    return fig


def plot_lhs_case_gallery(
    design_df: pd.DataFrame,
    n_show: int = 9,
    base_seed: int = 20260406,
    n_voters: int = 900,
):
    """Preview a small subset of sampled cases without exploding notebook size."""
    chosen_idx = np.linspace(0, len(design_df) - 1, num=min(n_show, len(design_df)), dtype=int)
    selected = design_df.iloc[chosen_idx].reset_index(drop=True)
    n_cols = 3
    n_rows = int(math.ceil(len(selected) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4.6 * n_rows), dpi=150)
    axes = np.atleast_1d(axes).ravel()

    for ax, (_, row) in zip(axes, selected.iterrows()):
        electorate = _build_sampled_electorate(row, seed=base_seed + int(row["case_id"]), n_voters=n_voters)
        candidates = build_candidate_slate(row["candidate_slate"])
        title = (
            f"maj={row['majority_share']:.2f}, overlap={row['overlap_scale']:.2f}\n"
            f"disp={row['dispersion_scale']:.2f}, {row['candidate_slate']}"
        )
        _group_scatter(ax, electorate, candidates, title=title, show_legend=(ax is axes[0]))

    for ax in axes[len(selected):]:
        ax.axis("off")

    fig.tight_layout()
    return fig


def _run_system_case(electorate, candidates, system_spec):
    ballots = BallotProfile.from_preferences(
        electorate,
        candidates,
        approval_threshold=APPROVAL_THRESHOLD,
    )
    system = system_spec.build()
    result = system.run(ballots, candidates)
    metrics = compute_metrics(result, electorate, candidates)
    group_summary = compute_group_metrics(result, electorate, candidates)
    by_group = {group.group_name: group for group in group_summary.groups}
    majority = by_group["Majority group"]
    minority = by_group["Minority group"]
    return {
        "system": system_spec.display_name,
        "aggregate_welfare": -metrics.mean_voter_distance,
        "aggregate_mean_distance": metrics.mean_voter_distance,
        "majority_welfare": majority.welfare,
        "minority_welfare": minority.welfare,
        "majority_mean_distance": majority.mean_voter_distance,
        "minority_mean_distance": minority.mean_voter_distance,
        "majority_satisfaction": majority.majority_satisfaction,
        "minority_satisfaction": minority.majority_satisfaction,
        "welfare_gap": majority.welfare - minority.welfare,
        "distance_to_median": metrics.distance_to_median,
    }


def run_lhs_grid(
    design_df: pd.DataFrame,
    base_seed: int = 20260406,
    n_voters: int = 1800,
) -> pd.DataFrame:
    """Evaluate all systems on each sampled scenario."""
    records = []
    for _, row in design_df.iterrows():
        electorate = _build_sampled_electorate(
            row,
            seed=base_seed + int(row["case_id"]),
            n_voters=n_voters,
        )
        candidates = build_candidate_slate(row["candidate_slate"])
        scenario_rows = []
        for system_spec in SYSTEM_SPECS:
            result_row = _run_system_case(electorate, candidates, system_spec)
            result_row.update(row.to_dict())
            scenario_rows.append(result_row)

        plurality_row = next(item for item in scenario_rows if item["system"] == "Plurality")
        for item in scenario_rows:
            item["aggregate_welfare_delta_vs_plurality"] = (
                item["aggregate_welfare"] - plurality_row["aggregate_welfare"]
            )
            item["majority_welfare_delta_vs_plurality"] = (
                item["majority_welfare"] - plurality_row["majority_welfare"]
            )
            item["minority_welfare_delta_vs_plurality"] = (
                item["minority_welfare"] - plurality_row["minority_welfare"]
            )
            item["welfare_gap_delta_vs_plurality"] = (
                item["welfare_gap"] - plurality_row["welfare_gap"]
            )
            item["aggregate_better_minority_worse"] = bool(
                item["aggregate_welfare_delta_vs_plurality"] > 1e-9
                and item["minority_welfare_delta_vs_plurality"] < -1e-9
            )
            item["improves_both_aggregate_and_minority"] = bool(
                item["aggregate_welfare_delta_vs_plurality"] > 1e-9
                and item["minority_welfare_delta_vs_plurality"] > 1e-9
            )
            item["minority_help_aggregate_cost"] = bool(
                item["aggregate_welfare_delta_vs_plurality"] < -1e-9
                and item["minority_welfare_delta_vs_plurality"] > 1e-9
            )
        records.extend(scenario_rows)
    return pd.DataFrame.from_records(records)


def summarize_lhs_systems(results_df: pd.DataFrame) -> pd.DataFrame:
    """System-level summary for the sampled design."""
    summary = (
        results_df.groupby("system", as_index=False)
        .agg(
            divergence_rate=("aggregate_better_minority_worse", "mean"),
            alignment_rate=("improves_both_aggregate_and_minority", "mean"),
            minority_help_cost_rate=("minority_help_aggregate_cost", "mean"),
            mean_aggregate_delta=("aggregate_welfare_delta_vs_plurality", "mean"),
            mean_minority_delta=("minority_welfare_delta_vs_plurality", "mean"),
            mean_gap_delta=("welfare_gap_delta_vs_plurality", "mean"),
        )
    )
    summary["system"] = pd.Categorical(summary["system"], categories=SYSTEM_ORDER, ordered=True)
    return summary.sort_values("system").reset_index(drop=True)


def plot_lhs_system_summary(results_df: pd.DataFrame):
    """Compact bars summarizing the sampled-design results."""
    summary = summarize_lhs_systems(results_df)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2), dpi=150)

    sns.barplot(data=summary, x="system", y="divergence_rate", ax=axes[0], color="#e76f51")
    axes[0].set_title("Aggregate gains with minority losses", fontsize=10)
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Share of sampled cases")
    axes[0].tick_params(axis="x", rotation=35)

    sns.barplot(data=summary, x="system", y="alignment_rate", ax=axes[1], color="#2a9d8f")
    axes[1].set_title("Aggregate and minority gains together", fontsize=10)
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Share of sampled cases")
    axes[1].tick_params(axis="x", rotation=35)

    sns.barplot(data=summary, x="system", y="minority_help_cost_rate", ax=axes[2], color="#577590")
    axes[2].set_title("Minority gains with aggregate losses", fontsize=10)
    axes[2].set_xlabel("")
    axes[2].set_ylabel("Share of sampled cases")
    axes[2].tick_params(axis="x", rotation=35)

    fig.tight_layout()
    return fig


def plot_lhs_metric_map(
    results_df: pd.DataFrame,
    metric: str = "minority_welfare_delta_vs_plurality",
    systems: list[str] | None = None,
):
    """Small multiples over the sampled design space for a selected metric."""
    systems = SYSTEM_ORDER if systems is None else systems
    n_cols = 4
    n_rows = int(math.ceil(len(systems) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3.8 * n_rows), dpi=150)
    axes = np.atleast_1d(axes).ravel()
    palette = dict(zip(SLATE_ORDER, ["#264653", "#2a9d8f", "#e76f51"]))

    for ax, system_name in zip(axes, systems):
        subset = results_df[results_df["system"] == system_name]
        for slate_name in SLATE_ORDER:
            slate_subset = subset[subset["candidate_slate"] == slate_name]
            sc = ax.scatter(
                slate_subset["majority_share"],
                slate_subset["overlap_scale"],
                c=slate_subset[metric],
                cmap="coolwarm",
                vmin=results_df[metric].min(),
                vmax=results_df[metric].max(),
                s=70,
                edgecolors=palette[slate_name],
                linewidths=1.1,
                alpha=0.85,
            )
        ax.set_title(system_name, fontsize=9)
        ax.set_xlabel("Majority share")
        ax.set_ylabel("Overlap scale")
        ax.grid(True, alpha=0.18, linewidth=0.5)

    for ax in axes[len(systems):]:
        ax.axis("off")

    cbar = fig.colorbar(sc, ax=axes.tolist(), fraction=0.025, pad=0.02)
    cbar.set_label(metric.replace("_", " "))
    fig.tight_layout()
    return fig


def select_interesting_lhs_cases(results_df: pd.DataFrame) -> pd.DataFrame:
    """Pick a few representative sampled cases for deeper Monte Carlo checks."""
    rows = []

    divergence = results_df[results_df["aggregate_better_minority_worse"]].copy()
    if not divergence.empty:
        divergence["score"] = (
            divergence["aggregate_welfare_delta_vs_plurality"]
            - divergence["minority_welfare_delta_vs_plurality"]
        )
        row = divergence.sort_values("score", ascending=False).iloc[0].copy()
        row["diagnostic_label"] = "Aggregate up, minority down"
        rows.append(row)

    alignment = results_df[results_df["improves_both_aggregate_and_minority"]].copy()
    if not alignment.empty:
        alignment["score"] = (
            alignment["aggregate_welfare_delta_vs_plurality"]
            + alignment["minority_welfare_delta_vs_plurality"]
        )
        row = alignment.sort_values("score", ascending=False).iloc[0].copy()
        row["diagnostic_label"] = "Aggregate and minority up"
        rows.append(row)

    minority_help = results_df[results_df["minority_help_aggregate_cost"]].copy()
    if not minority_help.empty:
        minority_help["score"] = (
            minority_help["minority_welfare_delta_vs_plurality"]
            - minority_help["aggregate_welfare_delta_vs_plurality"]
        )
        row = minority_help.sort_values("score", ascending=False).iloc[0].copy()
        row["diagnostic_label"] = "Minority up, aggregate down"
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    cols = [
        "diagnostic_label",
        "system",
        "case_id",
        "majority_share",
        "minority_share",
        "overlap_scale",
        "dispersion_scale",
        "candidate_slate",
        "aggregate_welfare_delta_vs_plurality",
        "minority_welfare_delta_vs_plurality",
        "welfare_gap_delta_vs_plurality",
    ]
    return pd.DataFrame(rows)[cols].reset_index(drop=True)


def run_lhs_case_monte_carlo(
    case_row: pd.Series | dict,
    n_trials: int = 80,
    base_seed: int = 20260406,
    n_voters: int = 1800,
) -> pd.DataFrame:
    """Monte Carlo reruns for one sampled case against plurality."""
    target_system = case_row["system"]
    target_spec = next(spec for spec in SYSTEM_SPECS if spec.display_name == target_system)
    plurality_spec = next(spec for spec in SYSTEM_SPECS if spec.display_name == "Plurality")
    rows = []
    for trial in range(n_trials):
        electorate = _build_sampled_electorate(
            case_row,
            seed=base_seed + 1000 * int(case_row["case_id"]) + trial,
            n_voters=n_voters,
        )
        candidates = build_candidate_slate(case_row["candidate_slate"])
        plurality_result = _run_system_case(electorate, candidates, plurality_spec)
        target_result = _run_system_case(electorate, candidates, target_spec)
        aggregate_delta = target_result["aggregate_welfare"] - plurality_result["aggregate_welfare"]
        minority_delta = target_result["minority_welfare"] - plurality_result["minority_welfare"]
        majority_delta = target_result["majority_welfare"] - plurality_result["majority_welfare"]
        rows.append(
            {
                "diagnostic_label": case_row["diagnostic_label"],
                "system": target_system,
                "trial": trial,
                "case_id": int(case_row["case_id"]),
                "aggregate_welfare_delta_vs_plurality": aggregate_delta,
                "minority_welfare_delta_vs_plurality": minority_delta,
                "majority_welfare_delta_vs_plurality": majority_delta,
                "welfare_gap_delta_vs_plurality": (
                    target_result["welfare_gap"] - plurality_result["welfare_gap"]
                ),
                "aggregate_better_minority_worse": bool(
                    aggregate_delta > 1e-9 and minority_delta < -1e-9
                ),
                "improves_both_aggregate_and_minority": bool(
                    aggregate_delta > 1e-9 and minority_delta > 1e-9
                ),
                "minority_help_aggregate_cost": bool(
                    aggregate_delta < -1e-9 and minority_delta > 1e-9
                ),
            }
        )
    return pd.DataFrame(rows)


def run_selected_lhs_monte_carlo(
    selected_cases: pd.DataFrame,
    n_trials: int = 80,
    base_seed: int = 20260406,
    n_voters: int = 1800,
) -> pd.DataFrame:
    """Run Monte Carlo checks for a handful of selected sampled cases."""
    runs = [
        run_lhs_case_monte_carlo(
            row,
            n_trials=n_trials,
            base_seed=base_seed,
            n_voters=n_voters,
        )
        for _, row in selected_cases.iterrows()
    ]
    return pd.concat(runs, ignore_index=True) if runs else pd.DataFrame()


def summarize_lhs_monte_carlo(mc_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize Monte Carlo reruns for the selected cases."""
    return (
        mc_df.groupby(["diagnostic_label", "system"], as_index=False)
        .agg(
            mean_aggregate_delta=("aggregate_welfare_delta_vs_plurality", "mean"),
            std_aggregate_delta=("aggregate_welfare_delta_vs_plurality", "std"),
            mean_minority_delta=("minority_welfare_delta_vs_plurality", "mean"),
            std_minority_delta=("minority_welfare_delta_vs_plurality", "std"),
            divergence_rate=("aggregate_better_minority_worse", "mean"),
            alignment_rate=("improves_both_aggregate_and_minority", "mean"),
            minority_help_cost_rate=("minority_help_aggregate_cost", "mean"),
        )
    )


def plot_lhs_monte_carlo(mc_df: pd.DataFrame):
    """Visualize the Monte Carlo distributions for the selected diagnostic cases."""
    long_df = mc_df.melt(
        id_vars=["diagnostic_label", "system", "trial"],
        value_vars=[
            "aggregate_welfare_delta_vs_plurality",
            "minority_welfare_delta_vs_plurality",
        ],
        var_name="metric",
        value_name="delta",
    )
    label_map = {
        "aggregate_welfare_delta_vs_plurality": "Aggregate delta",
        "minority_welfare_delta_vs_plurality": "Minority delta",
    }
    long_df["metric"] = long_df["metric"].map(label_map)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.4), dpi=150)
    for ax, metric_name in zip(axes, ["Aggregate delta", "Minority delta"]):
        subset = long_df[long_df["metric"] == metric_name]
        sns.boxplot(
            data=subset,
            x="diagnostic_label",
            y="delta",
            color="#dceaf6",
            ax=ax,
        )
        sns.stripplot(
            data=subset,
            x="diagnostic_label",
            y="delta",
            hue="system",
            dodge=False,
            size=3.2,
            alpha=0.55,
            ax=ax,
        )
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
        ax.set_title(metric_name, fontsize=10)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=15)
        if ax is axes[0]:
            ax.legend(loc="best", fontsize=8, framealpha=0.9)
        else:
            ax.get_legend().remove()
    fig.tight_layout()
    return fig
