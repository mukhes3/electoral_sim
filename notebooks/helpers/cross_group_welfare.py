"""Notebook-specific helpers for cross-group versus aggregate welfare analysis."""
from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from electoral_sim.ballots import BallotProfile
from electoral_sim.candidates import CandidateSet
from electoral_sim.electorate import gaussian_mixture_electorate
from electoral_sim.fractional import FractionalBallotContinuous
from electoral_sim.metrics import compute_group_metrics, compute_metrics
from electoral_sim.systems import (
    ApprovalVoting,
    CondorcetSchulze,
    InstantRunoff,
    PartyListPR,
    Plurality,
    ScoreVoting,
)


RATIO_ORDER = ["60:40", "90:10", "95:5", "99:1"]
OVERLAP_ORDER = ["Low overlap", "Medium overlap", "High overlap"]
DISPERSION_ORDER = ["Tight groups", "Broad groups"]
SLATE_ORDER = ["Balanced slate", "Bridge slate", "Majority-heavy slate"]
SYSTEM_ORDER = [
    "Plurality",
    "IRV",
    "Approval",
    "Score",
    "Condorcet",
    "Party-list PR",
    "Fractional",
]

DISPLAY_LABELS = {
    "60:40": "60:40",
    "90:10": "90:10",
    "95:5": "95:5",
    "99:1": "99:1",
    "Low overlap": "Low\noverlap",
    "Medium overlap": "Medium\noverlap",
    "High overlap": "High\noverlap",
    "Tight groups": "Tight",
    "Broad groups": "Broad",
    "Balanced slate": "Balanced",
    "Bridge slate": "Bridge",
    "Majority-heavy slate": "Majority-heavy",
}

RATIO_WEIGHTS = {
    "60:40": (0.60, 0.40),
    "90:10": (0.90, 0.10),
    "95:5": (0.95, 0.05),
    "99:1": (0.99, 0.01),
}

OVERLAP_MEANS = {
    "Low overlap": ([0.24, 0.56], [0.76, 0.44]),
    "Medium overlap": ([0.35, 0.54], [0.65, 0.46]),
    "High overlap": ([0.43, 0.52], [0.57, 0.48]),
}

DISPERSION_COVS = {
    "Tight groups": [[0.007, 0.001], [0.001, 0.009]],
    "Broad groups": [[0.018, 0.002], [0.002, 0.020]],
}

SLATE_SPECS = {
    "Balanced slate": {
        "positions": np.array(
            [
                [0.14, 0.62],
                [0.28, 0.56],
                [0.44, 0.51],
                [0.56, 0.49],
                [0.72, 0.44],
                [0.86, 0.38],
            ]
        ),
        "labels": ["L-Base", "L-Bridge", "L-Center", "R-Center", "R-Bridge", "R-Base"],
    },
    "Bridge slate": {
        "positions": np.array(
            [
                [0.18, 0.60],
                [0.34, 0.54],
                [0.50, 0.50],
                [0.66, 0.46],
                [0.82, 0.40],
            ]
        ),
        "labels": ["L-Base", "L-Bridge", "Bridge", "R-Bridge", "R-Base"],
    },
    "Majority-heavy slate": {
        "positions": np.array(
            [
                [0.10, 0.64],
                [0.22, 0.58],
                [0.34, 0.54],
                [0.48, 0.50],
                [0.70, 0.46],
                [0.84, 0.40],
            ]
        ),
        "labels": ["L-Base", "L-Plus", "L-Bridge", "Center", "R-Bridge", "R-Base"],
    },
}

APPROVAL_THRESHOLD = 0.65


@dataclass(frozen=True)
class SystemSpec:
    key: str
    display_name: str

    def build(self):
        if self.key == "plurality":
            return Plurality()
        if self.key == "irv":
            return InstantRunoff()
        if self.key == "approval":
            return ApprovalVoting()
        if self.key == "score":
            return ScoreVoting()
        if self.key == "condorcet":
            return CondorcetSchulze()
        if self.key == "party_list_pr":
            return PartyListPR()
        if self.key == "fractional":
            return FractionalBallotContinuous(sigma=0.3)
        raise ValueError(f"Unknown system key: {self.key}")


SYSTEM_SPECS = [
    SystemSpec("plurality", "Plurality"),
    SystemSpec("irv", "IRV"),
    SystemSpec("approval", "Approval"),
    SystemSpec("score", "Score"),
    SystemSpec("condorcet", "Condorcet"),
    SystemSpec("party_list_pr", "Party-list PR"),
    SystemSpec("fractional", "Fractional"),
]


def build_two_group_electorate(
    ratio_name: str,
    overlap_name: str,
    dispersion_name: str,
    seed: int,
    n_voters: int = 1800,
    dim_names: list[str] | None = None,
):
    """
    Build a two-group electorate with labeled majority and minority groups.

    The notebook varies group size, overlap, and internal dispersion while
    keeping the rest of the simulation logic fixed.
    """
    majority_weight, minority_weight = RATIO_WEIGHTS[ratio_name]
    majority_mean, minority_mean = OVERLAP_MEANS[overlap_name]
    cov = DISPERSION_COVS[dispersion_name]
    rng = np.random.default_rng(seed)
    return gaussian_mixture_electorate(
        n_voters,
        [
            {
                "weight": majority_weight,
                "mean": majority_mean,
                "cov": cov,
                "group": "Majority group",
            },
            {
                "weight": minority_weight,
                "mean": minority_mean,
                "cov": cov,
                "group": "Minority group",
            },
        ],
        rng=rng,
        dim_names=dim_names or ["economic", "social"],
    )


def build_candidate_slate(name: str) -> CandidateSet:
    spec = SLATE_SPECS[name]
    return CandidateSet(spec["positions"].copy(), list(spec["labels"]))


def helper_overview() -> pd.DataFrame:
    """Compact table documenting the grid used by the notebook."""
    return pd.DataFrame(
        {
            "dimension": [
                "Population ratios",
                "Between-group overlap",
                "Within-group dispersion",
                "Candidate slates",
                "Electoral systems",
            ],
            "values": [
                ", ".join(RATIO_ORDER),
                ", ".join(OVERLAP_ORDER),
                ", ".join(DISPERSION_ORDER),
                ", ".join(SLATE_ORDER),
                ", ".join(SYSTEM_ORDER),
            ],
        }
    )


def cross_group_helper_overview() -> pd.DataFrame:
    """Backward-friendly alias used by the notebook import cell."""
    return helper_overview()


def _group_scatter(ax, electorate, candidates, title: str, show_legend: bool = False):
    """Plot a compact group-colored spatial view for notebook previews."""
    colors = {name: color for name, color in zip(["Majority group", "Minority group"], ["#2a9d8f", "#e76f51"])}
    for group_id, name in electorate.group_names.items():
        mask = electorate.group_ids == group_id
        ax.scatter(
            electorate.preferences[mask, 0],
            electorate.preferences[mask, 1],
            s=10,
            alpha=0.22,
            color=colors.get(name, "#457b9d"),
            label=name,
            rasterized=True,
        )
    ax.scatter(
        candidates.positions[:, 0],
        candidates.positions[:, 1],
        s=70,
        marker="X",
        color="black",
        linewidths=0.8,
    )
    for pos, label in zip(candidates.positions, candidates.labels):
        ax.annotate(label, xy=(pos[0], pos[1]), xytext=(4, 5), textcoords="offset points", fontsize=6.5)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("economic", fontsize=8)
    ax.set_ylabel("social", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.18, linewidth=0.5)
    if show_legend:
        ax.legend(loc="upper left", fontsize=7, framealpha=0.9)


def plot_electorate_gallery(
    dispersion_name: str,
    slate_name: str = "Balanced slate",
    seed: int = 20260403,
    n_voters: int = 1000,
):
    """Show all ratio/overlap combinations for one dispersion regime."""
    candidates = build_candidate_slate(slate_name)
    fig, axes = plt.subplots(len(RATIO_ORDER), len(OVERLAP_ORDER), figsize=(12, 12), dpi=150)
    for row_idx, ratio_name in enumerate(RATIO_ORDER):
        for col_idx, overlap_name in enumerate(OVERLAP_ORDER):
            electorate = build_two_group_electorate(
                ratio_name,
                overlap_name,
                dispersion_name,
                seed + 50 * row_idx + col_idx,
                n_voters=n_voters,
            )
            title = f"{ratio_name}, {DISPLAY_LABELS[overlap_name].replace(chr(10), ' ')}"
            _group_scatter(
                axes[row_idx, col_idx],
                electorate,
                candidates,
                title=title,
                show_legend=(row_idx == 0 and col_idx == 0),
            )
    fig.suptitle(f"Electorates with {dispersion_name.lower()}", fontsize=13, y=1.01)
    fig.tight_layout()
    return fig


def plot_candidate_slate_gallery(
    ratio_name: str = "90:10",
    overlap_name: str = "Medium overlap",
    dispersion_name: str = "Tight groups",
    seed: int = 20260403,
    n_voters: int = 1000,
):
    """Show the hand-built slates on a common electorate."""
    electorate = build_two_group_electorate(
        ratio_name,
        overlap_name,
        dispersion_name,
        seed=seed,
        n_voters=n_voters,
    )
    fig, axes = plt.subplots(1, len(SLATE_ORDER), figsize=(15, 4), dpi=150)
    for ax, slate_name in zip(axes, SLATE_ORDER):
        candidates = build_candidate_slate(slate_name)
        _group_scatter(ax, electorate, candidates, title=slate_name, show_legend=(slate_name == SLATE_ORDER[0]))
    fig.tight_layout()
    return fig


def _extract_group_metrics(group_summary):
    rows = {group.group_name: group for group in group_summary.groups}
    majority = rows["Majority group"]
    minority = rows["Minority group"]
    return majority, minority


def _run_system_case(electorate, candidates, system_spec: SystemSpec):
    ballots = BallotProfile.from_preferences(
        electorate,
        candidates,
        approval_threshold=APPROVAL_THRESHOLD,
    )
    system = system_spec.build()
    result = system.run(ballots, candidates)
    metrics = compute_metrics(result, electorate, candidates)
    group_summary = compute_group_metrics(result, electorate, candidates)
    majority, minority = _extract_group_metrics(group_summary)
    aggregate_welfare = -metrics.mean_voter_distance
    majority_welfare = majority.welfare
    minority_welfare = minority.welfare
    return {
        "system": system_spec.display_name,
        "aggregate_welfare": aggregate_welfare,
        "aggregate_mean_distance": metrics.mean_voter_distance,
        "majority_welfare": majority_welfare,
        "majority_mean_distance": majority.mean_voter_distance,
        "minority_welfare": minority_welfare,
        "minority_mean_distance": minority.mean_voter_distance,
        "majority_satisfaction": majority.majority_satisfaction,
        "minority_satisfaction": minority.majority_satisfaction,
        "welfare_gap": majority_welfare - minority_welfare,
        "distance_gap": minority.mean_voter_distance - majority.mean_voter_distance,
        "distance_to_median": metrics.distance_to_median,
    }


def run_full_grid(
    base_seed: int = 20260403,
    n_voters: int = 1800,
    dim_names: list[str] | None = None,
) -> pd.DataFrame:
    """Run the full structured grid and return one compact row per system-case."""
    records = []
    case_index = 0
    for ratio_name in RATIO_ORDER:
        for overlap_name in OVERLAP_ORDER:
            for dispersion_name in DISPERSION_ORDER:
                for slate_name in SLATE_ORDER:
                    electorate = build_two_group_electorate(
                        ratio_name,
                        overlap_name,
                        dispersion_name,
                        seed=base_seed + case_index,
                        n_voters=n_voters,
                        dim_names=dim_names,
                    )
                    candidates = build_candidate_slate(slate_name)
                    scenario_rows = []
                    for system_spec in SYSTEM_SPECS:
                        row = _run_system_case(electorate, candidates, system_spec)
                        row.update(
                            {
                                "ratio_profile": ratio_name,
                                "overlap_profile": overlap_name,
                                "dispersion_profile": dispersion_name,
                                "candidate_slate": slate_name,
                                "case_id": case_index,
                            }
                        )
                        scenario_rows.append(row)

                    plurality_row = next(row for row in scenario_rows if row["system"] == "Plurality")
                    for row in scenario_rows:
                        row["aggregate_welfare_delta_vs_plurality"] = (
                            row["aggregate_welfare"] - plurality_row["aggregate_welfare"]
                        )
                        row["majority_welfare_delta_vs_plurality"] = (
                            row["majority_welfare"] - plurality_row["majority_welfare"]
                        )
                        row["minority_welfare_delta_vs_plurality"] = (
                            row["minority_welfare"] - plurality_row["minority_welfare"]
                        )
                        row["welfare_gap_delta_vs_plurality"] = (
                            row["welfare_gap"] - plurality_row["welfare_gap"]
                        )
                        row["aggregate_better_minority_worse"] = bool(
                            row["aggregate_welfare_delta_vs_plurality"] > 1e-9
                            and row["minority_welfare_delta_vs_plurality"] < -1e-9
                        )
                        row["improves_both_aggregate_and_minority"] = bool(
                            row["aggregate_welfare_delta_vs_plurality"] > 1e-9
                            and row["minority_welfare_delta_vs_plurality"] > 1e-9
                        )
                    records.extend(scenario_rows)
                    case_index += 1

    return pd.DataFrame.from_records(records)


def run_cross_group_grid(
    base_seed: int = 20260403,
    n_voters: int = 1800,
    dim_names: list[str] | None = None,
) -> pd.DataFrame:
    """Notebook-facing alias for the full structured comparison grid."""
    return run_full_grid(
        base_seed=base_seed,
        n_voters=n_voters,
        dim_names=dim_names,
    )


def summarize_system_patterns(results_df: pd.DataFrame) -> pd.DataFrame:
    """System-level summary for the full notebook grid."""
    summary = (
        results_df.groupby("system", as_index=False)
        .agg(
            divergence_rate=("aggregate_better_minority_worse", "mean"),
            alignment_rate=("improves_both_aggregate_and_minority", "mean"),
            mean_aggregate_delta=("aggregate_welfare_delta_vs_plurality", "mean"),
            mean_minority_delta=("minority_welfare_delta_vs_plurality", "mean"),
            mean_gap_delta=("welfare_gap_delta_vs_plurality", "mean"),
            mean_distance_to_median=("distance_to_median", "mean"),
        )
    )
    summary["system"] = pd.Categorical(summary["system"], categories=SYSTEM_ORDER, ordered=True)
    return summary.sort_values("system").reset_index(drop=True)


def summarize_condition_patterns(results_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate over systems to see which conditions most often create divergence."""
    summary = (
        results_df.groupby(
            ["ratio_profile", "overlap_profile", "dispersion_profile", "candidate_slate"],
            as_index=False,
        )
        .agg(
            divergence_share=("aggregate_better_minority_worse", "mean"),
            alignment_share=("improves_both_aggregate_and_minority", "mean"),
            mean_aggregate_delta=("aggregate_welfare_delta_vs_plurality", "mean"),
            mean_minority_delta=("minority_welfare_delta_vs_plurality", "mean"),
            mean_gap_delta=("welfare_gap_delta_vs_plurality", "mean"),
        )
    )
    return summary.sort_values(
        ["divergence_share", "mean_aggregate_delta", "mean_minority_delta"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def top_divergence_cases(results_df: pd.DataFrame, n: int = 12) -> pd.DataFrame:
    """Most striking cases where aggregate welfare improves but minority welfare falls."""
    cols = [
        "system",
        "ratio_profile",
        "overlap_profile",
        "dispersion_profile",
        "candidate_slate",
        "aggregate_welfare_delta_vs_plurality",
        "minority_welfare_delta_vs_plurality",
        "welfare_gap_delta_vs_plurality",
    ]
    divergence = results_df[results_df["aggregate_better_minority_worse"]].copy()
    divergence["divergence_score"] = (
        divergence["aggregate_welfare_delta_vs_plurality"]
        - divergence["minority_welfare_delta_vs_plurality"]
    )
    return divergence.sort_values("divergence_score", ascending=False)[cols].head(n).reset_index(drop=True)


def top_alignment_cases(results_df: pd.DataFrame, n: int = 12) -> pd.DataFrame:
    """Cases where a system improves both aggregate and minority welfare."""
    cols = [
        "system",
        "ratio_profile",
        "overlap_profile",
        "dispersion_profile",
        "candidate_slate",
        "aggregate_welfare_delta_vs_plurality",
        "minority_welfare_delta_vs_plurality",
        "welfare_gap_delta_vs_plurality",
    ]
    alignment = results_df[results_df["improves_both_aggregate_and_minority"]].copy()
    alignment["alignment_score"] = (
        alignment["aggregate_welfare_delta_vs_plurality"]
        + alignment["minority_welfare_delta_vs_plurality"]
    )
    return alignment.sort_values("alignment_score", ascending=False)[cols].head(n).reset_index(drop=True)


def plot_system_summary(results_df: pd.DataFrame):
    """Compact bar charts for the main system-level patterns."""
    summary = summarize_system_patterns(results_df)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2), dpi=150)

    sns.barplot(data=summary, x="system", y="divergence_rate", ax=axes[0], color="#e76f51")
    axes[0].set_title("How often aggregate gains hide minority losses", fontsize=10)
    axes[0].set_ylabel("Share of cases")
    axes[0].set_xlabel("")
    axes[0].tick_params(axis="x", rotation=35)

    sns.barplot(data=summary, x="system", y="alignment_rate", ax=axes[1], color="#2a9d8f")
    axes[1].set_title("How often both aggregate and minority welfare improve", fontsize=10)
    axes[1].set_ylabel("Share of cases")
    axes[1].set_xlabel("")
    axes[1].tick_params(axis="x", rotation=35)

    sns.barplot(data=summary, x="system", y="mean_gap_delta", ax=axes[2], color="#457b9d")
    axes[2].axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    axes[2].set_title("Average change in the welfare gap vs plurality", fontsize=10)
    axes[2].set_ylabel("Gap delta")
    axes[2].set_xlabel("")
    axes[2].tick_params(axis="x", rotation=35)

    fig.tight_layout()
    return fig


def plot_condition_heatmaps(results_df: pd.DataFrame):
    """Heatmaps showing where divergence is most common after averaging over slates."""
    aggregated = (
        results_df.groupby(
            ["system", "ratio_profile", "overlap_profile", "dispersion_profile"],
            as_index=False,
        )
        .agg(
            divergence_rate=("aggregate_better_minority_worse", "mean"),
            alignment_rate=("improves_both_aggregate_and_minority", "mean"),
            mean_gap_delta=("welfare_gap_delta_vs_plurality", "mean"),
        )
    )

    fig, axes = plt.subplots(len(DISPERSION_ORDER), len(SYSTEM_ORDER), figsize=(18, 6.5), dpi=150)
    for row_idx, dispersion_name in enumerate(DISPERSION_ORDER):
        subset = aggregated[aggregated["dispersion_profile"] == dispersion_name]
        for col_idx, system_name in enumerate(SYSTEM_ORDER):
            ax = axes[row_idx, col_idx]
            system_subset = subset[subset["system"] == system_name]
            pivot = (
                system_subset.pivot(
                    index="ratio_profile",
                    columns="overlap_profile",
                    values="divergence_rate",
                )
                .reindex(index=RATIO_ORDER, columns=OVERLAP_ORDER)
            )
            sns.heatmap(
                pivot,
                ax=ax,
                cmap="Reds",
                vmin=0.0,
                vmax=1.0,
                cbar=(row_idx == 0 and col_idx == len(SYSTEM_ORDER) - 1),
                annot=True,
                fmt=".2f",
                linewidths=0.5,
                linecolor="white",
            )
            ax.set_title(system_name, fontsize=9)
            ax.set_xlabel("" if row_idx == 0 else "Overlap")
            ax.set_ylabel(dispersion_name if col_idx == 0 else "")
            ax.tick_params(labelsize=7)
    fig.suptitle("Share of cases where aggregate welfare rises but minority welfare falls", fontsize=13, y=1.02)
    fig.tight_layout()
    return fig


def case_record(
    ratio_name: str,
    overlap_name: str,
    dispersion_name: str,
    slate_name: str,
) -> dict[str, str]:
    """Small metadata bundle describing one electorate-and-slate case."""
    return {
        "ratio_profile": ratio_name,
        "overlap_profile": overlap_name,
        "dispersion_profile": dispersion_name,
        "candidate_slate": slate_name,
    }


def build_case_from_record(
    record: pd.Series | dict,
    base_seed: int = 20260403,
    n_voters: int = 1800,
    dim_names: list[str] | None = None,
):
    """Rebuild an electorate and candidate slate from a result row or metadata dict."""
    ratio_name = record["ratio_profile"]
    overlap_name = record["overlap_profile"]
    dispersion_name = record["dispersion_profile"]
    slate_name = record["candidate_slate"]

    if "case_id" in record:
        seed = base_seed + int(record["case_id"])
    else:
        case_lookup = 0
        for ratio in RATIO_ORDER:
            for overlap in OVERLAP_ORDER:
                for dispersion in DISPERSION_ORDER:
                    for slate in SLATE_ORDER:
                        if (
                            ratio == ratio_name
                            and overlap == overlap_name
                            and dispersion == dispersion_name
                            and slate == slate_name
                        ):
                            seed = base_seed + case_lookup
                            electorate = build_two_group_electorate(
                                ratio_name,
                                overlap_name,
                                dispersion_name,
                                seed=seed,
                                n_voters=n_voters,
                                dim_names=dim_names,
                            )
                            return electorate, build_candidate_slate(slate_name), seed
                        case_lookup += 1
        raise ValueError("Could not reconstruct case from metadata.")

    electorate = build_two_group_electorate(
        ratio_name,
        overlap_name,
        dispersion_name,
        seed=seed,
        n_voters=n_voters,
        dim_names=dim_names,
    )
    return electorate, build_candidate_slate(slate_name), seed


def run_case_systems(
    record: pd.Series | dict,
    systems: tuple[str, ...] = ("Plurality", "Fractional"),
    base_seed: int = 20260403,
    n_voters: int = 1800,
    dim_names: list[str] | None = None,
):
    """Run a selected set of systems on one reconstructed case."""
    electorate, candidates, seed = build_case_from_record(
        record,
        base_seed=base_seed,
        n_voters=n_voters,
        dim_names=dim_names,
    )
    ballots = BallotProfile.from_preferences(
        electorate,
        candidates,
        approval_threshold=APPROVAL_THRESHOLD,
    )
    outputs = {}
    for spec in SYSTEM_SPECS:
        if spec.display_name not in systems:
            continue
        system = spec.build()
        result = system.run(ballots, candidates)
        metrics = compute_metrics(result, electorate, candidates)
        group_summary = compute_group_metrics(result, electorate, candidates)
        outputs[spec.display_name] = {
            "result": result,
            "metrics": metrics,
            "group_summary": group_summary,
        }
    return electorate, candidates, outputs, seed


def representative_divergence_cases(results_df: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    """Top distinct divergence cases, one row per electorate-and-slate combination."""
    divergence = results_df[results_df["aggregate_better_minority_worse"]].copy()
    divergence["divergence_score"] = (
        divergence["aggregate_welfare_delta_vs_plurality"]
        - divergence["minority_welfare_delta_vs_plurality"]
    )
    divergence = divergence.sort_values("divergence_score", ascending=False)
    distinct = divergence.drop_duplicates(
        subset=["ratio_profile", "overlap_profile", "dispersion_profile", "candidate_slate"]
    )
    cols = [
        "system",
        "ratio_profile",
        "overlap_profile",
        "dispersion_profile",
        "candidate_slate",
        "aggregate_welfare_delta_vs_plurality",
        "minority_welfare_delta_vs_plurality",
        "welfare_gap_delta_vs_plurality",
    ]
    return distinct[cols].head(n).reset_index(drop=True)


def plot_representative_divergence_cases(
    results_df: pd.DataFrame,
    n: int = 3,
    base_seed: int = 20260403,
    n_voters: int = 1800,
):
    """Plot plurality and fractional outcomes for a few representative disagreement cases."""
    selected = representative_divergence_cases(results_df, n=n)
    fig, axes = plt.subplots(1, len(selected), figsize=(5.5 * len(selected), 4.8), dpi=150)
    if len(selected) == 1:
        axes = [axes]

    colors = {"Majority group": "#2a9d8f", "Minority group": "#e76f51"}
    for ax, (_, row) in zip(axes, selected.iterrows()):
        electorate, candidates, outputs, _ = run_case_systems(
            row,
            systems=("Plurality", "Fractional"),
            base_seed=base_seed,
            n_voters=n_voters,
        )
        for group_id, group_name in electorate.group_names.items():
            mask = electorate.group_ids == group_id
            ax.scatter(
                electorate.preferences[mask, 0],
                electorate.preferences[mask, 1],
                s=10,
                alpha=0.20,
                color=colors[group_name],
                rasterized=True,
            )
            group_mean = electorate.preferences[mask].mean(axis=0)
            ax.scatter(
                group_mean[0],
                group_mean[1],
                s=90,
                marker="D",
                color=colors[group_name],
                edgecolors="white",
                linewidths=1.0,
            )

        ax.scatter(
            candidates.positions[:, 0],
            candidates.positions[:, 1],
            s=65,
            marker="X",
            color="black",
            linewidths=0.8,
        )
        for pos, label in zip(candidates.positions, candidates.labels):
            ax.annotate(label, xy=(pos[0], pos[1]), xytext=(4, 4), textcoords="offset points", fontsize=6.2)

        median = electorate.geometric_median()
        ax.scatter(median[0], median[1], s=110, marker="P", color="#264653", edgecolors="white", linewidths=1.0)
        plurality_outcome = outputs["Plurality"]["result"].outcome_position
        fractional_outcome = outputs["Fractional"]["result"].outcome_position
        ax.scatter(plurality_outcome[0], plurality_outcome[1], s=120, marker="o", color="#577590", edgecolors="white", linewidths=1.0)
        ax.scatter(fractional_outcome[0], fractional_outcome[1], s=130, marker="*", color="#e63946", edgecolors="black", linewidths=0.8)

        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("economic", fontsize=8)
        ax.set_ylabel("social", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.18, linewidth=0.5)
        ax.set_title(
            f"{row['ratio_profile']}, {row['overlap_profile']}\n{row['dispersion_profile']}, {row['candidate_slate']}",
            fontsize=9,
        )

    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color="#577590", markersize=7, label="Plurality outcome"),
        plt.Line2D([], [], marker="*", linestyle="", color="#e63946", markeredgecolor="black", markersize=10, label="Fractional outcome"),
        plt.Line2D([], [], marker="P", linestyle="", color="#264653", markersize=8, label="Electorate median"),
        plt.Line2D([], [], marker="D", linestyle="", color="#2a9d8f", markersize=7, label="Group means"),
    ]
    fig.legend(handles=handles, loc="upper center", ncols=4, frameon=True, fontsize=8, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout()
    return fig


def group_welfare_decomposition(
    record: pd.Series | dict,
    system_name: str = "Fractional",
    baseline_name: str = "Plurality",
    base_seed: int = 20260403,
    n_voters: int = 1800,
) -> pd.DataFrame:
    """
    Break a welfare change into group-level contributions.

    Because aggregate welfare is a weighted average of group welfare, this table
    makes it visible when a small group's loss is being dominated by a larger
    group's gain.
    """
    electorate, _, outputs, _ = run_case_systems(
        record,
        systems=(baseline_name, system_name),
        base_seed=base_seed,
        n_voters=n_voters,
    )
    baseline_groups = {group.group_name: group for group in outputs[baseline_name]["group_summary"].groups}
    system_groups = {group.group_name: group for group in outputs[system_name]["group_summary"].groups}

    rows = []
    for group_name in ["Majority group", "Minority group"]:
        baseline_group = baseline_groups[group_name]
        system_group = system_groups[group_name]
        delta = system_group.welfare - baseline_group.welfare
        rows.append(
            {
                "group": group_name,
                "population_share": baseline_group.population_share,
                f"{baseline_name.lower()}_welfare": baseline_group.welfare,
                f"{system_name.lower()}_welfare": system_group.welfare,
                "welfare_delta": delta,
                "contribution_to_aggregate_delta": baseline_group.population_share * delta,
            }
        )

    total_delta = outputs[system_name]["metrics"].mean_voter_distance * -1 - outputs[baseline_name]["metrics"].mean_voter_distance * -1
    rows.append(
        {
            "group": "Total",
            "population_share": 1.0,
            f"{baseline_name.lower()}_welfare": outputs[baseline_name]["metrics"].mean_voter_distance * -1,
            f"{system_name.lower()}_welfare": outputs[system_name]["metrics"].mean_voter_distance * -1,
            "welfare_delta": total_delta,
            "contribution_to_aggregate_delta": sum(row["contribution_to_aggregate_delta"] for row in rows),
        }
    )
    return pd.DataFrame(rows)


def candidate_support_table(
    record: pd.Series | dict,
    base_seed: int = 20260403,
    n_voters: int = 1800,
) -> pd.DataFrame:
    """Compare candidate proximity and first-choice support across groups."""
    electorate, candidates, _, _ = run_case_systems(
        record,
        systems=("Plurality",),
        base_seed=base_seed,
        n_voters=n_voters,
    )
    ballots = BallotProfile.from_preferences(
        electorate,
        candidates,
        approval_threshold=APPROVAL_THRESHOLD,
    )
    majority_mask = electorate.group_ids == 0
    minority_mask = electorate.group_ids == 1
    majority_center = electorate.preferences[majority_mask].mean(axis=0)
    minority_center = electorate.preferences[minority_mask].mean(axis=0)
    support = pd.DataFrame(
        {
            "candidate": candidates.labels,
            "distance_to_majority_mean": np.linalg.norm(candidates.positions - majority_center, axis=1),
            "distance_to_minority_mean": np.linalg.norm(candidates.positions - minority_center, axis=1),
            "majority_first_choice_share": (
                np.bincount(ballots.plurality[majority_mask], minlength=candidates.n_candidates)
                / max(int(majority_mask.sum()), 1)
            ),
            "minority_first_choice_share": (
                np.bincount(ballots.plurality[minority_mask], minlength=candidates.n_candidates)
                / max(int(minority_mask.sum()), 1)
            ),
        }
    )
    support["support_gap"] = support["majority_first_choice_share"] - support["minority_first_choice_share"]
    return support.sort_values(
        ["minority_first_choice_share", "majority_first_choice_share"],
        ascending=False,
    ).reset_index(drop=True)


def overlap_counterfactual_table(
    record: pd.Series | dict,
    system_name: str = "Fractional",
    baseline_name: str = "Plurality",
    base_seed: int = 20260403,
    n_voters: int = 1800,
) -> pd.DataFrame:
    """Hold ratio, dispersion, and slate fixed while varying only overlap."""
    rows = []
    for overlap_name in OVERLAP_ORDER:
        counterfactual = case_record(
            ratio_name=record["ratio_profile"],
            overlap_name=overlap_name,
            dispersion_name=record["dispersion_profile"],
            slate_name=record["candidate_slate"],
        )
        electorate, _, outputs, _ = run_case_systems(
            counterfactual,
            systems=(baseline_name, system_name),
            base_seed=base_seed,
            n_voters=n_voters,
        )
        baseline_metrics = outputs[baseline_name]["metrics"]
        system_metrics = outputs[system_name]["metrics"]
        baseline_groups = {group.group_name: group for group in outputs[baseline_name]["group_summary"].groups}
        system_groups = {group.group_name: group for group in outputs[system_name]["group_summary"].groups}
        rows.append(
            {
                "overlap_profile": overlap_name,
                "aggregate_welfare_delta_vs_plurality": -system_metrics.mean_voter_distance + baseline_metrics.mean_voter_distance,
                "minority_welfare_delta_vs_plurality": (
                    system_groups["Minority group"].welfare - baseline_groups["Minority group"].welfare
                ),
                "majority_welfare_delta_vs_plurality": (
                    system_groups["Majority group"].welfare - baseline_groups["Majority group"].welfare
                ),
                "welfare_gap_delta_vs_plurality": (
                    (system_groups["Majority group"].welfare - system_groups["Minority group"].welfare)
                    - (baseline_groups["Majority group"].welfare - baseline_groups["Minority group"].welfare)
                ),
                "aggregate_better_minority_worse": bool(
                    (-system_metrics.mean_voter_distance + baseline_metrics.mean_voter_distance) > 1e-9
                    and (system_groups["Minority group"].welfare - baseline_groups["Minority group"].welfare) < -1e-9
                ),
                "n_voters": electorate.n_voters,
            }
        )
    return pd.DataFrame(rows)
