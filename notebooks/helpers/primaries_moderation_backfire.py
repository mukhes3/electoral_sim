"""Notebook-specific helpers for the primaries moderation notebook."""
from __future__ import annotations

import gc

from IPython.display import Markdown, display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from electoral_sim.ballots import BallotProfile
from electoral_sim.candidates import CandidateSet
from electoral_sim.electorate import gaussian_mixture_electorate
from electoral_sim.primaries import (
    PartySpec,
    PrimaryType,
    run_open_primary_top_k,
    run_two_party_primary,
)
from electoral_sim.strategies.base import StrategyModel
from electoral_sim.systems import InstantRunoff, Plurality
from electoral_sim.utils import plot_electorate


VOTER_ORDER = [
    "Balanced polarized",
    "Centrist majority",
    "Dominant left",
    "High overlap",
    "Factionalized left majority",
]
CANDIDATE_ORDER = [
    "Symmetric ladder",
    "Moderate rich",
    "Activist heavy",
    "Asymmetric insurgency",
]
TURNOUT_ORDER = [
    "Even participation",
    "Activist primary",
    "Moderate primary",
    "Right surge",
]
PIPELINE_ORDER = [
    "Closed plurality",
    "Semi plurality",
    "Closed IRV",
    "Top-4 open IRV",
]

DISPLAY_LABELS = {
    "Balanced polarized": "Balanced\npolarized",
    "Centrist majority": "Centrist\nmajority",
    "Dominant left": "Dominant\nleft",
    "High overlap": "High\noverlap",
    "Factionalized left majority": "Factionalized\nleft majority",
    "Symmetric ladder": "Symmetric\nladder",
    "Moderate rich": "Moderate\nrich",
    "Activist heavy": "Activist\nheavy",
    "Asymmetric insurgency": "Asymmetric\ninsurgency",
    "Even participation": "Even",
    "Activist primary": "Activist",
    "Moderate primary": "Moderate",
    "Right surge": "Right surge",
}

VOTER_SPECS = {
    "Balanced polarized": [
        {"weight": 0.46, "mean": [0.24, 0.56], "cov": [[0.010, 0.003], [0.003, 0.014]]},
        {"weight": 0.46, "mean": [0.76, 0.44], "cov": [[0.010, -0.003], [-0.003, 0.014]]},
        {"weight": 0.08, "mean": [0.50, 0.50], "cov": [[0.006, 0.0], [0.0, 0.006]]},
    ],
    "Centrist majority": [
        {"weight": 0.54, "mean": [0.50, 0.50], "cov": [[0.010, 0.0], [0.0, 0.010]]},
        {"weight": 0.23, "mean": [0.23, 0.54], "cov": [[0.009, 0.002], [0.002, 0.012]]},
        {"weight": 0.23, "mean": [0.77, 0.46], "cov": [[0.009, -0.002], [-0.002, 0.012]]},
    ],
    "Dominant left": [
        {"weight": 0.52, "mean": [0.30, 0.54], "cov": [[0.010, 0.002], [0.002, 0.012]]},
        {"weight": 0.22, "mean": [0.44, 0.50], "cov": [[0.008, 0.001], [0.001, 0.010]]},
        {"weight": 0.26, "mean": [0.74, 0.44], "cov": [[0.012, -0.002], [-0.002, 0.014]]},
    ],
    "High overlap": [
        {"weight": 0.48, "mean": [0.40, 0.53], "cov": [[0.012, 0.002], [0.002, 0.012]]},
        {"weight": 0.48, "mean": [0.60, 0.47], "cov": [[0.012, -0.002], [-0.002, 0.012]]},
        {"weight": 0.04, "mean": [0.50, 0.50], "cov": [[0.004, 0.0], [0.0, 0.004]]},
    ],
    "Factionalized left majority": [
        {"weight": 0.28, "mean": [0.18, 0.62], "cov": [[0.008, 0.002], [0.002, 0.010]]},
        {"weight": 0.28, "mean": [0.36, 0.50], "cov": [[0.008, 0.002], [0.002, 0.010]]},
        {"weight": 0.18, "mean": [0.52, 0.52], "cov": [[0.006, 0.0], [0.0, 0.008]]},
        {"weight": 0.26, "mean": [0.78, 0.42], "cov": [[0.011, -0.002], [-0.002, 0.012]]},
    ],
}

CANDIDATE_SPECS = {
    "Symmetric ladder": np.array([
        [0.14, 0.60], [0.28, 0.54], [0.40, 0.48],
        [0.60, 0.52], [0.72, 0.46], [0.86, 0.40],
    ]),
    "Moderate rich": np.array([
        [0.24, 0.56], [0.34, 0.52], [0.44, 0.50],
        [0.56, 0.50], [0.66, 0.48], [0.76, 0.44],
    ]),
    "Activist heavy": np.array([
        [0.08, 0.66], [0.18, 0.58], [0.32, 0.48],
        [0.68, 0.52], [0.82, 0.42], [0.92, 0.34],
    ]),
    "Asymmetric insurgency": np.array([
        [0.24, 0.58], [0.36, 0.52], [0.46, 0.50],
        [0.62, 0.52], [0.82, 0.46], [0.94, 0.30],
    ]),
}


class FixedMaskTurnoutStrategy(StrategyModel):
    """Use a pre-computed turnout mask while keeping ballot expression sincere."""

    def __init__(self, active_mask):
        self.active_mask = np.asarray(active_mask, dtype=bool)

    @property
    def name(self) -> str:
        return "Fixed mask turnout"

    def generate_ballots(
        self,
        electorate,
        candidates,
        approval_threshold=None,
        context=None,
    ):
        sincere = BallotProfile.from_preferences(
            electorate,
            candidates,
            approval_threshold=approval_threshold,
        )
        if len(self.active_mask) != sincere.n_voters:
            raise ValueError("Turnout mask does not match the electorate size.")
        return BallotProfile(
            plurality=sincere.plurality,
            rankings=sincere.rankings,
            scores=sincere.scores,
            approvals=sincere.approvals,
            distances=sincere.distances,
            approval_threshold=sincere.approval_threshold,
            n_voters=sincere.n_voters,
            n_candidates=sincere.n_candidates,
            active_voter_mask=self.active_mask,
        )


def build_voter_profile(name, seed, dim_names, n_voters=1400):
    rng = np.random.default_rng(seed)
    return gaussian_mixture_electorate(
        n_voters,
        VOTER_SPECS[name],
        rng=rng,
        dim_names=dim_names,
    )


def build_candidate_profile(name):
    labels = ["L-Base", "L-Bridge", "L-Moderate", "R-Moderate", "R-Bridge", "R-Base"]
    if name == "Asymmetric insurgency":
        labels[-1] = "R-Insurgent"
    return CandidateSet(CANDIDATE_SPECS[name].copy(), labels)


def build_parties():
    return [
        PartySpec("Left Party", [0, 1, 2], Plurality()),
        PartySpec("Right Party", [3, 4, 5], Plurality()),
    ]


def voter_side_masks(electorate, candidates):
    distances = np.linalg.norm(
        electorate.preferences[:, None, :] - candidates.positions[None, :, :],
        axis=2,
    )
    nearest = distances.argmin(axis=1)
    left_mask = nearest <= 2
    right_mask = ~left_mask
    x_coord = electorate.preferences[:, 0]
    center_distance = np.abs(x_coord - 0.5)
    return left_mask, right_mask, x_coord, center_distance


def turnout_setup(name, electorate, candidates, seed):
    rng = np.random.default_rng(seed)
    left_mask, right_mask, x_coord, center_distance = voter_side_masks(electorate, candidates)

    memberships = {
        "Left Party": left_mask.copy(),
        "Right Party": right_mask.copy(),
    }
    open_primary_strategy = None
    general_strategy = None

    if name == "Even participation":
        pass
    elif name == "Activist primary":
        memberships = {
            "Left Party": left_mask & (center_distance >= np.quantile(center_distance[left_mask], 0.45)),
            "Right Party": right_mask & (center_distance >= np.quantile(center_distance[right_mask], 0.45)),
        }
        open_mask = center_distance >= np.quantile(center_distance, 0.45)
        open_primary_strategy = FixedMaskTurnoutStrategy(open_mask)
    elif name == "Moderate primary":
        memberships = {
            "Left Party": left_mask & (center_distance <= np.quantile(center_distance[left_mask], 0.70)),
            "Right Party": right_mask & (center_distance <= np.quantile(center_distance[right_mask], 0.70)),
        }
        open_mask = center_distance <= np.quantile(center_distance, 0.70)
        open_primary_strategy = FixedMaskTurnoutStrategy(open_mask)
    elif name == "Right surge":
        memberships = {
            "Left Party": left_mask & (center_distance <= np.quantile(center_distance[left_mask], 0.75)),
            "Right Party": right_mask & (
                (center_distance >= np.quantile(center_distance[right_mask], 0.35))
                | (x_coord > 0.78)
            ),
        }
        open_mask = (x_coord > 0.58) | (center_distance >= np.quantile(center_distance, 0.60))
        open_primary_strategy = FixedMaskTurnoutStrategy(open_mask)
        general_mask = rng.random(electorate.n_voters) < np.where(x_coord > 0.5, 0.93, 0.75)
        general_strategy = FixedMaskTurnoutStrategy(general_mask)
    else:
        raise ValueError(f"Unknown turnout profile: {name}")

    return {
        "memberships": memberships,
        "open_primary_strategy": open_primary_strategy,
        "general_strategy": general_strategy,
    }


def classify_result(delta, tolerance):
    if delta < -tolerance:
        return "helps moderation"
    if delta > tolerance:
        return "backfires"
    return "roughly neutral"


def clean_result_row(
    result,
    pipeline,
    voter_name,
    candidate_name,
    turnout_name,
    trial,
    tolerance,
):
    if pipeline == "Top-4 open IRV":
        selection_divergence = result.finalist_divergence
    else:
        selection_divergence = result.primary_divergence

    delta = result.general_metrics.distance_to_median - result.baseline_metrics.distance_to_median
    return {
        "voter_profile": voter_name,
        "candidate_profile": candidate_name,
        "turnout_profile": turnout_name,
        "pipeline": pipeline,
        "trial": trial,
        "winner": result.general_result.winner_indices[0],
        "distance_to_median": result.general_metrics.distance_to_median,
        "baseline_distance_to_median": result.baseline_metrics.distance_to_median,
        "distance_to_mean": result.general_metrics.distance_to_mean,
        "distance_to_median_delta": delta,
        "mean_voter_distance": result.general_metrics.mean_voter_distance,
        "majority_satisfaction": result.general_metrics.majority_satisfaction,
        "selection_divergence": selection_divergence,
        "verdict": classify_result(delta, tolerance),
    }


def run_pipeline_suite(voter_name, candidate_name, turnout_name, seed, dim_names, tolerance):
    electorate = build_voter_profile(voter_name, seed, dim_names=dim_names)
    candidates = build_candidate_profile(candidate_name)
    parties = build_parties()
    turnout = turnout_setup(turnout_name, electorate, candidates, seed + 1000)

    rows = []
    rows.append(
        clean_result_row(
            run_two_party_primary(
                electorate,
                candidates,
                parties,
                Plurality(),
                primary_type=PrimaryType.CLOSED,
                memberships=turnout["memberships"],
                general_strategy=turnout["general_strategy"],
            ),
            "Closed plurality",
            voter_name,
            candidate_name,
            turnout_name,
            trial=seed,
            tolerance=tolerance,
        )
    )
    rows.append(
        clean_result_row(
            run_two_party_primary(
                electorate,
                candidates,
                parties,
                Plurality(),
                primary_type=PrimaryType.SEMI,
                general_strategy=turnout["general_strategy"],
            ),
            "Semi plurality",
            voter_name,
            candidate_name,
            turnout_name,
            trial=seed,
            tolerance=tolerance,
        )
    )
    rows.append(
        clean_result_row(
            run_two_party_primary(
                electorate,
                candidates,
                parties,
                InstantRunoff(),
                primary_type=PrimaryType.CLOSED,
                memberships=turnout["memberships"],
                general_strategy=turnout["general_strategy"],
            ),
            "Closed IRV",
            voter_name,
            candidate_name,
            turnout_name,
            trial=seed,
            tolerance=tolerance,
        )
    )
    rows.append(
        clean_result_row(
            run_open_primary_top_k(
                electorate,
                candidates,
                InstantRunoff(),
                top_k=4,
                primary_strategy=turnout["open_primary_strategy"],
                general_strategy=turnout["general_strategy"],
            ),
            "Top-4 open IRV",
            voter_name,
            candidate_name,
            turnout_name,
            trial=seed,
            tolerance=tolerance,
        )
    )
    return rows


def run_full_grid(n_trials, base_seed, dim_names, tolerance):
    rows = []
    draw_seed = base_seed
    for voter_name in VOTER_ORDER:
        for candidate_name in CANDIDATE_ORDER:
            for turnout_name in TURNOUT_ORDER:
                for trial in range(n_trials):
                    draw_seed += 1
                    for row in run_pipeline_suite(
                        voter_name,
                        candidate_name,
                        turnout_name,
                        draw_seed,
                        dim_names=dim_names,
                        tolerance=tolerance,
                    ):
                        row["trial"] = trial
                        rows.append(row)
    return pd.DataFrame(rows)


def summarize_full_grid(full_mc_df, tolerance):
    summary = (
        full_mc_df.groupby(
            [
                "voter_profile",
                "candidate_profile",
                "turnout_profile",
                "pipeline",
            ],
            observed=False,
        )
        .agg(
            mean_distance_to_median=("distance_to_median", "mean"),
            mean_baseline_distance=("baseline_distance_to_median", "mean"),
            mean_distance_to_median_delta=("distance_to_median_delta", "mean"),
            help_rate=("distance_to_median_delta", lambda s: float((s < -tolerance).mean())),
            backfire_rate=("distance_to_median_delta", lambda s: float((s > tolerance).mean())),
            neutral_rate=("distance_to_median_delta", lambda s: float((np.abs(s) <= tolerance).mean())),
            mean_divergence=("selection_divergence", "mean"),
            mean_majority_satisfaction=("majority_satisfaction", "mean"),
        )
        .reset_index()
    )

    for column, order in [
        ("voter_profile", VOTER_ORDER),
        ("candidate_profile", CANDIDATE_ORDER),
        ("turnout_profile", TURNOUT_ORDER),
        ("pipeline", PIPELINE_ORDER),
    ]:
        summary[column] = pd.Categorical(summary[column], categories=order, ordered=True)

    summary = summary.sort_values(
        [
            "voter_profile",
            "candidate_profile",
            "turnout_profile",
            "pipeline",
        ]
    ).reset_index(drop=True)

    summary["mean_verdict"] = summary["mean_distance_to_median_delta"].map(
        lambda delta: classify_result(delta, tolerance)
    )
    return summary


def pipeline_overview(summary):
    overview = (
        summary.groupby("pipeline", observed=False)[
            [
                "mean_distance_to_median_delta",
                "help_rate",
                "backfire_rate",
                "neutral_rate",
                "mean_divergence",
            ]
        ]
        .mean()
        .reindex(PIPELINE_ORDER)
    )
    return overview.reset_index()


def plot_turnout_snapshot(ax, voter_name, candidate_name, turnout_name, seed, dim_names):
    electorate = build_voter_profile(voter_name, seed, dim_names=dim_names)
    candidates = build_candidate_profile(candidate_name)
    turnout = turnout_setup(turnout_name, electorate, candidates, seed + 500)

    left_active = turnout["memberships"]["Left Party"]
    right_active = turnout["memberships"]["Right Party"]
    inactive = ~(left_active | right_active)

    ax.scatter(
        electorate.preferences[inactive, 0],
        electorate.preferences[inactive, 1],
        s=5,
        alpha=0.12,
        color="0.6",
    )
    ax.scatter(
        electorate.preferences[left_active, 0],
        electorate.preferences[left_active, 1],
        s=8,
        alpha=0.40,
        color="#4361ee",
    )
    ax.scatter(
        electorate.preferences[right_active, 0],
        electorate.preferences[right_active, 1],
        s=8,
        alpha=0.40,
        color="#e63946",
    )
    ax.scatter(
        candidates.positions[:3, 0],
        candidates.positions[:3, 1],
        s=65,
        color="#1d4ed8",
        edgecolors="black",
        linewidths=0.6,
    )
    ax.scatter(
        candidates.positions[3:, 0],
        candidates.positions[3:, 1],
        s=65,
        color="#dc2626",
        edgecolors="black",
        linewidths=0.6,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(DISPLAY_LABELS[turnout_name])
    ax.set_xlabel(dim_names[0])
    ax.set_ylabel(dim_names[1])


def plot_metric_small_multiples(summary, metric, title, cmap, center=None, vmin=None, vmax=None, fmt=".2f"):
    fig, axes = plt.subplots(
        len(PIPELINE_ORDER),
        len(TURNOUT_ORDER),
        figsize=(18, 14),
        dpi=150,
    )

    for row_idx, pipeline in enumerate(PIPELINE_ORDER):
        for col_idx, turnout_name in enumerate(TURNOUT_ORDER):
            ax = axes[row_idx, col_idx]
            subset = summary[
                (summary["pipeline"] == pipeline)
                & (summary["turnout_profile"] == turnout_name)
            ].copy()
            subset["voter_display"] = subset["voter_profile"].map(DISPLAY_LABELS)
            subset["candidate_display"] = subset["candidate_profile"].map(DISPLAY_LABELS)
            pivot = subset.pivot(
                index="voter_display",
                columns="candidate_display",
                values=metric,
            )
            sns.heatmap(
                pivot,
                annot=True,
                fmt=fmt,
                cmap=cmap,
                center=center,
                vmin=vmin,
                vmax=vmax,
                cbar=(row_idx == 0 and col_idx == len(TURNOUT_ORDER) - 1),
                ax=ax,
            )
            if row_idx == 0:
                ax.set_title(DISPLAY_LABELS[turnout_name])
            if col_idx == 0:
                ax.set_ylabel(pipeline)
            else:
                ax.set_ylabel("")
            if row_idx == len(PIPELINE_ORDER) - 1:
                ax.set_xlabel("candidate profile")
            else:
                ax.set_xlabel("")

    fig.suptitle(title, y=1.01, fontsize=14)
    fig.tight_layout()
    return fig


def representative_state(voter_name, candidate_name, turnout_name, seed, dim_names):
    electorate = build_voter_profile(voter_name, seed, dim_names=dim_names)
    candidates = build_candidate_profile(candidate_name)
    turnout = turnout_setup(turnout_name, electorate, candidates, seed + 500)
    return electorate, candidates, turnout


def run_deep_dive(voter_name, candidate_name, turnout_name, n_trials, base_seed, dim_names, tolerance):
    rows = []
    draw_seed = base_seed
    for trial in range(n_trials):
        draw_seed += 1
        for row in run_pipeline_suite(
            voter_name,
            candidate_name,
            turnout_name,
            draw_seed,
            dim_names=dim_names,
            tolerance=tolerance,
        ):
            row["trial"] = trial
            rows.append(row)
    mc_df = pd.DataFrame(rows)
    mc_df["pipeline"] = pd.Categorical(mc_df["pipeline"], categories=PIPELINE_ORDER, ordered=True)
    return mc_df


def plot_deep_dive(case, n_trials, seed, dim_names, tolerance):
    voter_name = case["voter_profile"]
    candidate_name = case["candidate_profile"]
    turnout_name = case["turnout_profile"]
    headline = case["headline"]
    takeaway = case["takeaway"]

    display(Markdown(f"### {headline}"))
    display(Markdown(takeaway))

    electorate, candidates, turnout = representative_state(
        voter_name,
        candidate_name,
        turnout_name,
        seed,
        dim_names=dim_names,
    )
    mc_df = run_deep_dive(
        voter_name,
        candidate_name,
        turnout_name,
        n_trials=n_trials,
        base_seed=seed,
        dim_names=dim_names,
        tolerance=tolerance,
    )
    mc_summary = (
        mc_df.groupby("pipeline", observed=False)
        .agg(
            mean_delta=("distance_to_median_delta", "mean"),
            help_rate=("distance_to_median_delta", lambda s: float((s < -tolerance).mean())),
            backfire_rate=("distance_to_median_delta", lambda s: float((s > tolerance).mean())),
        )
        .reindex(PIPELINE_ORDER)
        .reset_index()
    )

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), dpi=150)
    plot_electorate(
        electorate,
        candidates,
        title=(
            f"{DISPLAY_LABELS[voter_name].replace(chr(10), ' ')} + "
            f"{DISPLAY_LABELS[candidate_name].replace(chr(10), ' ')}"
        ),
        ax=axes[0],
    )
    if axes[0].legend_ is not None:
        axes[0].legend_.remove()

    left_active = turnout["memberships"]["Left Party"]
    right_active = turnout["memberships"]["Right Party"]
    inactive = ~(left_active | right_active)
    axes[1].scatter(
        electorate.preferences[inactive, 0],
        electorate.preferences[inactive, 1],
        s=5,
        alpha=0.12,
        color="0.6",
    )
    axes[1].scatter(
        electorate.preferences[left_active, 0],
        electorate.preferences[left_active, 1],
        s=8,
        alpha=0.40,
        color="#4361ee",
    )
    axes[1].scatter(
        electorate.preferences[right_active, 0],
        electorate.preferences[right_active, 1],
        s=8,
        alpha=0.40,
        color="#e63946",
    )
    axes[1].scatter(
        candidates.positions[:3, 0],
        candidates.positions[:3, 1],
        s=70,
        color="#1d4ed8",
        edgecolors="black",
        linewidths=0.6,
    )
    axes[1].scatter(
        candidates.positions[3:, 0],
        candidates.positions[3:, 1],
        s=70,
        color="#dc2626",
        edgecolors="black",
        linewidths=0.6,
    )
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)
    axes[1].set_title(f"Primary participants: {turnout_name}")
    axes[1].set_xlabel(dim_names[0])
    axes[1].set_ylabel(dim_names[1])

    sns.violinplot(
        data=mc_df,
        x="pipeline",
        y="distance_to_median_delta",
        order=PIPELINE_ORDER,
        inner="box",
        cut=0,
        ax=axes[2],
    )
    axes[2].axhline(0.0, color="black", linestyle=":", linewidth=1)
    axes[2].set_title("Monte Carlo distribution of moderation shift")
    axes[2].set_xlabel("")
    axes[2].set_ylabel("distance_to_median_delta")
    axes[2].tick_params(axis="x", rotation=25)
    fig.tight_layout()

    display(mc_summary)
    del mc_df
    gc.collect()


__all__ = [
    "VOTER_ORDER",
    "CANDIDATE_ORDER",
    "TURNOUT_ORDER",
    "PIPELINE_ORDER",
    "DISPLAY_LABELS",
    "build_voter_profile",
    "build_candidate_profile",
    "run_full_grid",
    "summarize_full_grid",
    "pipeline_overview",
    "plot_turnout_snapshot",
    "plot_metric_small_multiples",
    "plot_deep_dive",
]
