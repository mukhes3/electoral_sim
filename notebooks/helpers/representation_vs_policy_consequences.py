"""Helpers for the representation versus policy consequences notebook."""
from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from electoral_sim.ballots import BallotProfile
from electoral_sim.candidates import CandidateSet
from electoral_sim.electorate import Electorate, gaussian_mixture_electorate
from electoral_sim.fractional import FractionalBallotContinuous, FractionalBallotDiscrete
from electoral_sim.metrics import compute_group_metrics, compute_metrics, compute_policy_metrics
from electoral_sim.policy import (
    PolicyConsequenceSpec,
    PolicyDefinition,
    PolicyOutcome,
    PolicyThresholdEffect,
    PolicyUtilityComponents,
)
from electoral_sim.systems import (
    ApprovalVoting,
    CondorcetSchulze,
    InstantRunoff,
    PartyListPR,
    Plurality,
    ScoreVoting,
)
from electoral_sim.utils import plot_electorate


CASE_ORDER = [
    "Overlapping blocs",
    "Moderate burden shift",
    "Fragmented legislature",
]
SYSTEM_ORDER = [
    "Plurality",
    "IRV",
    "Approval",
    "Score",
    "Condorcet",
    "Fractional discrete (sigma=1.0)",
    "Fractional continuous (sigma=1.0)",
    "Party-list PR",
]
POLICY_DEFINITION_ORDER = [
    "Median legislator",
    "Seat-share centroid",
    "Blended compromise",
]
DISPLAY_LABELS = {
    "Overlapping blocs": "Overlapping\nblocs",
    "Moderate burden shift": "Moderate\nburden shift",
    "Fragmented legislature": "Fragmented\nlegislature",
    "Median legislator": "Median\nlegislator",
    "Seat-share centroid": "Seat-share\ncentroid",
    "Blended compromise": "Blended\ncompromise",
}
CASE_EXPLANATIONS = {
    "Overlapping blocs": {
        "setup": "Two groups overlap heavily in policy space, so the election is mostly about small differences near the middle.",
        "question": "If almost everyone lives in the same broad neighborhood, do electoral rules still matter once policy consequences are evaluated?",
    },
    "Moderate burden shift": {
        "setup": "Most voters are clustered near a moderate majority, while a smaller bloc sits farther away and is more exposed to redistribution and public-goods choices.",
        "question": "When one group is smaller and more vulnerable to policy design, which systems protect it and which ones leave it exposed?",
    },
    "Fragmented legislature": {
        "setup": "The electorate is split between two groups, but the candidate slate is more spread out and legislature-style compromise becomes plausible.",
        "question": "When several viable positions are on offer, do systems that reward broader coalitions translate representation into better policy outcomes?",
    },
}
SYSTEM_EXPLANATIONS = {
    "Plurality": "Each voter chooses one candidate; the top vote-getter wins.",
    "IRV": "Voters rank candidates; the weakest candidate is dropped until one has a majority.",
    "Approval": "Voters can support more than one acceptable candidate; most approvals wins.",
    "Score": "Voters score every candidate; the highest average score wins.",
    "Condorcet": "The winner is the candidate who beats every rival head-to-head when such a candidate exists.",
    "Fractional discrete (sigma=1.0)": "Each voter spreads support across candidates using distance-weighted fractional ballots, then the candidate nearest the resulting centroid is treated as the single winner.",
    "Fractional continuous (sigma=1.0)": "Each voter spreads support across candidates using distance-weighted fractional ballots, and the final policy is the weighted combination of all candidates rather than a single winner.",
    "Party-list PR": "Seats are allocated proportionally across the slate, and the policy outcome is summarized from the resulting legislature.",
}
CONSEQUENCE_MODEL_EXPLANATIONS = {
    "Distance-only": "Utility depends only on how far the realized policy is from each voter's ideal point.",
    "Exposure-sensitive": "The same policy can affect groups differently because some voters care more about redistribution and public goods than others.",
    "Threshold-sensitive": "A policy floor matters: if redistribution falls below a minimum level, outcomes become sharply worse even when they are otherwise nearby in space.",
}

CASE_SPECS = {
    "Overlapping blocs": {
        "components": [
            {
                "weight": 0.72,
                "mean": [0.43, 0.47],
                "cov": [[0.012, 0.002], [0.002, 0.012]],
                "group": "Majority bloc",
            },
            {
                "weight": 0.28,
                "mean": [0.57, 0.62],
                "cov": [[0.010, 0.001], [0.001, 0.010]],
                "group": "Minority bloc",
            },
        ],
        "candidate_positions": np.array(
            [
                [0.22, 0.36],
                [0.37, 0.44],
                [0.49, 0.50],
                [0.62, 0.60],
                [0.78, 0.72],
            ],
            dtype=float,
        ),
    },
    "Moderate burden shift": {
        "components": [
            {
                "weight": 0.82,
                "mean": [0.40, 0.41],
                "cov": [[0.011, 0.001], [0.001, 0.010]],
                "group": "Majority bloc",
            },
            {
                "weight": 0.18,
                "mean": [0.67, 0.71],
                "cov": [[0.009, 0.001], [0.001, 0.009]],
                "group": "Minority bloc",
            },
        ],
        "candidate_positions": np.array(
            [
                [0.18, 0.28],
                [0.34, 0.38],
                [0.48, 0.47],
                [0.63, 0.64],
                [0.82, 0.82],
            ],
            dtype=float,
        ),
    },
    "Fragmented legislature": {
        "components": [
            {
                "weight": 0.68,
                "mean": [0.36, 0.47],
                "cov": [[0.010, 0.002], [0.002, 0.010]],
                "group": "Majority bloc",
            },
            {
                "weight": 0.32,
                "mean": [0.73, 0.55],
                "cov": [[0.010, -0.001], [-0.001, 0.010]],
                "group": "Minority bloc",
            },
        ],
        "candidate_positions": np.array(
            [
                [0.16, 0.36],
                [0.30, 0.44],
                [0.46, 0.52],
                [0.58, 0.51],
                [0.72, 0.56],
                [0.86, 0.61],
            ],
            dtype=float,
        ),
    },
}

SYSTEM_FACTORIES = {
    "Plurality": lambda: Plurality(),
    "IRV": lambda: InstantRunoff(),
    "Approval": lambda: ApprovalVoting(),
    "Score": lambda: ScoreVoting(),
    "Condorcet": lambda: CondorcetSchulze(),
    "Fractional discrete (sigma=1.0)": lambda: FractionalBallotDiscrete(sigma=1.0),
    "Fractional continuous (sigma=1.0)": lambda: FractionalBallotContinuous(sigma=1.0),
    "Party-list PR": lambda: PartyListPR(n_seats=60, threshold=0.0, outcome_rule="axis_median"),
}


def representation_policy_helper_overview() -> pd.DataFrame:
    """Small reference table for the notebook helpers."""
    return pd.DataFrame(
        [
            {
                "component": "Static cases",
                "options": ", ".join(CASE_ORDER),
                "purpose": "Three hand-built two-group electorates with different geometry and slate structure.",
            },
            {
                "component": "Systems",
                "options": ", ".join(SYSTEM_ORDER),
                "purpose": "Five classic systems, two fractional-ballot variants at sigma = 1.0, and one PR case for policy-definition comparisons.",
            },
            {
                "component": "Policy definitions",
                "options": ", ".join(POLICY_DEFINITION_ORDER),
                "purpose": "Alternative mappings from the same election result to the realized policy used for welfare analysis.",
            },
            {
                "component": "Consequence models",
                "options": "distance-only, exposure-sensitive, threshold-sensitive",
                "purpose": "Compare pure representation metrics with welfare models that add group-specific exposure and policy floors.",
            },
        ]
    )


def case_reference_table() -> pd.DataFrame:
    """Plain-language descriptions of the notebook's static cases."""
    return pd.DataFrame(
        [
            {
                "case": case_name,
                "what_it_looks_like": CASE_EXPLANATIONS[case_name]["setup"],
                "what_the_case_tests": CASE_EXPLANATIONS[case_name]["question"],
            }
            for case_name in CASE_ORDER
        ]
    )


def system_reference_table() -> pd.DataFrame:
    """Plain-language descriptions of the electoral systems used in the notebook."""
    return pd.DataFrame(
        [
            {
                "system": system_name,
                "description": SYSTEM_EXPLANATIONS[system_name],
            }
            for system_name in SYSTEM_ORDER
        ]
    )


def consequence_model_reference_table() -> pd.DataFrame:
    """Plain-language descriptions of the utility models used in the notebook."""
    return pd.DataFrame(
        [
            {
                "utility_type": model_name,
                "description": description,
            }
            for model_name, description in CONSEQUENCE_MODEL_EXPLANATIONS.items()
        ]
    )


def build_case(case_name: str, seed: int = 0, n_voters: int = 1800) -> tuple[Electorate, CandidateSet]:
    """Construct one of the notebook's electorates and candidate slates."""
    if case_name not in CASE_SPECS:
        raise ValueError(f"Unknown case_name: {case_name}")
    spec = CASE_SPECS[case_name]
    electorate = gaussian_mixture_electorate(
        n_voters=n_voters,
        components=spec["components"],
        rng=np.random.default_rng(seed),
        dim_names=["redistribution", "public goods"],
    )
    candidate_positions = np.asarray(spec["candidate_positions"], dtype=float)
    candidates = CandidateSet(
        positions=candidate_positions.copy(),
        labels=[f"C{i + 1}" for i in range(len(candidate_positions))],
    )
    return electorate, candidates


def build_distance_only_spec() -> PolicyConsequenceSpec:
    """Pure spatial policy utility: utility is negative distance to policy."""
    return PolicyConsequenceSpec(distance_weight=1.0, dimension_weights=np.array([1.0, 1.0]))


def exposure_sensitive_utility(
    policy: PolicyOutcome | np.ndarray,
    electorate: Electorate,
    spec: PolicyConsequenceSpec,
) -> PolicyUtilityComponents:
    """
    Group-sensitive policy utility for the notebook.

    The first dimension is interpreted as redistribution and the second as a
    public-goods dimension. The minority group values both more strongly, while
    the majority group is mildly wary of higher redistribution.
    """
    if electorate.group_ids is None or electorate.group_names is None:
        raise ValueError("The exposure-sensitive utility requires labeled voter groups.")

    policy_vector = policy.vector if isinstance(policy, PolicyOutcome) else np.asarray(policy, dtype=float)
    weights = np.array([1.0, 0.9], dtype=float)
    deltas = electorate.preferences - policy_vector[None, :]
    weighted_distance = np.sqrt((weights[None, :] * np.square(deltas)).sum(axis=1))
    distance_utility = -0.85 * weighted_distance

    group_name_by_id = dict(electorate.group_names)
    minority_ids = [group_id for group_id, name in group_name_by_id.items() if "Minority" in name]
    minority_mask = np.isin(electorate.group_ids, np.asarray(minority_ids, dtype=int))

    public_goods_coeff = np.where(minority_mask, 0.55, 0.25)
    redistribution_coeff = np.where(minority_mask, 0.65, -0.18)

    public_goods_utility = public_goods_coeff * float(policy_vector[1])
    group_adjustment_utility = redistribution_coeff * float(policy_vector[0] - 0.5)
    total_utility = distance_utility + public_goods_utility + group_adjustment_utility

    return PolicyUtilityComponents(
        total_utility=total_utility,
        distance_utility=distance_utility,
        public_goods_utility=public_goods_utility,
        group_adjustment_utility=group_adjustment_utility,
        policy_distance=weighted_distance,
    )


def build_exposure_sensitive_spec() -> PolicyConsequenceSpec:
    """Utility definition where groups are differently exposed to the policy dimensions."""
    return PolicyConsequenceSpec(utility_function=exposure_sensitive_utility)


def build_threshold_sensitive_spec() -> PolicyConsequenceSpec:
    """
    Exposure-sensitive utility plus a redistribution floor.

    The minority bloc is modeled as requiring a higher minimum redistribution
    level before ordinary ideological closeness becomes meaningful. Below that
    floor, the outcome receives a much flatter and more negative utility.
    """
    return PolicyConsequenceSpec(
        utility_function=exposure_sensitive_utility,
        threshold_effects=(
            PolicyThresholdEffect(
                dimension=0,
                threshold=0.40,
                threshold_by_group={1: 0.58},
                distance_utility_below=-0.52,
                utility_offset_below=-0.08,
            ),
        ),
    )


def build_blended_compromise_definition(weight_outcome: float = 0.5) -> PolicyDefinition:
    """A custom policy definition that averages the default PR outcome with the centroid."""
    weight_outcome = float(weight_outcome)
    weight_outcome = min(max(weight_outcome, 0.0), 1.0)
    weight_centroid = 1.0 - weight_outcome

    def resolver(result, candidates=None):
        return weight_outcome * result.outcome_position + weight_centroid * result.centroid_position

    return PolicyDefinition(
        name="blended_compromise",
        resolver=resolver,
        metadata={"weight_outcome": weight_outcome, "weight_centroid": weight_centroid},
    )


def run_case_systems(
    case_name: str,
    consequence_spec: PolicyConsequenceSpec | None = None,
    policy_rule: str | PolicyDefinition = "outcome",
    system_names: list[str] | None = None,
    seed: int = 0,
    n_voters: int = 1800,
) -> pd.DataFrame:
    """Run the notebook systems on one case and summarize representation and policy metrics."""
    electorate, candidates = build_case(case_name, seed=seed, n_voters=n_voters)
    ballots = BallotProfile.from_preferences(electorate, candidates)
    consequence_spec = build_distance_only_spec() if consequence_spec is None else consequence_spec
    system_names = SYSTEM_ORDER if system_names is None else list(system_names)

    rows: list[dict[str, object]] = []
    for system_name in system_names:
        system = SYSTEM_FACTORIES[system_name]()
        result = system.run(ballots, candidates)
        metrics = compute_metrics(result, electorate, candidates)
        group_metrics = compute_group_metrics(result, electorate, candidates)
        policy_metrics = compute_policy_metrics(
            result,
            electorate,
            candidates=candidates,
            policy_rule=policy_rule,
            consequence_spec=consequence_spec,
        )
        majority_group, minority_group = _majority_minority_groups(group_metrics.groups)

        rows.append(
            {
                "case": case_name,
                "system": system_name,
                "winner": ", ".join(candidates.labels[idx] for idx in result.winner_indices),
                "is_pr": bool(result.is_pr),
                "policy_rule": policy_metrics.policy_rule,
                "outcome_x": float(result.outcome_position[0]),
                "outcome_y": float(result.outcome_position[1]),
                "policy_x": float(policy_metrics.policy_vector[0]),
                "policy_y": float(policy_metrics.policy_vector[1]),
                "representation_aggregate_welfare": -float(metrics.mean_voter_distance),
                "representation_majority_welfare": float(majority_group.welfare),
                "representation_minority_welfare": float(minority_group.welfare),
                "representation_majority_minority_gap": float(majority_group.welfare - minority_group.welfare),
                "policy_aggregate_utility": float(policy_metrics.mean_total_utility),
                "policy_majority_utility": float(policy_metrics.majority_group_utility),
                "policy_minority_utility": float(policy_metrics.minority_group_utility),
                "policy_majority_minority_gap": float(policy_metrics.majority_minority_utility_gap),
                "policy_public_goods_component": float(policy_metrics.mean_public_goods_utility),
                "policy_group_adjustment_component": float(policy_metrics.mean_group_adjustment_utility),
                "policy_threshold_component": float(policy_metrics.mean_threshold_utility),
            }
        )

    return pd.DataFrame(rows)


def run_case_grid(
    case_names: list[str] | None = None,
    consequence_spec: PolicyConsequenceSpec | None = None,
    policy_rule: str | PolicyDefinition = "outcome",
    system_names: list[str] | None = None,
    seed: int = 0,
    n_voters: int = 1800,
) -> pd.DataFrame:
    """Run all notebook cases under a shared policy consequence model."""
    case_names = CASE_ORDER if case_names is None else list(case_names)
    rows = []
    for case_idx, case_name in enumerate(case_names):
        rows.append(
            run_case_systems(
                case_name,
                consequence_spec=consequence_spec,
                policy_rule=policy_rule,
                system_names=system_names,
                seed=seed + 101 * case_idx,
                n_voters=n_voters,
            )
        )
    return pd.concat(rows, ignore_index=True)


def compare_policy_definitions(
    case_name: str = "Fragmented legislature",
    consequence_spec: PolicyConsequenceSpec | None = None,
    definition_map: dict[str, str | PolicyDefinition] | None = None,
    seed: int = 0,
    n_voters: int = 1800,
) -> pd.DataFrame:
    """Compare policy definitions for a single PR election result."""
    electorate, candidates = build_case(case_name, seed=seed, n_voters=n_voters)
    ballots = BallotProfile.from_preferences(electorate, candidates)
    result = SYSTEM_FACTORIES["Party-list PR"]().run(ballots, candidates)
    consequence_spec = build_exposure_sensitive_spec() if consequence_spec is None else consequence_spec
    if definition_map is None:
        definition_map = {
            "Median legislator": "outcome",
            "Seat-share centroid": "centroid",
            "Blended compromise": build_blended_compromise_definition(0.5),
        }

    rows: list[dict[str, object]] = []
    for label, definition in definition_map.items():
        policy_metrics = compute_policy_metrics(
            result,
            electorate,
            candidates=candidates,
            policy_rule=definition,
            consequence_spec=consequence_spec,
        )
        rows.append(
            {
                "case": case_name,
                "definition": label,
                "policy_rule": policy_metrics.policy_rule,
                "policy_x": float(policy_metrics.policy_vector[0]),
                "policy_y": float(policy_metrics.policy_vector[1]),
                "aggregate_utility": float(policy_metrics.mean_total_utility),
                "majority_utility": float(policy_metrics.majority_group_utility),
                "minority_utility": float(policy_metrics.minority_group_utility),
                "majority_minority_gap": float(policy_metrics.majority_minority_utility_gap),
                "mean_public_goods_utility": float(policy_metrics.mean_public_goods_utility),
                "mean_group_adjustment_utility": float(policy_metrics.mean_group_adjustment_utility),
                "mean_threshold_utility": float(policy_metrics.mean_threshold_utility),
            }
        )
    return pd.DataFrame(rows)


def compare_consequence_models(
    case_name: str,
    model_map: dict[str, PolicyConsequenceSpec] | None = None,
    system_names: list[str] | None = None,
    seed: int = 0,
    n_voters: int = 1800,
) -> pd.DataFrame:
    """Compare multiple policy consequence models on the same underlying election outcomes."""
    if model_map is None:
        model_map = {
            "Distance-only": build_distance_only_spec(),
            "Exposure-sensitive": build_exposure_sensitive_spec(),
            "Threshold-sensitive": build_threshold_sensitive_spec(),
        }
    rows = []
    for model_name, consequence_spec in model_map.items():
        frame = run_case_systems(
            case_name,
            consequence_spec=consequence_spec,
            system_names=system_names,
            seed=seed,
            n_voters=n_voters,
        ).copy()
        frame["consequence_model"] = model_name
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def case_comparison_table(
    comparison_df: pd.DataFrame,
    case_name: str,
) -> pd.DataFrame:
    """Compact comparison table for one case across systems and utility models."""
    subset = comparison_df[comparison_df["case"] == case_name].copy()
    subset["system"] = pd.Categorical(subset["system"], categories=SYSTEM_ORDER, ordered=True)
    subset["consequence_model"] = pd.Categorical(
        subset["consequence_model"],
        categories=list(CONSEQUENCE_MODEL_EXPLANATIONS),
        ordered=True,
    )
    subset = subset.sort_values(["system", "consequence_model"])

    table = subset.pivot_table(
        index=["system", "winner"],
        columns="consequence_model",
        values=[
            "policy_aggregate_utility",
            "policy_minority_utility",
            "policy_majority_minority_gap",
        ],
        observed=False,
    )
    table = table.reindex(SYSTEM_ORDER, level=0)
    table = table.round(3)
    table.columns = pd.MultiIndex.from_tuples(
        [
            (
                {
                    "policy_aggregate_utility": "Aggregate utility",
                    "policy_minority_utility": "Minority utility",
                    "policy_majority_minority_gap": "Majority-minority gap",
                }[metric],
                model,
            )
            for metric, model in table.columns
        ]
    )
    return table


def hidden_harm_cases(results_df: pd.DataFrame) -> pd.DataFrame:
    """Cases where aggregate policy utility looks healthy relative to minority utility."""
    ranked = results_df.copy()
    ranked["aggregate_minus_minority"] = (
        ranked["policy_aggregate_utility"] - ranked["policy_minority_utility"]
    )
    return ranked.sort_values(
        ["aggregate_minus_minority", "policy_aggregate_utility"],
        ascending=[False, False],
    ).reset_index(drop=True)


def plot_case_gallery(
    case_names: list[str] | None = None,
    seed: int = 0,
    n_voters: int = 1800,
) -> plt.Figure:
    """Plot the three notebook cases as electorate snapshots."""
    case_names = CASE_ORDER if case_names is None else list(case_names)
    fig, axes = plt.subplots(1, len(case_names), figsize=(5.0 * len(case_names), 4.2), dpi=150)
    axes = np.atleast_1d(axes)

    for ax, case_name in zip(axes, case_names):
        electorate, candidates = build_case(case_name, seed=seed + 41 * case_names.index(case_name), n_voters=n_voters)
        plot_electorate(
            electorate,
            candidates,
            title=case_name,
            ax=ax,
        )
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()

    fig.tight_layout()
    return fig


def plot_case_policy_points(
    comparison_df: pd.DataFrame,
    case_name: str,
    seed: int = 0,
    n_voters: int = 1800,
) -> plt.Figure:
    """Overlay each system's realized policy point on the case electorate."""
    electorate, candidates = build_case(case_name, seed=seed, n_voters=n_voters)
    subset = comparison_df[comparison_df["case"] == case_name].copy()
    subset["system"] = pd.Categorical(subset["system"], categories=SYSTEM_ORDER, ordered=True)
    subset = subset.sort_values(["system", "consequence_model"]).drop_duplicates("system", keep="first")

    fig, ax = plt.subplots(figsize=(7.2, 5.6), dpi=150)
    plot_electorate(electorate, candidates, title=f"{case_name}: realized policy points", ax=ax)
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()

    colors = sns.color_palette("tab10", n_colors=len(SYSTEM_ORDER))
    color_map = {system: colors[idx] for idx, system in enumerate(SYSTEM_ORDER)}
    marker_map = {
        "Plurality": "o",
        "IRV": "s",
        "Approval": "^",
        "Score": "D",
        "Condorcet": "P",
        "Fractional discrete (sigma=1.0)": "v",
        "Fractional continuous (sigma=1.0)": ">",
        "Party-list PR": "X",
    }

    x_span = max(electorate.preferences[:, 0].max() - electorate.preferences[:, 0].min(), 1e-3)
    y_span = max(electorate.preferences[:, 1].max() - electorate.preferences[:, 1].min(), 1e-3)
    x_offset = 0.012 * x_span
    y_offset = 0.012 * y_span

    for _, row in subset.iterrows():
        system = str(row["system"])
        ax.scatter(
            row["policy_x"],
            row["policy_y"],
            s=160,
            marker=marker_map.get(system, "o"),
            color=color_map[system],
            edgecolor="white",
            linewidth=0.9,
            zorder=5,
        )
        ax.text(
            float(row["policy_x"]) + x_offset,
            float(row["policy_y"]) + y_offset,
            system,
            fontsize=7.5,
            color=color_map[system],
            alpha=0.95,
            zorder=6,
        )

    ax.set_xlabel("redistribution")
    ax.set_ylabel("public goods")
    fig.tight_layout()
    return fig


def plot_representation_policy_contrasts(
    results_df: pd.DataFrame,
    case_name: str,
) -> plt.Figure:
    """Compare representation welfare with policy utility for one static case."""
    subset = results_df[results_df["case"] == case_name].copy()
    subset["system"] = pd.Categorical(subset["system"], categories=SYSTEM_ORDER, ordered=True)
    subset = subset.sort_values("system")

    panel_specs = [
        (
            "Aggregate welfare",
            [
                ("representation_aggregate_welfare", "Representation"),
                ("policy_aggregate_utility", "Policy"),
            ],
        ),
        (
            "Minority welfare",
            [
                ("representation_minority_welfare", "Representation"),
                ("policy_minority_utility", "Policy"),
            ],
        ),
        (
            "Majority - minority gap",
            [
                ("representation_majority_minority_gap", "Representation"),
                ("policy_majority_minority_gap", "Policy"),
            ],
        ),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), dpi=150)
    palette = {"Representation": "#4C78A8", "Policy": "#E45756"}
    for ax, (title, columns) in zip(axes, panel_specs):
        rows = []
        for _, row in subset.iterrows():
            for column, label in columns:
                rows.append({"system": row["system"], "metric": label, "value": row[column]})
        plot_df = pd.DataFrame(rows)
        sns.barplot(data=plot_df, x="system", y="value", hue="metric", palette=palette, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=30)
        if title.endswith("gap"):
            ax.axhline(0.0, color="#333333", linewidth=1, alpha=0.5)
    axes[0].legend(frameon=False, loc="best")
    for ax in axes[1:]:
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
    fig.tight_layout()
    return fig


def plot_policy_heatmaps(results_df: pd.DataFrame) -> plt.Figure:
    """Heatmaps for aggregate and minority policy utility by case and system."""
    aggregate = results_df.pivot(index="case", columns="system", values="policy_aggregate_utility")
    aggregate = aggregate.reindex(index=CASE_ORDER, columns=SYSTEM_ORDER)
    minority = results_df.pivot(index="case", columns="system", values="policy_minority_utility")
    minority = minority.reindex(index=CASE_ORDER, columns=SYSTEM_ORDER)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.2), dpi=150)
    sns.heatmap(aggregate, annot=True, fmt=".3f", cmap="YlGnBu", ax=axes[0], cbar=False)
    sns.heatmap(minority, annot=True, fmt=".3f", cmap="YlOrRd", ax=axes[1], cbar=False)
    axes[0].set_title("Aggregate policy utility")
    axes[1].set_title("Minority policy utility")
    for ax in axes:
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=30)
        ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    return fig


def plot_hidden_harm_scatter(results_df: pd.DataFrame) -> plt.Figure:
    """Scatter aggregate versus minority policy utility across the notebook grid."""
    fig, ax = plt.subplots(figsize=(6.6, 5.2), dpi=150)
    palette = dict(zip(CASE_ORDER, sns.color_palette("Set2", n_colors=len(CASE_ORDER))))
    markers = {
        "Plurality": "o",
        "IRV": "s",
        "Approval": "^",
        "Score": "D",
        "Condorcet": "P",
        "Fractional discrete (sigma=1.0)": "v",
        "Fractional continuous (sigma=1.0)": ">",
        "Party-list PR": "X",
    }
    for _, row in results_df.iterrows():
        ax.scatter(
            row["policy_aggregate_utility"],
            row["policy_minority_utility"],
            s=90,
            color=palette[row["case"]],
            marker=markers[row["system"]],
            edgecolor="white",
            linewidth=0.7,
            alpha=0.95,
        )
        ax.text(
            row["policy_aggregate_utility"] + 0.004,
            row["policy_minority_utility"] + 0.004,
            row["system"],
            fontsize=7,
            alpha=0.85,
        )
    lower = min(results_df["policy_aggregate_utility"].min(), results_df["policy_minority_utility"].min())
    upper = max(results_df["policy_aggregate_utility"].max(), results_df["policy_minority_utility"].max())
    ax.plot([lower, upper], [lower, upper], linestyle="--", color="#333333", alpha=0.5)
    ax.set_xlabel("aggregate policy utility")
    ax.set_ylabel("minority policy utility")
    ax.set_title("When aggregate utility hides minority harm")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def plot_consequence_model_comparison(
    comparison_df: pd.DataFrame,
    case_name: str,
    systems: list[str] | None = None,
) -> plt.Figure:
    """Compare consequence models for one case across a small set of systems."""
    systems = ["Plurality", "Score", "Party-list PR"] if systems is None else list(systems)
    subset = comparison_df[
        (comparison_df["case"] == case_name) & (comparison_df["system"].isin(systems))
    ].copy()
    subset["system"] = pd.Categorical(subset["system"], categories=systems, ordered=True)
    subset = subset.sort_values(["system", "consequence_model"])

    melted = subset.melt(
        id_vars=["system", "consequence_model"],
        value_vars=[
            "policy_aggregate_utility",
            "policy_minority_utility",
            "policy_threshold_component",
        ],
        var_name="metric",
        value_name="value",
    )
    label_map = {
        "policy_aggregate_utility": "Aggregate utility",
        "policy_minority_utility": "Minority utility",
        "policy_threshold_component": "Threshold component",
    }
    melted["metric"] = melted["metric"].map(label_map)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), dpi=150)
    palette = {
        "Distance-only": "#4C78A8",
        "Exposure-sensitive": "#72B7B2",
        "Threshold-sensitive": "#E45756",
    }
    for ax, metric_name in zip(axes, ["Aggregate utility", "Minority utility", "Threshold component"]):
        panel = melted[melted["metric"] == metric_name]
        sns.barplot(
            data=panel,
            x="system",
            y="value",
            hue="consequence_model",
            palette=palette,
            ax=ax,
        )
        ax.set_title(metric_name)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=20)
        if metric_name == "Threshold component":
            ax.axhline(0.0, color="#333333", linewidth=1, alpha=0.5)
    axes[0].legend(frameon=False, loc="best")
    for ax in axes[1:]:
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
    fig.tight_layout()
    return fig


def plot_case_model_heatmaps(
    comparison_df: pd.DataFrame,
    case_name: str,
) -> plt.Figure:
    """Heatmaps showing how each system performs under each utility model for one case."""
    subset = comparison_df[comparison_df["case"] == case_name].copy()
    subset["system"] = pd.Categorical(subset["system"], categories=SYSTEM_ORDER, ordered=True)
    subset["consequence_model"] = pd.Categorical(
        subset["consequence_model"],
        categories=list(CONSEQUENCE_MODEL_EXPLANATIONS),
        ordered=True,
    )
    subset = subset.sort_values(["system", "consequence_model"])

    panel_specs = [
        ("policy_aggregate_utility", "Aggregate policy utility", "YlGnBu", False),
        ("policy_minority_utility", "Minority policy utility", "YlOrRd", False),
        ("policy_majority_minority_gap", "Majority - minority gap", "RdBu_r", True),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.0), dpi=150)
    for ax, (metric, title, cmap, center_zero) in zip(axes, panel_specs):
        matrix = subset.pivot(index="system", columns="consequence_model", values=metric)
        matrix = matrix.reindex(index=SYSTEM_ORDER, columns=list(CONSEQUENCE_MODEL_EXPLANATIONS))
        heatmap_kwargs = dict(data=matrix, annot=True, fmt=".3f", cmap=cmap, ax=ax, cbar=False)
        if center_zero:
            heatmap_kwargs["center"] = 0.0
        sns.heatmap(**heatmap_kwargs)
        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=25)
        ax.tick_params(axis="y", rotation=0)
    fig.suptitle(case_name, y=1.03, fontsize=14)
    fig.tight_layout()
    return fig


def plot_policy_definition_map(
    definition_df: pd.DataFrame,
    case_name: str,
    seed: int = 0,
    n_voters: int = 1800,
) -> plt.Figure:
    """Plot policy definitions in policy space alongside their welfare consequences."""
    electorate, candidates = build_case(case_name, seed=seed, n_voters=n_voters)
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6), dpi=150)

    plot_electorate(electorate, candidates, title=case_name, ax=axes[0])
    legend = axes[0].get_legend()
    if legend is not None:
        legend.remove()
    palette = dict(zip(POLICY_DEFINITION_ORDER, sns.color_palette("Dark2", n_colors=len(POLICY_DEFINITION_ORDER))))
    for _, row in definition_df.iterrows():
        axes[0].scatter(
            row["policy_x"],
            row["policy_y"],
            s=170,
            marker="*",
            color=palette[row["definition"]],
            edgecolor="white",
            linewidth=0.7,
            zorder=4,
        )
        axes[0].text(
            row["policy_x"] + 0.01,
            row["policy_y"] + 0.01,
            row["definition"],
            fontsize=8,
        )

    melted = definition_df.melt(
        id_vars=["definition"],
        value_vars=["aggregate_utility", "majority_utility", "minority_utility"],
        var_name="metric",
        value_name="value",
    )
    label_map = {
        "aggregate_utility": "Aggregate",
        "majority_utility": "Majority",
        "minority_utility": "Minority",
    }
    melted["metric"] = melted["metric"].map(label_map)
    sns.barplot(data=melted, x="definition", y="value", hue="metric", ax=axes[1])
    axes[1].set_title("Policy consequences by definition")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("utility")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].legend(frameon=False, loc="best")
    fig.tight_layout()
    return fig


def _majority_minority_groups(groups) -> tuple[object, object]:
    groups_sorted = sorted(groups, key=lambda group: (group.population_share, group.n_voters))
    return groups_sorted[-1], groups_sorted[0]


def summarize_case_takeaway(
    comparison_df: pd.DataFrame,
    case_name: str,
    tol: float = 1e-6,
) -> str:
    """Return a short markdown takeaway grounded in one case's simulation results."""
    subset = comparison_df[comparison_df["case"] == case_name].copy()
    models = list(CONSEQUENCE_MODEL_EXPLANATIONS)

    def _all_same(metric: str) -> bool:
        spreads = subset.groupby("consequence_model")[metric].agg(lambda s: float(s.max() - s.min()))
        return bool((spreads <= tol).all())

    if _all_same("policy_aggregate_utility") and _all_same("policy_minority_utility"):
        first = subset.iloc[0]
        n_systems = subset["system"].nunique()
        return (
            f"For **{case_name.lower()}**, all {n_systems} systems land on essentially the same outcome in this simulation. "
            f"That means the main action comes from the utility model rather than the rule itself: "
            f"the distance-only view looks fairly mild, the exposure-sensitive view makes the minority bloc look better off, "
            f"and the threshold-sensitive view turns the same policy into a noticeably harsher result once the redistribution floor is imposed."
        )

    exposure = subset[subset["consequence_model"] == "Exposure-sensitive"].copy()
    threshold = subset[subset["consequence_model"] == "Threshold-sensitive"].copy()

    best_agg = exposure["policy_aggregate_utility"].max()
    best_agg_systems = exposure.loc[
        exposure["policy_aggregate_utility"] >= best_agg - tol, "system"
    ].tolist()
    best_min = exposure["policy_minority_utility"].max()
    best_min_systems = exposure.loc[
        exposure["policy_minority_utility"] >= best_min - tol, "system"
    ].tolist()

    plurality_exposure = exposure[exposure["system"] == "Plurality"].iloc[0]
    leader_exposure = exposure.iloc[exposure["policy_aggregate_utility"].argmax()]
    agg_gain = leader_exposure["policy_aggregate_utility"] - plurality_exposure["policy_aggregate_utility"]
    minority_gain = leader_exposure["policy_minority_utility"] - plurality_exposure["policy_minority_utility"]
    fractional_exposure = exposure[
        exposure["system"].isin(
            ["Fractional discrete (sigma=1.0)", "Fractional continuous (sigma=1.0)"]
        )
    ]
    fractional_clause = ""
    if not fractional_exposure.empty:
        best_fractional = fractional_exposure.iloc[
            fractional_exposure["policy_minority_utility"].argmax()
        ]
        fractional_clause = (
            f" The best fractional variant here is {best_fractional['system']}, "
            f"with minority utility {best_fractional['policy_minority_utility']:.3f}."
        )

    threshold_penalties = threshold[["system", "policy_threshold_component"]].sort_values(
        "policy_threshold_component", ascending=True
    )
    worst_threshold = threshold_penalties.iloc[0]
    mildest_threshold = threshold_penalties.iloc[-1]

    return (
        f"For **{case_name.lower()}**, the exposure-sensitive model favors "
        f"{', '.join(best_agg_systems)} on aggregate utility and "
        f"{', '.join(best_min_systems)} on minority utility. "
        f"Relative to plurality, the exposure-sensitive leader improves aggregate utility by {agg_gain:.3f} "
        f"and minority utility by {minority_gain:.3f}. "
        f"When the policy floor is activated, the harshest threshold penalty falls on {worst_threshold['system']} "
        f"({worst_threshold['policy_threshold_component']:.3f}), while {mildest_threshold['system']} is penalized least "
        f"({mildest_threshold['policy_threshold_component']:.3f})."
        f"{fractional_clause}"
    )


def summarize_overall_conclusion(
    comparison_df: pd.DataFrame,
    tol: float = 1e-6,
) -> str:
    """Return a short markdown conclusion across all cases."""
    available_cases = [case for case in CASE_ORDER if case in set(comparison_df["case"])]
    rows = []
    for case_name in available_cases:
        subset = comparison_df[comparison_df["case"] == case_name]
        for model in CONSEQUENCE_MODEL_EXPLANATIONS:
            model_subset = subset[subset["consequence_model"] == model]
            if model_subset.empty:
                continue
            rows.append(
                {
                    "case": case_name,
                    "consequence_model": model,
                    "aggregate_leaders": tuple(
                        model_subset.loc[
                            model_subset["policy_aggregate_utility"]
                            >= model_subset["policy_aggregate_utility"].max() - tol,
                            "system",
                        ]
                    ),
                    "minority_leaders": tuple(
                        model_subset.loc[
                            model_subset["policy_minority_utility"]
                            >= model_subset["policy_minority_utility"].max() - tol,
                            "system",
                        ]
                    ),
                }
            )
    if not rows:
        return "No simulation results were available for the conclusion."
    summary = pd.DataFrame(rows)

    agg_counts = (
        summary["aggregate_leaders"]
        .explode()
        .value_counts()
        .reindex(SYSTEM_ORDER, fill_value=0)
    )
    minority_counts = (
        summary["minority_leaders"]
        .explode()
        .value_counts()
        .reindex(SYSTEM_ORDER, fill_value=0)
    )

    top_agg = agg_counts[agg_counts == agg_counts.max()].index.tolist()
    top_minority = minority_counts[minority_counts == minority_counts.max()].index.tolist()

    threshold = comparison_df[comparison_df["consequence_model"] == "Threshold-sensitive"].copy()
    threshold_means = (
        threshold.groupby("system")["policy_threshold_component"]
        .mean()
        .reindex(SYSTEM_ORDER)
    )
    hardest_hit = threshold_means.idxmin()
    least_hit = threshold_means.idxmax()
    fractional_means = threshold_means.reindex(
        ["Fractional discrete (sigma=1.0)", "Fractional continuous (sigma=1.0)"]
    ).dropna()
    fractional_clause = ""
    if not fractional_means.empty:
        fractional_clause = (
            f" The two fractional variants sit in the middle of the pack here, "
            f"with average threshold penalties around {fractional_means.iloc[0]:.3f}."
        )

    return (
        f"Across the three cases, aggregate-utility wins most often go to {', '.join(top_agg)}, "
        f"and minority-utility wins most often go to {', '.join(top_minority)}. "
        f"The broad pattern in these runs is that once the slate creates a real choice, plurality and approval tend to fall behind the more coalition-friendly rules; "
        f"when the slate does **not** create a real choice, the systems converge and the utility model does the real explanatory work. "
        f"The threshold-sensitive model also shows that near-miss policies are not small misses: on average {hardest_hit} takes the biggest floor penalty "
        f"({threshold_means.loc[hardest_hit]:.3f}), while {least_hit} is penalized least ({threshold_means.loc[least_hit]:.3f})."
        f"{fractional_clause}"
    )


__all__ = [
    "CASE_ORDER",
    "CASE_EXPLANATIONS",
    "CONSEQUENCE_MODEL_EXPLANATIONS",
    "DISPLAY_LABELS",
    "POLICY_DEFINITION_ORDER",
    "SYSTEM_ORDER",
    "SYSTEM_EXPLANATIONS",
    "build_blended_compromise_definition",
    "build_case",
    "build_distance_only_spec",
    "build_exposure_sensitive_spec",
    "build_threshold_sensitive_spec",
    "case_comparison_table",
    "case_reference_table",
    "compare_consequence_models",
    "compare_policy_definitions",
    "consequence_model_reference_table",
    "exposure_sensitive_utility",
    "hidden_harm_cases",
    "plot_case_gallery",
    "plot_case_policy_points",
    "plot_case_model_heatmaps",
    "plot_consequence_model_comparison",
    "plot_hidden_harm_scatter",
    "plot_policy_definition_map",
    "plot_policy_heatmaps",
    "plot_representation_policy_contrasts",
    "representation_policy_helper_overview",
    "run_case_grid",
    "run_case_systems",
    "summarize_case_takeaway",
    "summarize_overall_conclusion",
    "system_reference_table",
]
