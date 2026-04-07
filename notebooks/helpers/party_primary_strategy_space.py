"""Notebook helpers for exploring parties, primaries, turnout, and strategy together."""
from __future__ import annotations

from dataclasses import dataclass
import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import qmc

from electoral_sim.ballots import BallotProfile
from electoral_sim.candidates import CandidateSet
from electoral_sim.electorate import gaussian_mixture_electorate
from electoral_sim.metrics import compute_group_metrics, compute_metrics
from electoral_sim.parties import PartySet, fixed_parties, nearest_party_distances
from electoral_sim.primaries import (
    PartySpec,
    PrimaryType,
    assign_party_membership,
    build_party_specs_from_positions,
    run_open_primary_top_k,
    run_two_party_primary,
)
from electoral_sim.strategies import (
    ApprovalThresholdStrategy,
    PluralityCompromiseStrategy,
    RankedBuryingStrategy,
    RankedTruncationStrategy,
    ScoreMaxMinStrategy,
    SincereStrategy,
    StrategyModel,
    VotingContext,
)
from electoral_sim.systems import (
    ApprovalVoting,
    CondorcetSchulze,
    InstantRunoff,
    Plurality,
    ScoreVoting,
)


RATIO_ORDER = ["60:40", "90:10", "95:5", "99:1"]
ELECTORATE_ORDER = [
    "Aligned polarization",
    "Shared center",
    "Cross-pressured minority",
    "Asymmetric sorting",
]
PRIMARY_ORDER = [
    "No primary",
    "Closed plurality",
    "Closed IRV",
    "Semi plurality",
    "Top-4 open",
]
TURNOUT_ORDER = [
    "Even turnout",
    "Activist primary",
    "Moderate primary",
    "Minority surge",
]
STRATEGY_ORDER = [
    "Sincere",
    "Defensive compromise",
    "Aggressive compromise",
    "Adversarial ranking",
]
SLATE_ORDER = [
    "Balanced ladder",
    "Bridge-heavy",
    "Majority-heavy",
    "Insurgent right",
]
SYSTEM_ORDER = ["Plurality", "IRV", "Approval", "Score", "Condorcet"]


RATIO_WEIGHTS = {
    "60:40": (0.60, 0.40),
    "90:10": (0.90, 0.10),
    "95:5": (0.95, 0.05),
    "99:1": (0.99, 0.01),
}

ELECTORATE_SPECS = {
    "Aligned polarization": {
        "majority": [
            {"weight": 0.72, "mean": [0.24, 0.58], "cov": [[0.010, 0.002], [0.002, 0.012]]},
            {"weight": 0.28, "mean": [0.40, 0.52], "cov": [[0.008, 0.001], [0.001, 0.010]]},
        ],
        "minority": [
            {"weight": 0.28, "mean": [0.58, 0.48], "cov": [[0.008, -0.001], [-0.001, 0.010]]},
            {"weight": 0.72, "mean": [0.78, 0.40], "cov": [[0.010, -0.002], [-0.002, 0.012]]},
        ],
    },
    "Shared center": {
        "majority": [
            {"weight": 0.58, "mean": [0.40, 0.53], "cov": [[0.010, 0.001], [0.001, 0.011]]},
            {"weight": 0.42, "mean": [0.52, 0.49], "cov": [[0.009, 0.000], [0.000, 0.009]]},
        ],
        "minority": [
            {"weight": 0.52, "mean": [0.48, 0.51], "cov": [[0.010, 0.000], [0.000, 0.010]]},
            {"weight": 0.48, "mean": [0.62, 0.47], "cov": [[0.010, -0.001], [-0.001, 0.011]]},
        ],
    },
    "Cross-pressured minority": {
        "majority": [
            {"weight": 0.45, "mean": [0.22, 0.60], "cov": [[0.010, 0.002], [0.002, 0.012]]},
            {"weight": 0.55, "mean": [0.66, 0.46], "cov": [[0.015, -0.002], [-0.002, 0.015]]},
        ],
        "minority": [
            {"weight": 0.50, "mean": [0.44, 0.54], "cov": [[0.008, 0.001], [0.001, 0.010]]},
            {"weight": 0.50, "mean": [0.60, 0.46], "cov": [[0.008, -0.001], [-0.001, 0.010]]},
        ],
    },
    "Asymmetric sorting": {
        "majority": [
            {"weight": 0.34, "mean": [0.18, 0.62], "cov": [[0.008, 0.001], [0.001, 0.010]]},
            {"weight": 0.46, "mean": [0.38, 0.54], "cov": [[0.009, 0.001], [0.001, 0.011]]},
            {"weight": 0.20, "mean": [0.64, 0.48], "cov": [[0.012, -0.001], [-0.001, 0.012]]},
        ],
        "minority": [
            {"weight": 0.36, "mean": [0.54, 0.50], "cov": [[0.009, 0.000], [0.000, 0.010]]},
            {"weight": 0.64, "mean": [0.80, 0.40], "cov": [[0.010, -0.002], [-0.002, 0.012]]},
        ],
    },
}

SLATE_SPECS = {
    "Balanced ladder": np.array(
        [
            [0.12, 0.62],
            [0.26, 0.56],
            [0.40, 0.50],
            [0.60, 0.50],
            [0.74, 0.44],
            [0.88, 0.38],
        ]
    ),
    "Bridge-heavy": np.array(
        [
            [0.18, 0.60],
            [0.30, 0.55],
            [0.44, 0.51],
            [0.56, 0.49],
            [0.70, 0.45],
            [0.82, 0.40],
        ]
    ),
    "Majority-heavy": np.array(
        [
            [0.10, 0.64],
            [0.22, 0.58],
            [0.34, 0.54],
            [0.50, 0.50],
            [0.72, 0.44],
            [0.86, 0.38],
        ]
    ),
    "Insurgent right": np.array(
        [
            [0.16, 0.60],
            [0.30, 0.55],
            [0.42, 0.51],
            [0.64, 0.49],
            [0.80, 0.44],
            [0.94, 0.28],
        ]
    ),
}

SLATE_LABELS = {
    "Balanced ladder": ["L-Base", "L-Bridge", "L-Moderate", "R-Moderate", "R-Bridge", "R-Base"],
    "Bridge-heavy": ["L-Base", "L-Bridge", "L-Center", "R-Center", "R-Bridge", "R-Base"],
    "Majority-heavy": ["L-Base", "L-Plus", "L-Bridge", "Center", "R-Bridge", "R-Base"],
    "Insurgent right": ["L-Base", "L-Bridge", "L-Moderate", "R-Moderate", "R-Bridge", "R-Insurgent"],
}

DISPLAY_LABELS = {
    "Aligned polarization": "Aligned\npolarization",
    "Shared center": "Shared\ncenter",
    "Cross-pressured minority": "Cross-pressured\nminority",
    "Asymmetric sorting": "Asymmetric\nsorting",
    "Even turnout": "Even",
    "Activist primary": "Activist\nprimary",
    "Moderate primary": "Moderate\nprimary",
    "Minority surge": "Minority\nsurge",
    "Balanced ladder": "Balanced\nladder",
    "Bridge-heavy": "Bridge-\nheavy",
    "Majority-heavy": "Majority-\nheavy",
    "Insurgent right": "Insurgent\nright",
}


@dataclass(frozen=True)
class SystemSpec:
    key: str
    display_name: str
    family: str

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
        raise ValueError(f"Unknown system key: {self.key}")


@dataclass(frozen=True)
class PrimarySpec:
    key: str
    display_name: str
    kind: str
    primary_type: PrimaryType | None = None
    primary_family: str | None = None
    top_k: int | None = None

    def build_primary_system(self):
        if self.primary_family == "plurality":
            return Plurality()
        if self.primary_family == "ranked":
            return InstantRunoff()
        return None


SYSTEM_SPECS = [
    SystemSpec("plurality", "Plurality", "plurality"),
    SystemSpec("irv", "IRV", "ranked"),
    SystemSpec("approval", "Approval", "approval"),
    SystemSpec("score", "Score", "score"),
    SystemSpec("condorcet", "Condorcet", "ranked"),
]
SYSTEM_SPECS_BY_NAME = {spec.display_name: spec for spec in SYSTEM_SPECS}

PRIMARY_SPECS = [
    PrimarySpec("none", "No primary", "none"),
    PrimarySpec("closed_plurality", "Closed plurality", "two_party", PrimaryType.CLOSED, "plurality"),
    PrimarySpec("closed_irv", "Closed IRV", "two_party", PrimaryType.CLOSED, "ranked"),
    PrimarySpec("semi_plurality", "Semi plurality", "two_party", PrimaryType.SEMI, "plurality"),
    PrimarySpec("top4_open", "Top-4 open", "open_top_k", PrimaryType.OPEN, "plurality", top_k=4),
]
PRIMARY_SPECS_BY_NAME = {spec.display_name: spec for spec in PRIMARY_SPECS}


class CompositeMaskStrategy(StrategyModel):
    """Apply a ballot-expression strategy and then force a fixed turnout mask."""

    def __init__(self, base_strategy: StrategyModel, active_mask: np.ndarray | None = None):
        self.base_strategy = base_strategy
        self.active_mask = None if active_mask is None else np.asarray(active_mask, dtype=bool)

    @property
    def name(self) -> str:
        if self.active_mask is None:
            return self.base_strategy.name
        return f"{self.base_strategy.name} + fixed turnout"

    def generate_ballots(
        self,
        electorate,
        candidates,
        approval_threshold=None,
        context=None,
    ):
        ballots = self.base_strategy.generate_ballots(
            electorate,
            candidates,
            approval_threshold=approval_threshold,
            context=context,
        )
        if self.active_mask is None:
            return ballots
        if self.active_mask.shape != (ballots.n_voters,):
            raise ValueError("Turnout mask does not match the electorate size.")
        return BallotProfile(
            plurality=ballots.plurality,
            rankings=ballots.rankings,
            scores=ballots.scores,
            approvals=ballots.approvals,
            distances=ballots.distances,
            approval_threshold=ballots.approval_threshold,
            n_voters=ballots.n_voters,
            n_candidates=ballots.n_candidates,
            active_voter_mask=ballots.active_voter_mask & self.active_mask,
        )


def build_party_positions() -> PartySet:
    """Two-party anchors used to model rational party choice by ideological proximity."""
    return fixed_parties(
        [[0.28, 0.56], [0.72, 0.44]],
        ["Left Party", "Right Party"],
    )


def build_grouped_electorate(
    electorate_name: str,
    ratio_name: str,
    seed: int,
    n_voters: int = 1800,
    dim_names: list[str] | None = None,
):
    """
    Build a labeled two-group electorate.

    Group labels represent a majority and minority population. Party choice is
    not hard-coded here. That gets inferred later from whichever party anchor
    sits closest to each voter in the spatial map.
    """
    if ratio_name in RATIO_WEIGHTS:
        majority_weight, minority_weight = RATIO_WEIGHTS[ratio_name]
    else:
        try:
            left, right = str(ratio_name).split(":")
            majority_pct = float(left)
            minority_pct = float(right)
        except ValueError as exc:
            raise KeyError(
                f"Unknown ratio label: {ratio_name}. Expected one of {sorted(RATIO_WEIGHTS)} "
                "or a label like '68:32'."
            ) from exc
        total = majority_pct + minority_pct
        if total <= 0:
            raise ValueError(f"Invalid ratio label: {ratio_name}")
        majority_weight = majority_pct / total
        minority_weight = minority_pct / total

    profile = ELECTORATE_SPECS[electorate_name]
    components = []
    for group_name, group_weight in [
        ("Majority group", majority_weight),
        ("Minority group", minority_weight),
    ]:
        for component in profile["majority" if group_name == "Majority group" else "minority"]:
            components.append(
                {
                    "weight": group_weight * component["weight"],
                    "mean": component["mean"],
                    "cov": component["cov"],
                    "group": group_name,
                }
            )

    # Very small electorates and extreme ratios can occasionally drop the
    # minority entirely under a pure multinomial draw. For this notebook the
    # comparison only makes sense when both labeled groups are present, so
    # redraw until that condition is met.
    for attempt in range(50):
        rng = np.random.default_rng(seed + attempt)
        electorate = gaussian_mixture_electorate(
            n_voters,
            components,
            rng=rng,
            dim_names=dim_names or ["economic", "social"],
        )
        group_names = set(electorate.group_labels().values())
        if {"Majority group", "Minority group"}.issubset(group_names):
            return electorate

    raise RuntimeError(
        "Could not sample an electorate containing both majority and minority groups. "
        "Try increasing n_voters or adjusting the ratio."
    )


def build_candidate_slate(name: str) -> CandidateSet:
    """Return one of the hand-built candidate slates."""
    return CandidateSet(SLATE_SPECS[name].copy(), list(SLATE_LABELS[name]))


def full_space_helper_overview() -> pd.DataFrame:
    """Compact description of the notebook design space."""
    return pd.DataFrame(
        {
            "dimension": [
                "Electorates",
                "Primary pipelines",
                "Turnout profiles",
                "Strategy profiles",
                "Majority:minority ratios",
                "Candidate slates",
                "General-election systems",
            ],
            "values": [
                ", ".join(ELECTORATE_ORDER),
                ", ".join(PRIMARY_ORDER),
                ", ".join(TURNOUT_ORDER),
                ", ".join(STRATEGY_ORDER),
                ", ".join(RATIO_ORDER),
                ", ".join(SLATE_ORDER),
                ", ".join(SYSTEM_ORDER),
            ],
        }
    )


def mixed_lhs_helper_overview() -> pd.DataFrame:
    """Compact description of the mixed Latin-hypercube design."""
    return pd.DataFrame(
        {
            "dimension": [
                "Majority share",
                "Electorate family",
                "Primary pipeline",
                "Turnout profile",
                "Turnout intensity",
                "Strategy profile",
                "Strategy intensity",
                "Candidate slate",
                "General-election systems",
            ],
            "values": [
                "[0.60, 0.99]",
                ", ".join(ELECTORATE_ORDER),
                ", ".join(PRIMARY_ORDER),
                ", ".join(TURNOUT_ORDER),
                "0 = mild turnout distortion, 1 = strong turnout distortion",
                ", ".join(STRATEGY_ORDER),
                "0 = mild strategic pressure, 1 = strong strategic pressure",
                ", ".join(SLATE_ORDER),
                ", ".join(SYSTEM_ORDER),
            ],
        }
    )


def sample_mixed_latin_hypercube_design(
    n_cases: int = 48,
    seed: int = 20260407,
) -> pd.DataFrame:
    """
    Sample a compact mixed design over continuous and discrete notebook axes.

    The majority share and the intensity columns are sampled continuously.
    Electorate family, primary, turnout family, strategy family, and slate are
    assigned by binning Latin-hypercube coordinates.
    """
    sampler = qmc.LatinHypercube(d=8, seed=seed)
    raw = sampler.random(n=n_cases)

    majority_share = RATIO_WEIGHTS["60:40"][0] + raw[:, 0] * (RATIO_WEIGHTS["99:1"][0] - RATIO_WEIGHTS["60:40"][0])
    electorate_idx = np.minimum((raw[:, 1] * len(ELECTORATE_ORDER)).astype(int), len(ELECTORATE_ORDER) - 1)
    primary_idx = np.minimum((raw[:, 2] * len(PRIMARY_ORDER)).astype(int), len(PRIMARY_ORDER) - 1)
    turnout_idx = np.minimum((raw[:, 3] * len(TURNOUT_ORDER)).astype(int), len(TURNOUT_ORDER) - 1)
    turnout_strength = raw[:, 4]
    strategy_idx = np.minimum((raw[:, 5] * len(STRATEGY_ORDER)).astype(int), len(STRATEGY_ORDER) - 1)
    strategy_strength = raw[:, 6]
    slate_idx = np.minimum((raw[:, 7] * len(SLATE_ORDER)).astype(int), len(SLATE_ORDER) - 1)

    design = pd.DataFrame(
        {
            "case_id": np.arange(n_cases, dtype=int),
            "majority_share": majority_share,
            "minority_share": 1.0 - majority_share,
            "electorate": [ELECTORATE_ORDER[idx] for idx in electorate_idx],
            "primary": [PRIMARY_ORDER[idx] for idx in primary_idx],
            "turnout": [TURNOUT_ORDER[idx] for idx in turnout_idx],
            "turnout_strength": turnout_strength,
            "strategy": [STRATEGY_ORDER[idx] for idx in strategy_idx],
            "strategy_strength": strategy_strength,
            "candidate_slate": [SLATE_ORDER[idx] for idx in slate_idx],
        }
    )
    design["ratio"] = [
        f"{int(round(100 * share))}:{100 - int(round(100 * share))}"
        for share in design["majority_share"]
    ]
    return design.sort_values(
        ["majority_share", "electorate", "primary", "turnout", "strategy", "candidate_slate"]
    ).reset_index(drop=True)


def plot_mixed_lhs_parameter_coverage(design_df: pd.DataFrame):
    """Show how the sampled cases cover the mixed design space."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), dpi=150)
    turnout_palette = {
        "Even turnout": "#457b9d",
        "Activist primary": "#e76f51",
        "Moderate primary": "#2a9d8f",
        "Minority surge": "#f4a261",
    }
    strategy_markers = {
        "Sincere": "o",
        "Defensive compromise": "s",
        "Aggressive compromise": "^",
        "Adversarial ranking": "X",
    }

    for turnout_name in TURNOUT_ORDER:
        for strategy_name in STRATEGY_ORDER:
            subset = design_df[
                (design_df["turnout"] == turnout_name)
                & (design_df["strategy"] == strategy_name)
            ]
            if subset.empty:
                continue
            axes[0].scatter(
                subset["majority_share"],
                subset["turnout_strength"],
                s=70,
                alpha=0.8,
                color=turnout_palette[turnout_name],
                marker=strategy_markers[strategy_name],
            )

    axes[0].set_title("Coverage of ratio and turnout intensity", fontsize=10)
    axes[0].set_xlabel("Majority share")
    axes[0].set_ylabel("Turnout intensity")

    slate_palette = dict(zip(SLATE_ORDER, ["#264653", "#2a9d8f", "#e9c46a", "#e76f51"]))
    for slate_name in SLATE_ORDER:
        subset = design_df[design_df["candidate_slate"] == slate_name]
        axes[1].scatter(
            subset["majority_share"],
            subset["strategy_strength"],
            s=80,
            alpha=0.85,
            color=slate_palette[slate_name],
            label=slate_name,
        )

    axes[1].set_title("Coverage of ratio and strategy intensity", fontsize=10)
    axes[1].set_xlabel("Majority share")
    axes[1].set_ylabel("Strategy intensity")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


def _seed_offset(label: str) -> int:
    return sum((idx + 1) * ord(ch) for idx, ch in enumerate(label))


def _build_stage_context(electorate, candidates) -> VotingContext:
    """Use sincere first preferences as a simple polling proxy."""
    ballots = BallotProfile.from_preferences(electorate, candidates)
    counts = ballots.plurality_counts().astype(float)
    total = counts.sum()
    poll_shares = counts / total if total > 0 else np.zeros_like(counts)
    frontrunners = np.argsort(-counts, kind="stable")[: min(2, len(counts))].tolist()
    return VotingContext(
        poll_shares=poll_shares,
        frontrunner_indices=frontrunners,
    )


def _base_expression_strategy(
    strategy_name: str,
    family: str,
    seed: int,
    strength: float | None = None,
) -> StrategyModel:
    """
    Build the expression strategy for a given ballot family.

    Strategy profiles are intentionally family-matched so a single notebook
    axis can span plurality, approval, score, and ranked systems without each
    system being forced into an unnatural model.
    """
    if strategy_name == "Sincere":
        return SincereStrategy()

    strength = 1.0 if strength is None else float(np.clip(strength, 0.0, 1.0))

    if strategy_name == "Defensive compromise":
        if family == "plurality":
            compromise_rate = 0.40 + 0.40 * strength
            return PluralityCompromiseStrategy(compromise_rate=compromise_rate, rng=np.random.default_rng(seed))
        if family == "approval":
            return ApprovalThresholdStrategy(utility_threshold=0.55 + 0.15 * strength)
        if family == "score":
            return ScoreMaxMinStrategy(utility_threshold=0.55 + 0.15 * strength)
        if family == "ranked":
            max_ranked = 3 if strength < 0.5 else 2
            return RankedTruncationStrategy(max_ranked=max_ranked)
        return SincereStrategy()

    if strategy_name == "Aggressive compromise":
        if family == "plurality":
            compromise_rate = 0.65 + 0.35 * strength
            return PluralityCompromiseStrategy(compromise_rate=compromise_rate, rng=np.random.default_rng(seed))
        if family == "approval":
            return ApprovalThresholdStrategy(utility_threshold=0.68 + 0.22 * strength)
        if family == "score":
            return ScoreMaxMinStrategy(utility_threshold=0.68 + 0.22 * strength)
        if family == "ranked":
            max_ranked = 2 if strength < 0.35 else 1
            return RankedTruncationStrategy(max_ranked=max_ranked)
        return SincereStrategy()

    if strategy_name == "Adversarial ranking":
        if family == "plurality":
            compromise_rate = 0.55 + 0.35 * strength
            return PluralityCompromiseStrategy(compromise_rate=compromise_rate, rng=np.random.default_rng(seed))
        if family == "approval":
            return ApprovalThresholdStrategy(utility_threshold=0.62 + 0.20 * strength)
        if family == "score":
            return ScoreMaxMinStrategy(utility_threshold=0.62 + 0.20 * strength)
        if family == "ranked":
            bury_rate = 0.40 + 0.55 * strength
            return RankedBuryingStrategy(bury_rate=bury_rate, rng=np.random.default_rng(seed))
        return SincereStrategy()

    raise ValueError(f"Unknown strategy profile: {strategy_name}")


def _build_stage_strategy(
    strategy_name: str,
    family: str,
    seed: int,
    active_mask: np.ndarray | None = None,
    strength: float | None = None,
) -> StrategyModel:
    base_strategy = _base_expression_strategy(strategy_name, family, seed, strength=strength)
    return CompositeMaskStrategy(base_strategy, active_mask=active_mask)


def _minority_mask(electorate) -> np.ndarray:
    labels = electorate.group_labels()
    minority_id = next(group_id for group_id, name in labels.items() if name == "Minority group")
    return electorate.group_ids == minority_id


def build_turnout_setup(
    turnout_name: str,
    electorate,
    candidates: CandidateSet,
    party_positions: PartySet,
    party_specs: list[PartySpec],
    seed: int,
    turnout_strength: float | None = None,
) -> dict[str, object]:
    """
    Build turnout inputs while keeping turnout separate from ballot strategy.

    Closed and semi-open primary memberships are derived from nearest-party
    rules and then filtered by the turnout profile. Open-primary turnout and
    general-election turnout are stored as explicit active-voter masks.
    """
    rng = np.random.default_rng(seed)
    turnout_strength = 1.0 if turnout_strength is None else float(np.clip(turnout_strength, 0.0, 1.0))
    closed = assign_party_membership(
        electorate,
        candidates,
        party_specs,
        primary_type=PrimaryType.CLOSED,
        party_positions=party_positions,
    )
    semi = assign_party_membership(
        electorate,
        candidates,
        party_specs,
        primary_type=PrimaryType.SEMI,
        party_positions=party_positions,
    )

    party_dists = nearest_party_distances(electorate.preferences, party_positions)
    center_distance = np.abs(electorate.preferences[:, 0] - 0.5)
    partisanship = np.abs(party_dists[:, 0] - party_dists[:, 1])
    minority_mask = _minority_mask(electorate)
    majority_mask = ~minority_mask

    general_mask = np.ones(electorate.n_voters, dtype=bool)
    open_primary_mask = np.ones(electorate.n_voters, dtype=bool)
    closed_memberships = {name: mask.copy() for name, mask in closed.items()}
    semi_memberships = {name: mask.copy() for name, mask in semi.items()}

    if turnout_name == "Even turnout":
        pass
    elif turnout_name == "Activist primary":
        activist_quantile = 0.25 + 0.40 * turnout_strength
        activist_mask = center_distance >= np.quantile(center_distance, activist_quantile)
        open_primary_mask = activist_mask
        for name, mask in closed_memberships.items():
            threshold = np.quantile(center_distance[mask], activist_quantile) if mask.any() else 0.0
            closed_memberships[name] = mask & (center_distance >= threshold)
        for name, mask in semi_memberships.items():
            threshold = np.quantile(center_distance[mask], activist_quantile) if mask.any() else 0.0
            semi_memberships[name] = mask & (center_distance >= threshold)
    elif turnout_name == "Moderate primary":
        moderate_quantile = 0.80 - 0.35 * turnout_strength
        moderate_mask = center_distance <= np.quantile(center_distance, moderate_quantile)
        open_primary_mask = moderate_mask
        for name, mask in closed_memberships.items():
            threshold = np.quantile(center_distance[mask], moderate_quantile) if mask.any() else 0.0
            closed_memberships[name] = mask & (center_distance <= threshold)
        for name, mask in semi_memberships.items():
            threshold = np.quantile(center_distance[mask], moderate_quantile) if mask.any() else 0.0
            semi_memberships[name] = mask & (center_distance <= threshold)
    elif turnout_name == "Minority surge":
        open_prob = np.where(minority_mask, 0.78 + 0.19 * turnout_strength, 0.88 - 0.18 * turnout_strength)
        general_prob = np.where(minority_mask, 0.82 + 0.16 * turnout_strength, 0.90 - 0.14 * turnout_strength)
        open_primary_mask = rng.random(electorate.n_voters) < open_prob
        general_mask = rng.random(electorate.n_voters) < general_prob
        for name, mask in closed_memberships.items():
            party_prob = np.where(minority_mask, 0.78 + 0.19 * turnout_strength, 0.86 - 0.16 * turnout_strength)
            closed_memberships[name] = mask & (rng.random(electorate.n_voters) < party_prob)
        for name, mask in semi_memberships.items():
            party_prob = np.where(minority_mask, 0.78 + 0.19 * turnout_strength, 0.86 - 0.16 * turnout_strength)
            semi_memberships[name] = mask & (rng.random(electorate.n_voters) < party_prob)
    else:
        raise ValueError(f"Unknown turnout profile: {turnout_name}")

    return {
        "closed_memberships": closed_memberships,
        "semi_memberships": semi_memberships,
        "open_primary_mask": open_primary_mask,
        "general_mask": general_mask,
        "majority_mask": majority_mask,
        "minority_mask": minority_mask,
        "partisanship": partisanship,
    }


def plot_party_snapshot(
    electorate,
    candidates: CandidateSet,
    parties: PartySet | None = None,
    ax: plt.Axes | None = None,
    title: str | None = None,
    show_legend: bool = True,
):
    """Scatter voters, party anchors, and candidates in a single spatial panel."""
    if electorate.n_dims != 2:
        raise ValueError("This helper expects a 2D electorate.")
    ax = ax or plt.gca()
    labels = electorate.group_labels()
    colors = {name: color for name, color in [("Majority group", "#457b9d"), ("Minority group", "#e76f51")]}
    for group_id, group_name in labels.items():
        mask = electorate.group_ids == group_id
        ax.scatter(
            electorate.preferences[mask, 0],
            electorate.preferences[mask, 1],
            s=10,
            alpha=0.18,
            color=colors.get(group_name, "#888888"),
            label=group_name if show_legend else None,
        )

    ax.scatter(
        candidates.positions[:, 0],
        candidates.positions[:, 1],
        marker="X",
        s=130,
        color="#222222",
        label="Candidates" if show_legend else None,
        zorder=5,
    )
    for idx, label in enumerate(candidates.labels):
        ax.annotate(label, candidates.positions[idx], xytext=(4, 5), textcoords="offset points", fontsize=7)

    if parties is not None:
        ax.scatter(
            parties.positions[:, 0],
            parties.positions[:, 1],
            marker="D",
            s=110,
            color="#2a9d8f",
            label="Party anchors" if show_legend else None,
            zorder=6,
        )
        for idx, label in enumerate(parties.labels):
            ax.annotate(label, parties.positions[idx], xytext=(4, -10), textcoords="offset points", fontsize=7)

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel(electorate.dim_names[0])
    ax.set_ylabel(electorate.dim_names[1])
    if title is not None:
        ax.set_title(title, fontsize=10)
    if show_legend:
        ax.legend(frameon=False, fontsize=8, loc="upper center", ncol=2)
    return ax


def _winner_party_name(result_row: dict, parties: PartySet) -> str | None:
    winner_position = np.asarray(result_row["winner_position"], dtype=float).reshape(1, -1)
    idx = nearest_party_distances(winner_position, parties).argmin(axis=1)[0]
    return parties.labels[int(idx)]


def _group_metric_lookup(summary) -> dict[str, dict[str, float]]:
    return {
        group.group_name: {
            "welfare": group.welfare,
            "mean_voter_distance": group.mean_voter_distance,
            "majority_satisfaction": group.majority_satisfaction,
            "population_share": group.population_share,
        }
        for group in summary.groups
    }


def _run_baseline(
    electorate,
    candidates: CandidateSet,
    system_spec: SystemSpec,
    strategy_name: str,
    general_mask: np.ndarray,
    seed: int,
    strategy_strength: float | None = None,
):
    system = system_spec.build()
    context = _build_stage_context(electorate, candidates)
    strategy = _build_stage_strategy(
        strategy_name,
        system_spec.family,
        seed,
        active_mask=general_mask,
        strength=strategy_strength,
    )
    ballots = BallotProfile.from_strategy(
        electorate,
        candidates,
        strategy=strategy,
        context=context,
    )
    result = system.run(ballots, candidates)
    metrics = compute_metrics(result, electorate, candidates)
    groups = compute_group_metrics(result, electorate, candidates)
    return result, metrics, groups


def run_design_case(
    electorate_name: str,
    primary_name: str,
    turnout_name: str,
    strategy_name: str,
    ratio_name: str | None,
    slate_name: str,
    system_name: str,
    seed: int,
    n_voters: int = 1800,
    majority_share: float | None = None,
    turnout_strength: float | None = None,
    strategy_strength: float | None = None,
) -> dict[str, object]:
    """
    Run one combination from the full notebook design space.

    The returned row is intentionally flat so the notebook can build tables,
    heatmaps, and filtered views without additional wrangling.
    """
    if majority_share is None:
        if ratio_name is None:
            raise ValueError("Either ratio_name or majority_share must be provided.")
        majority_share = RATIO_WEIGHTS[ratio_name][0]
    majority_share = float(np.clip(majority_share, 0.60, 0.99))
    ratio_label = (
        ratio_name
        if ratio_name is not None
        else f"{int(round(100 * majority_share))}:{100 - int(round(100 * majority_share))}"
    )

    electorate = build_grouped_electorate(
        electorate_name,
        ratio_name=ratio_name or "60:40",
        seed=seed,
        n_voters=n_voters,
        dim_names=None,
    )
    if not np.isclose(majority_share, RATIO_WEIGHTS.get(ratio_label, (majority_share,))[0]):
        # Rebuild using the sampled share while preserving the same electorate family.
        profile = ELECTORATE_SPECS[electorate_name]
        components = []
        for group_name, group_weight in [
            ("Majority group", majority_share),
            ("Minority group", 1.0 - majority_share),
        ]:
            source_key = "majority" if group_name == "Majority group" else "minority"
            for component in profile[source_key]:
                components.append(
                    {
                        "weight": group_weight * component["weight"],
                        "mean": component["mean"],
                        "cov": component["cov"],
                        "group": group_name,
                    }
                )
        for attempt in range(50):
            rng = np.random.default_rng(seed + attempt)
            electorate = gaussian_mixture_electorate(
                n_voters,
                components,
                rng=rng,
                dim_names=["economic", "social"],
            )
            group_names = set(electorate.group_labels().values())
            if {"Majority group", "Minority group"}.issubset(group_names):
                break
        else:
            raise RuntimeError("Could not sample an electorate with both groups present.")

    party_positions = build_party_positions()
    candidates = build_candidate_slate(slate_name)
    primary_spec = PRIMARY_SPECS_BY_NAME[primary_name]
    system_spec = SYSTEM_SPECS_BY_NAME[system_name]
    party_specs = build_party_specs_from_positions(
        candidates,
        party_positions,
        primary_systems=primary_spec.build_primary_system() if primary_spec.kind == "two_party" else Plurality(),
    )
    turnout = build_turnout_setup(
        turnout_name,
        electorate,
        candidates,
        party_positions,
        party_specs,
        seed=seed + _seed_offset("turnout"),
        turnout_strength=turnout_strength,
    )

    baseline_result, baseline_metrics, baseline_groups = _run_baseline(
        electorate,
        candidates,
        system_spec,
        strategy_name,
        turnout["general_mask"],
        seed + _seed_offset("baseline"),
        strategy_strength=strategy_strength,
    )

    if primary_spec.kind == "none":
        result = baseline_result
        metrics = baseline_metrics
        groups = baseline_groups
        final_candidates = candidates
        pipeline_meta = {
            "pipeline_type": "no_primary",
            "n_primary_voters": int(turnout["general_mask"].sum()),
            "distance_to_median_delta": 0.0,
        }
    elif primary_spec.kind == "two_party":
        primary_strategy = lambda party, primary_electorate, party_candidates: _build_stage_strategy(
            strategy_name,
            primary_spec.primary_family,
            seed + _seed_offset(f"primary-{party.name}"),
            strength=strategy_strength,
        )
        primary_context = lambda _party, primary_electorate, party_candidates: _build_stage_context(
            primary_electorate,
            party_candidates,
        )
        general_strategy = lambda stage_electorate, stage_candidates, _meta: _build_stage_strategy(
            strategy_name,
            system_spec.family,
            seed + _seed_offset("general"),
            active_mask=turnout["general_mask"],
            strength=strategy_strength,
        )
        general_context = lambda stage_electorate, stage_candidates, _meta: _build_stage_context(
            stage_electorate,
            stage_candidates,
        )
        memberships = (
            turnout["closed_memberships"]
            if primary_spec.primary_type == PrimaryType.CLOSED
            else turnout["semi_memberships"]
        )
        primary_result = run_two_party_primary(
            electorate,
            candidates,
            party_specs,
            general_system=system_spec.build(),
            primary_type=primary_spec.primary_type,
            memberships=memberships,
            primary_strategy=primary_strategy,
            primary_context=primary_context,
            general_strategy=general_strategy,
            general_context=general_context,
        )
        result = primary_result.general_result
        final_candidates = CandidateSet(
            np.array([item.nominee_position for item in primary_result.primary_results]),
            [
                f"{item.party_name} nominee ({candidates.labels[item.nominee_index]})"
                for item in primary_result.primary_results
            ],
        )
        metrics = compute_metrics(result, electorate, final_candidates)
        groups = compute_group_metrics(result, electorate, final_candidates)
        pipeline_meta = {
            "pipeline_type": "two_party_primary",
            "n_primary_voters": int(sum(mask.sum() for mask in memberships.values())),
            "distance_to_median_delta": metrics.distance_to_median - baseline_metrics.distance_to_median,
            "primary_voters_by_party": {name: int(mask.sum()) for name, mask in memberships.items()},
        }
    elif primary_spec.kind == "open_top_k":
        primary_strategy = _build_stage_strategy(
            strategy_name,
            primary_spec.primary_family,
            seed + _seed_offset("open-primary"),
            active_mask=turnout["open_primary_mask"],
            strength=strategy_strength,
        )
        primary_context = _build_stage_context(electorate, candidates)
        general_strategy = lambda stage_electorate, stage_candidates, _meta: _build_stage_strategy(
            strategy_name,
            system_spec.family,
            seed + _seed_offset("general"),
            active_mask=turnout["general_mask"],
            strength=strategy_strength,
        )
        general_context = lambda stage_electorate, stage_candidates, _meta: _build_stage_context(
            stage_electorate,
            stage_candidates,
        )
        primary_result = run_open_primary_top_k(
            electorate,
            candidates,
            general_system=system_spec.build(),
            top_k=primary_spec.top_k or 4,
            primary_strategy=primary_strategy,
            primary_context=primary_context,
            general_strategy=general_strategy,
            general_context=general_context,
        )
        result = primary_result.general_result
        final_candidates = CandidateSet(
            primary_result.primary_result.finalist_positions.copy(),
            list(primary_result.primary_result.finalist_labels),
        )
        metrics = compute_metrics(result, electorate, final_candidates)
        groups = compute_group_metrics(result, electorate, final_candidates)
        pipeline_meta = {
            "pipeline_type": "open_primary_top_k",
            "n_primary_voters": int(primary_result.primary_result.n_primary_voters),
            "distance_to_median_delta": metrics.distance_to_median - baseline_metrics.distance_to_median,
            "finalist_labels": list(primary_result.primary_result.finalist_labels),
        }
    else:
        raise ValueError(f"Unknown primary kind: {primary_spec.kind}")

    group_lookup = _group_metric_lookup(groups)
    baseline_group_lookup = _group_metric_lookup(baseline_groups)
    winner_index = result.winner_indices[0]
    winner_label = final_candidates.labels[winner_index]
    winner_position = result.outcome_position.copy()

    row = {
        "electorate": electorate_name,
        "primary": primary_name,
        "turnout": turnout_name,
        "strategy": strategy_name,
        "ratio": ratio_label,
        "majority_share": majority_share,
        "minority_share": 1.0 - majority_share,
        "turnout_strength": 1.0 if turnout_strength is None else float(np.clip(turnout_strength, 0.0, 1.0)),
        "strategy_strength": 1.0 if strategy_strength is None else float(np.clip(strategy_strength, 0.0, 1.0)),
        "candidate_slate": slate_name,
        "system": system_name,
        "winner": winner_label,
        "winner_party": _winner_party_name({"winner_position": winner_position}, party_positions),
        "winner_position": winner_position,
        "distance_to_median": metrics.distance_to_median,
        "distance_to_mean": metrics.distance_to_mean,
        "mean_voter_distance": metrics.mean_voter_distance,
        "majority_satisfaction": metrics.majority_satisfaction,
        "aggregate_welfare": -metrics.mean_voter_distance,
        "majority_group_welfare": group_lookup["Majority group"]["welfare"],
        "minority_group_welfare": group_lookup["Minority group"]["welfare"],
        "majority_group_distance": group_lookup["Majority group"]["mean_voter_distance"],
        "minority_group_distance": group_lookup["Minority group"]["mean_voter_distance"],
        "group_welfare_gap": (
            group_lookup["Minority group"]["welfare"] - group_lookup["Majority group"]["welfare"]
        ),
        "minority_minus_aggregate": (
            group_lookup["Minority group"]["welfare"] - (-metrics.mean_voter_distance)
        ),
        "baseline_distance_to_median": baseline_metrics.distance_to_median,
        "baseline_aggregate_welfare": -baseline_metrics.mean_voter_distance,
        "baseline_majority_group_welfare": baseline_group_lookup["Majority group"]["welfare"],
        "baseline_minority_group_welfare": baseline_group_lookup["Minority group"]["welfare"],
        "aggregate_welfare_delta_vs_no_primary": (
            -metrics.mean_voter_distance + baseline_metrics.mean_voter_distance
        ),
        "majority_group_welfare_delta_vs_no_primary": (
            group_lookup["Majority group"]["welfare"] - baseline_group_lookup["Majority group"]["welfare"]
        ),
        "minority_group_welfare_delta_vs_no_primary": (
            group_lookup["Minority group"]["welfare"] - baseline_group_lookup["Minority group"]["welfare"]
        ),
        "general_turnout_rate": float(turnout["general_mask"].mean()),
        "open_primary_turnout_rate": float(turnout["open_primary_mask"].mean()),
    }
    row.update(pipeline_meta)
    return row


def run_full_space_grid(
    electorate_names: list[str] | None = None,
    primary_names: list[str] | None = None,
    turnout_names: list[str] | None = None,
    strategy_names: list[str] | None = None,
    ratio_names: list[str] | None = None,
    slate_names: list[str] | None = None,
    system_names: list[str] | None = None,
    seed: int = 42,
    n_voters: int = 1800,
) -> pd.DataFrame:
    """Evaluate the full cross-product, or a user-supplied subset, of the design space."""
    electorate_names = ELECTORATE_ORDER if electorate_names is None else electorate_names
    primary_names = PRIMARY_ORDER if primary_names is None else primary_names
    turnout_names = TURNOUT_ORDER if turnout_names is None else turnout_names
    strategy_names = STRATEGY_ORDER if strategy_names is None else strategy_names
    ratio_names = RATIO_ORDER if ratio_names is None else ratio_names
    slate_names = SLATE_ORDER if slate_names is None else slate_names
    system_names = SYSTEM_ORDER if system_names is None else system_names

    rows = []
    for idx, combo in enumerate(
        itertools.product(
            electorate_names,
            primary_names,
            turnout_names,
            strategy_names,
            ratio_names,
            slate_names,
            system_names,
        )
    ):
        rows.append(
            run_design_case(
                electorate_name=combo[0],
                primary_name=combo[1],
                turnout_name=combo[2],
                strategy_name=combo[3],
                ratio_name=combo[4],
                slate_name=combo[5],
                system_name=combo[6],
                seed=seed + 17 * idx,
                n_voters=n_voters,
            )
        )
    return pd.DataFrame(rows)


def run_mixed_lhs_grid(
    design_df: pd.DataFrame,
    system_names: list[str] | None = None,
    seed: int = 20260407,
    n_voters: int = 1800,
) -> pd.DataFrame:
    """
    Evaluate a mixed Latin-hypercube design and expand each sampled case across systems.

    The sampled design carries the non-system axes. The returned frame keeps
    one row per sampled case per system so plurality can still serve as a
    within-case baseline.
    """
    system_names = SYSTEM_ORDER if system_names is None else system_names
    rows = []
    for _, sample in design_df.iterrows():
        case_seed = seed + 31 * int(sample["case_id"])
        for system_name in system_names:
            row = run_design_case(
                electorate_name=str(sample["electorate"]),
                primary_name=str(sample["primary"]),
                turnout_name=str(sample["turnout"]),
                strategy_name=str(sample["strategy"]),
                ratio_name=None,
                slate_name=str(sample["candidate_slate"]),
                system_name=system_name,
                seed=case_seed + 7 * SYSTEM_ORDER.index(system_name),
                n_voters=n_voters,
                majority_share=float(sample["majority_share"]),
                turnout_strength=float(sample["turnout_strength"]),
                strategy_strength=float(sample["strategy_strength"]),
            )
            row["case_id"] = int(sample["case_id"])
            rows.append(row)
    return pd.DataFrame(rows)
