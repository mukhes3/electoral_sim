"""Helpers for studying polarization under repeated elections."""
from __future__ import annotations

from dataclasses import dataclass
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from electoral_sim.ballots import BallotProfile
from electoral_sim.candidates import CandidateSet
from electoral_sim.electorate import Electorate, gaussian_mixture_electorate
from electoral_sim.fractional import FractionalBallotDiscrete
from electoral_sim.metrics import compute_metrics
from electoral_sim.strategies import StrategyModel
from electoral_sim.systems import (
    ApprovalVoting,
    CondorcetSchulze,
    InstantRunoff,
    Plurality,
    ScoreVoting,
)
from electoral_sim.utils import plot_election_result, plot_electorate


ELECTORATE_ORDER = [
    "Two blocs",
    "Bridge conflict",
    "Asymmetric resentment",
]
RATIO_ORDER = [
    "Original",
    "70:30",
    "50:50",
]
CANDIDATE_ORDER = [
    "Centrist ladder",
    "Polarized elites",
    "Asymmetric insurgency",
]
VOTER_MODEL_ORDER = [
    "Consensus pull",
    "Backlash",
    "Sorting pressure",
]
CANDIDATE_MODEL_ORDER = [
    "Static candidates",
    "Broad coalition chase",
    "Base reinforcement",
]
SYSTEM_ORDER = [
    "Plurality",
    "IRV",
    "Approval",
    "Score",
    "Condorcet",
    "Fractional (sigma=0.3)",
    "Fractional (sigma=1.0)",
]
ORACLE_ORDER = [
    "Geometric median oracle",
    "Depolarization oracle",
]

DISPLAY_LABELS = {
    "Two blocs": "Two\nblocs",
    "Bridge conflict": "Bridge\nconflict",
    "Asymmetric resentment": "Asymmetric\nresentment",
    "Centrist ladder": "Centrist\nladder",
    "Polarized elites": "Polarized\nelites",
    "Asymmetric insurgency": "Asymmetric\ninsurgency",
    "Original": "Original\nbalance",
    "70:30": "70:30\nbalance",
    "50:50": "50:50\nbalance",
    "Consensus pull": "Consensus\npull",
    "Backlash": "Backlash",
    "Sorting pressure": "Sorting\npressure",
    "Static candidates": "Static\ncandidates",
    "Broad coalition chase": "Broad coalition\nchase",
    "Base reinforcement": "Base\nreinforcement",
    "Fractional": "Fractional\n(sigma=0.3)",
    "Fractional (sigma=0.3)": "Fractional\n(sigma=0.3)",
    "Fractional (sigma=1.0)": "Fractional\n(sigma=1.0)",
}

RATIO_TARGETS = {
    "Original": None,
    "70:30": (0.70, 0.30),
    "50:50": (0.50, 0.50),
}

ELECTORATE_SPECS = {
    "Two blocs": [
        {
            "weight": 0.48,
            "mean": [0.24, 0.57],
            "cov": [[0.010, 0.002], [0.002, 0.012]],
            "group": "Left bloc",
            "camp": "Camp A",
        },
        {
            "weight": 0.48,
            "mean": [0.76, 0.43],
            "cov": [[0.010, -0.002], [-0.002, 0.012]],
            "group": "Right bloc",
            "camp": "Camp B",
        },
        {
            "weight": 0.04,
            "mean": [0.50, 0.50],
            "cov": [[0.005, 0.000], [0.000, 0.005]],
            "group": "Bridge voters",
            "camp": None,
        },
    ],
    "Bridge conflict": [
        {
            "weight": 0.34,
            "mean": [0.22, 0.58],
            "cov": [[0.010, 0.002], [0.002, 0.011]],
            "group": "Left bloc",
            "camp": "Camp A",
        },
        {
            "weight": 0.32,
            "mean": [0.52, 0.52],
            "cov": [[0.008, 0.001], [0.001, 0.009]],
            "group": "Bridge voters",
            "camp": None,
        },
        {
            "weight": 0.34,
            "mean": [0.80, 0.42],
            "cov": [[0.010, -0.002], [-0.002, 0.011]],
            "group": "Right bloc",
            "camp": "Camp B",
        },
    ],
    "Asymmetric resentment": [
        {
            "weight": 0.42,
            "mean": [0.20, 0.60],
            "cov": [[0.009, 0.002], [0.002, 0.011]],
            "group": "Left bloc",
            "camp": "Camp A",
        },
        {
            "weight": 0.23,
            "mean": [0.42, 0.52],
            "cov": [[0.008, 0.001], [0.001, 0.009]],
            "group": "Center-left",
            "camp": "Camp A",
        },
        {
            "weight": 0.20,
            "mean": [0.68, 0.46],
            "cov": [[0.010, -0.001], [-0.001, 0.010]],
            "group": "Center-right",
            "camp": "Camp B",
        },
        {
            "weight": 0.15,
            "mean": [0.90, 0.34],
            "cov": [[0.007, -0.001], [-0.001, 0.008]],
            "group": "Right edge",
            "camp": "Camp B",
        },
    ],
}

CANDIDATE_SPECS = {
    "Centrist ladder": np.array(
        [
            [0.16, 0.60],
            [0.32, 0.55],
            [0.46, 0.50],
            [0.58, 0.50],
            [0.72, 0.45],
            [0.86, 0.40],
        ]
    ),
    "Polarized elites": np.array(
        [
            [0.08, 0.68],
            [0.22, 0.58],
            [0.38, 0.48],
            [0.62, 0.52],
            [0.78, 0.42],
            [0.92, 0.32],
        ]
    ),
    "Asymmetric insurgency": np.array(
        [
            [0.18, 0.60],
            [0.34, 0.54],
            [0.48, 0.50],
            [0.66, 0.50],
            [0.82, 0.45],
            [0.96, 0.24],
        ]
    ),
}

CANDIDATE_LABELS = {
    "Centrist ladder": ["L-Base", "L-Bridge", "L-Center", "R-Center", "R-Bridge", "R-Base"],
    "Polarized elites": ["L-Edge", "L-Base", "L-Moderate", "R-Moderate", "R-Base", "R-Edge"],
    "Asymmetric insurgency": ["L-Base", "L-Bridge", "L-Center", "R-Center", "R-Bridge", "R-Insurgent"],
}


@dataclass(frozen=True)
class VoterDynamicsSpec:
    social_pull: float
    winner_pull: float
    backlash_pull: float
    backlash_threshold: float
    noise_scale: float = 0.0


@dataclass(frozen=True)
class CandidateDynamicsSpec:
    supporter_pull: float
    electorate_pull: float
    differentiation_pull: float


VOTER_DYNAMICS_SPECS = {
    "Consensus pull": VoterDynamicsSpec(
        social_pull=0.08,
        winner_pull=0.06,
        backlash_pull=0.00,
        backlash_threshold=0.30,
        noise_scale=0.002,
    ),
    "Backlash": VoterDynamicsSpec(
        social_pull=0.03,
        winner_pull=0.02,
        backlash_pull=0.08,
        backlash_threshold=0.16,
        noise_scale=0.003,
    ),
    "Sorting pressure": VoterDynamicsSpec(
        social_pull=0.02,
        winner_pull=0.01,
        backlash_pull=0.11,
        backlash_threshold=0.12,
        noise_scale=0.004,
    ),
}

CANDIDATE_DYNAMICS_SPECS = {
    "Static candidates": CandidateDynamicsSpec(
        supporter_pull=0.00,
        electorate_pull=0.00,
        differentiation_pull=0.00,
    ),
    "Broad coalition chase": CandidateDynamicsSpec(
        supporter_pull=0.12,
        electorate_pull=0.08,
        differentiation_pull=0.02,
    ),
    "Base reinforcement": CandidateDynamicsSpec(
        supporter_pull=0.18,
        electorate_pull=0.01,
        differentiation_pull=0.06,
    ),
}


def polarization_helper_overview() -> pd.DataFrame:
    """Small reference table for the repeated-election polarization helpers."""
    return pd.DataFrame(
        [
            {
                "component": "Electorate profiles",
                "options": ", ".join(ELECTORATE_ORDER),
                "purpose": "Starting voter landscapes for repeated-election experiments.",
            },
            {
                "component": "Candidate slates",
                "options": ", ".join(CANDIDATE_ORDER),
                "purpose": "Initial elite configurations before strategic adaptation.",
            },
            {
                "component": "Camp balance",
                "options": ", ".join(RATIO_ORDER),
                "purpose": "Relative size of the two opposing camps; neutral bridge groups stay fixed when present.",
            },
            {
                "component": "Voter dynamics",
                "options": ", ".join(VOTER_MODEL_ORDER),
                "purpose": "How voters move after each election under convergence or backlash.",
            },
            {
                "component": "Candidate dynamics",
                "options": ", ".join(CANDIDATE_MODEL_ORDER),
                "purpose": "How candidates respond to supporters, the electorate center, and rival pressure.",
            },
            {
                "component": "Polarization metrics",
                "options": "dispersion, pairwise distance, PC1 bimodality, candidate spread, group gap, camp displacement asymmetry",
                "purpose": "Summary metrics tracked over time alongside election outcomes.",
            },
            {
                "component": "Oracle benchmarks",
                "options": ", ".join(ORACLE_ORDER),
                "purpose": "Unconstrained outcome benchmarks for geometric-median targeting versus one-step depolarization.",
            },
        ]
    )


def _ratio_adjusted_components(name: str, ratio_name: str) -> list[dict]:
    if ratio_name not in RATIO_TARGETS:
        raise ValueError(f"Unknown camp-balance setting: {ratio_name}")
    components = [dict(component) for component in ELECTORATE_SPECS[name]]
    target = RATIO_TARGETS[ratio_name]
    if target is None:
        return components

    fixed_weight = sum(component["weight"] for component in components if component.get("camp") is None)
    adjustable_total = max(0.0, 1.0 - fixed_weight)

    for camp_name, target_share in zip(("Camp A", "Camp B"), target):
        camp_components = [component for component in components if component.get("camp") == camp_name]
        current_weight = sum(component["weight"] for component in camp_components)
        if current_weight <= 0.0:
            continue
        scale = (adjustable_total * target_share) / current_weight
        for component in camp_components:
            component["weight"] *= scale

    return components


def _camp_group_names(name: str) -> dict[str, list[str]]:
    camp_groups = {"Camp A": [], "Camp B": []}
    for component in ELECTORATE_SPECS[name]:
        camp_name = component.get("camp")
        if camp_name in camp_groups and component.get("group") is not None:
            camp_groups[camp_name].append(str(component["group"]))
    return camp_groups


def build_polarization_electorate(
    name: str,
    seed: int,
    dim_names: list[str] | None = None,
    n_voters: int = 1400,
    ratio_name: str = "Original",
) -> Electorate:
    """Construct one of the repeated-election electorate profiles."""
    if name not in ELECTORATE_SPECS:
        raise ValueError(f"Unknown electorate profile: {name}")
    rng = np.random.default_rng(seed)
    return gaussian_mixture_electorate(
        n_voters=n_voters,
        components=_ratio_adjusted_components(name, ratio_name),
        rng=rng,
        dim_names=dim_names or ["economic", "social"],
    )


def build_polarization_candidates(name: str) -> CandidateSet:
    """Construct one of the candidate slates used for dynamic experiments."""
    if name not in CANDIDATE_SPECS:
        raise ValueError(f"Unknown candidate profile: {name}")
    return CandidateSet(
        positions=CANDIDATE_SPECS[name].copy(),
        labels=list(CANDIDATE_LABELS[name]),
    )


def build_voter_dynamics(name: str) -> VoterDynamicsSpec:
    """Return one of the built-in voter update models."""
    if name not in VOTER_DYNAMICS_SPECS:
        raise ValueError(f"Unknown voter dynamics model: {name}")
    return VOTER_DYNAMICS_SPECS[name]


def build_candidate_dynamics(name: str) -> CandidateDynamicsSpec:
    """Return one of the built-in candidate update models."""
    if name not in CANDIDATE_DYNAMICS_SPECS:
        raise ValueError(f"Unknown candidate dynamics model: {name}")
    return CANDIDATE_DYNAMICS_SPECS[name]


def fractional_sigma_name(sigma: float) -> str:
    """Notebook-facing display name for a fractional system with a chosen sigma."""
    return f"Fractional (sigma={sigma:.2f})"


def parse_fractional_sigma(name: str) -> float | None:
    """Extract sigma from a fractional system display name when present."""
    if name == "Fractional":
        return 0.3
    match = re.fullmatch(r"Fractional \(sigma=([0-9]*\.?[0-9]+)\)", name)
    if match is None:
        return None
    return float(match.group(1))


def build_system(name: str):
    """Construct a notebook-facing electoral system by display name."""
    if name == "Plurality":
        return Plurality()
    if name == "IRV":
        return InstantRunoff()
    if name == "Approval":
        return ApprovalVoting()
    if name == "Score":
        return ScoreVoting()
    if name == "Condorcet":
        return CondorcetSchulze()
    sigma = parse_fractional_sigma(name)
    if sigma is not None:
        return FractionalBallotDiscrete(sigma=sigma)
    raise ValueError(f"Unknown system name: {name}")


def _copy_electorate(electorate: Electorate) -> Electorate:
    return Electorate(
        preferences=electorate.preferences.copy(),
        dim_names=list(electorate.dim_names) if electorate.dim_names is not None else None,
        group_ids=None if electorate.group_ids is None else electorate.group_ids.copy(),
        group_names=None if electorate.group_names is None else dict(electorate.group_names),
    )


def _copy_candidates(candidates: CandidateSet) -> CandidateSet:
    return CandidateSet(
        positions=candidates.positions.copy(),
        labels=list(candidates.labels),
    )


def _mean_pairwise_distance(points: np.ndarray, sample_size: int, rng: np.random.Generator) -> float:
    points = np.asarray(points, dtype=float)
    n_points = len(points)
    if n_points < 2:
        return 0.0

    max_pairs = n_points * (n_points - 1) // 2
    if max_pairs <= sample_size:
        dists = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
        upper = dists[np.triu_indices(n_points, k=1)]
        return float(upper.mean()) if len(upper) else 0.0

    left = rng.integers(0, n_points, size=sample_size)
    right = rng.integers(0, n_points, size=sample_size)
    unequal = left != right
    if not unequal.any():
        return 0.0
    sampled = np.linalg.norm(points[left[unequal]] - points[right[unequal]], axis=1)
    return float(sampled.mean())


def _pc1_scores(preferences: np.ndarray) -> np.ndarray:
    if preferences.shape[1] == 1:
        return preferences[:, 0]
    centered = preferences - preferences.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    return centered @ vh[0]


def _group_mask_by_name(electorate: Electorate, group_names: list[str]) -> np.ndarray:
    if electorate.group_ids is None or electorate.group_names is None:
        return np.zeros(electorate.n_voters, dtype=bool)
    target_ids = [
        group_id
        for group_id, group_name in electorate.group_names.items()
        if group_name in set(group_names)
    ]
    if not target_ids:
        return np.zeros(electorate.n_voters, dtype=bool)
    return np.isin(electorate.group_ids, np.asarray(target_ids, dtype=int))


def compute_camp_asymmetry_metrics(
    electorate: Electorate,
    baseline_electorate: Electorate,
    electorate_name: str | None,
) -> dict[str, float]:
    """Track whether one broad camp moves much more than the other over time."""
    empty_metrics = {
        "camp_a_displacement": float("nan"),
        "camp_b_displacement": float("nan"),
        "majority_camp_displacement": float("nan"),
        "minority_camp_displacement": float("nan"),
        "displacement_asymmetry_ratio": float("nan"),
        "normalized_displacement_asymmetry": float("nan"),
        "coalition_gap": float("nan"),
        "coalition_gap_change": float("nan"),
        "coalition_midpoint_drift": float("nan"),
        "majority_share": float("nan"),
    }
    if electorate_name is None or not electorate.has_groups or not baseline_electorate.has_groups:
        return empty_metrics

    camp_groups = _camp_group_names(electorate_name)
    if not camp_groups["Camp A"] or not camp_groups["Camp B"]:
        return empty_metrics

    baseline_masks = {
        camp_name: _group_mask_by_name(baseline_electorate, group_names)
        for camp_name, group_names in camp_groups.items()
    }
    current_masks = {
        camp_name: _group_mask_by_name(electorate, group_names)
        for camp_name, group_names in camp_groups.items()
    }
    if not baseline_masks["Camp A"].any() or not baseline_masks["Camp B"].any():
        return empty_metrics
    if not current_masks["Camp A"].any() or not current_masks["Camp B"].any():
        return empty_metrics

    baseline_centers = {
        camp_name: baseline_electorate.preferences[mask].mean(axis=0)
        for camp_name, mask in baseline_masks.items()
    }
    current_centers = {
        camp_name: electorate.preferences[mask].mean(axis=0)
        for camp_name, mask in current_masks.items()
    }
    displacement = {
        camp_name: float(np.linalg.norm(current_centers[camp_name] - baseline_centers[camp_name]))
        for camp_name in ("Camp A", "Camp B")
    }
    counts = {
        camp_name: int(mask.sum())
        for camp_name, mask in baseline_masks.items()
    }
    majority_camp = "Camp A" if counts["Camp A"] >= counts["Camp B"] else "Camp B"
    minority_camp = "Camp B" if majority_camp == "Camp A" else "Camp A"
    eps = 1e-9

    baseline_gap = float(np.linalg.norm(baseline_centers["Camp A"] - baseline_centers["Camp B"]))
    current_gap = float(np.linalg.norm(current_centers["Camp A"] - current_centers["Camp B"]))
    baseline_midpoint = 0.5 * (baseline_centers["Camp A"] + baseline_centers["Camp B"])
    current_midpoint = 0.5 * (current_centers["Camp A"] + current_centers["Camp B"])

    majority_displacement = displacement[majority_camp]
    minority_displacement = displacement[minority_camp]
    return {
        "camp_a_displacement": displacement["Camp A"],
        "camp_b_displacement": displacement["Camp B"],
        "majority_camp_displacement": majority_displacement,
        "minority_camp_displacement": minority_displacement,
        "displacement_asymmetry_ratio": minority_displacement / (majority_displacement + eps),
        "normalized_displacement_asymmetry": (
            (minority_displacement - majority_displacement)
            / (minority_displacement + majority_displacement + eps)
        ),
        "coalition_gap": current_gap,
        "coalition_gap_change": current_gap - baseline_gap,
        "coalition_midpoint_drift": float(np.linalg.norm(current_midpoint - baseline_midpoint)),
        "majority_share": counts[majority_camp] / max(1, counts["Camp A"] + counts["Camp B"]),
    }


def compute_polarization_metrics(
    electorate: Electorate,
    candidates: CandidateSet | None = None,
    winner_position: np.ndarray | None = None,
    sample_size: int = 6000,
    seed: int = 0,
) -> dict[str, float]:
    """Compute a compact set of polarization summaries for one time step."""
    rng = np.random.default_rng(seed)
    center = electorate.mean()
    voter_dists = np.linalg.norm(electorate.preferences - center, axis=1)
    pc1 = _pc1_scores(electorate.preferences)
    split = np.median(pc1)
    left = pc1[pc1 <= split]
    right = pc1[pc1 > split]
    if len(left) and len(right):
        between_gap = abs(float(right.mean()) - float(left.mean()))
        within_spread = 0.5 * (float(left.std()) + float(right.std()))
        bimodality = between_gap / max(within_spread, 1e-9)
    else:
        bimodality = 0.0

    metrics = {
        "voter_dispersion": float(voter_dists.mean()),
        "voter_pairwise_distance": _mean_pairwise_distance(
            electorate.preferences,
            sample_size=sample_size,
            rng=rng,
        ),
        "voter_p90_distance": float(np.quantile(voter_dists, 0.90)),
        "pc1_bimodality": float(bimodality),
        "group_center_gap": 0.0,
    }

    if electorate.has_groups:
        group_centers = []
        for mask in electorate.group_indices().values():
            if mask.any():
                group_centers.append(electorate.preferences[mask].mean(axis=0))
        if len(group_centers) >= 2:
            metrics["group_center_gap"] = _mean_pairwise_distance(
                np.vstack(group_centers),
                sample_size=len(group_centers) * len(group_centers),
                rng=rng,
            )

    if candidates is not None:
        metrics["candidate_pairwise_distance"] = _mean_pairwise_distance(
            candidates.positions,
            sample_size=candidates.n_candidates * max(candidates.n_candidates - 1, 1),
            rng=rng,
        )
    else:
        metrics["candidate_pairwise_distance"] = float("nan")

    if winner_position is not None:
        metrics["winner_to_center_distance"] = float(np.linalg.norm(np.asarray(winner_position) - center))
    else:
        metrics["winner_to_center_distance"] = float("nan")

    return metrics


def compute_voter_variance(electorate: Electorate) -> float:
    """Average squared distance from voters to the electorate mean."""
    center = electorate.mean()
    return float(np.mean(np.sum((electorate.preferences - center) ** 2, axis=1)))


def compute_next_step_polarization(
    electorate: Electorate,
    winner_position: np.ndarray,
    voter_dynamics: str | VoterDynamicsSpec = "Backlash",
) -> float:
    """
    One-step polarization objective used by the depolarization oracle.

    The objective uses voter variance after one deterministic update, which
    gives a smooth and inexpensive polarization proxy for the oracle search.
    """
    dynamics = (
        build_voter_dynamics(voter_dynamics)
        if isinstance(voter_dynamics, str)
        else voter_dynamics
    )
    deterministic = VoterDynamicsSpec(
        social_pull=dynamics.social_pull,
        winner_pull=dynamics.winner_pull,
        backlash_pull=dynamics.backlash_pull,
        backlash_threshold=dynamics.backlash_threshold,
        noise_scale=0.0,
    )
    advanced = advance_voters(
        electorate,
        winner_position=np.asarray(winner_position, dtype=float),
        dynamics=deterministic,
        seed=0,
    )
    return compute_voter_variance(advanced)


def choose_oracle_outcome(
    electorate: Electorate,
    oracle_name: str,
    voter_dynamics: str | VoterDynamicsSpec = "Backlash",
    tol: float = 1e-7,
) -> tuple[np.ndarray, float]:
    """Choose an unconstrained benchmark outcome for one election round."""
    if oracle_name == "Geometric median oracle":
        outcome = np.asarray(electorate.geometric_median(), dtype=float)
        value = float(np.linalg.norm(outcome - electorate.geometric_median()))
        return np.clip(outcome, 0.0, 1.0), value

    if oracle_name != "Depolarization oracle":
        raise ValueError(f"Unknown oracle_name: {oracle_name}")

    center = electorate.mean()
    median = np.asarray(electorate.geometric_median(), dtype=float)
    corners = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ],
        dtype=float,
    )
    initial_points = [center, median, *list(corners[: electorate.n_dims + 2])]
    bounds = [(0.0, 1.0)] * electorate.n_dims

    def objective(point: np.ndarray) -> float:
        return compute_next_step_polarization(
            electorate,
            point,
            voter_dynamics=voter_dynamics,
        )

    best_point = np.clip(median, 0.0, 1.0)
    best_value = float(objective(best_point))
    for point in initial_points:
        start = np.clip(np.asarray(point, dtype=float), 0.0, 1.0)
        result = minimize(objective, x0=start, bounds=bounds, tol=tol)
        candidate = np.clip(
            np.asarray(result.x if result.success else start, dtype=float),
            0.0,
            1.0,
        )
        value = float(objective(candidate))
        if value < best_value:
            best_point = candidate
            best_value = value
    return best_point, best_value


def advance_voters(
    electorate: Electorate,
    winner_position: np.ndarray,
    dynamics: VoterDynamicsSpec,
    seed: int = 0,
) -> Electorate:
    """Move voters after an election using convergence, backlash, and noise."""
    rng = np.random.default_rng(seed)
    prefs = electorate.preferences.copy()
    winner_position = np.asarray(winner_position, dtype=float)

    center = prefs.mean(axis=0)
    to_center = center - prefs
    to_winner = winner_position - prefs
    winner_dist = np.linalg.norm(to_winner, axis=1)
    backlash_scale = np.clip(
        (winner_dist - dynamics.backlash_threshold) / max(1e-9, 1.0 - dynamics.backlash_threshold),
        0.0,
        1.0,
    )

    updated = (
        prefs
        + dynamics.social_pull * to_center
        + dynamics.winner_pull * to_winner
        - dynamics.backlash_pull * backlash_scale[:, None] * to_winner
    )

    if dynamics.noise_scale > 0.0:
        updated += rng.normal(scale=dynamics.noise_scale, size=updated.shape)

    return Electorate(
        preferences=np.clip(updated, 0.0, 1.0),
        dim_names=list(electorate.dim_names) if electorate.dim_names is not None else None,
        group_ids=None if electorate.group_ids is None else electorate.group_ids.copy(),
        group_names=None if electorate.group_names is None else dict(electorate.group_names),
    )


def advance_candidates(
    electorate: Electorate,
    candidates: CandidateSet,
    dynamics: CandidateDynamicsSpec,
) -> CandidateSet:
    """Move candidates toward supporters, the center, and away from nearby rivals."""
    if (
        dynamics.supporter_pull == 0.0
        and dynamics.electorate_pull == 0.0
        and dynamics.differentiation_pull == 0.0
    ):
        return _copy_candidates(candidates)

    positions = candidates.positions.copy()
    electorate_center = electorate.mean()
    distances = np.linalg.norm(
        electorate.preferences[:, None, :] - positions[None, :, :],
        axis=2,
    )
    supporter_assignment = distances.argmin(axis=1)

    for candidate_idx in range(candidates.n_candidates):
        current = positions[candidate_idx]
        supporter_mask = supporter_assignment == candidate_idx
        if supporter_mask.any():
            supporter_center = electorate.preferences[supporter_mask].mean(axis=0)
        else:
            supporter_center = electorate_center

        step = dynamics.supporter_pull * (supporter_center - current)
        step += dynamics.electorate_pull * (electorate_center - current)

        if candidates.n_candidates > 1 and dynamics.differentiation_pull > 0.0:
            rival_distances = np.linalg.norm(positions - current, axis=1)
            rival_distances[candidate_idx] = np.inf
            nearest_idx = int(rival_distances.argmin())
            repulsion = current - positions[nearest_idx]
            repulsion_norm = np.linalg.norm(repulsion)
            if repulsion_norm > 1e-9:
                step += dynamics.differentiation_pull * (repulsion / repulsion_norm)

        positions[candidate_idx] = np.clip(current + step, 0.0, 1.0)

    return CandidateSet(positions=positions, labels=list(candidates.labels))


def run_polarization_trajectory(
    electorate: Electorate,
    candidates: CandidateSet,
    system_names: list[str] | None = None,
    n_steps: int = 12,
    voter_dynamics: str | VoterDynamicsSpec = "Backlash",
    candidate_dynamics: str | CandidateDynamicsSpec = "Broad coalition chase",
    approval_threshold: float | None = None,
    strategy: StrategyModel | None = None,
    electorate_name: str | None = None,
    ratio_name: str = "Original",
    seed: int = 0,
) -> pd.DataFrame:
    """Run one repeated-election trajectory per system from the same initial state."""
    system_names = SYSTEM_ORDER if system_names is None else list(system_names)
    voter_spec = (
        build_voter_dynamics(voter_dynamics)
        if isinstance(voter_dynamics, str)
        else voter_dynamics
    )
    candidate_spec = (
        build_candidate_dynamics(candidate_dynamics)
        if isinstance(candidate_dynamics, str)
        else candidate_dynamics
    )

    rows: list[dict[str, float | int | str | list[float]]] = []
    baseline_electorate = _copy_electorate(electorate)

    for system_offset, system_name in enumerate(system_names):
        system = build_system(system_name)
        current_electorate = _copy_electorate(electorate)
        current_candidates = _copy_candidates(candidates)
        system_seed = seed + 1009 * system_offset

        for step in range(n_steps):
            if strategy is None:
                ballots = BallotProfile.from_preferences(
                    current_electorate,
                    current_candidates,
                    approval_threshold=approval_threshold,
                )
            else:
                ballots = BallotProfile.from_strategy(
                    current_electorate,
                    current_candidates,
                    strategy=strategy,
                    approval_threshold=approval_threshold,
                    context=None,
                )

            result = system.run(ballots, current_candidates)
            election_metrics = compute_metrics(result, current_electorate, current_candidates)
            polarization_metrics = compute_polarization_metrics(
                current_electorate,
                current_candidates,
                winner_position=result.outcome_position,
                seed=system_seed + step,
            )
            asymmetry_metrics = compute_camp_asymmetry_metrics(
                current_electorate,
                baseline_electorate,
                electorate_name=electorate_name,
            )

            rows.append(
                {
                    "system": system_name,
                    "step": step,
                    "winner": ", ".join(current_candidates.labels[idx] for idx in result.winner_indices),
                    "outcome_position": result.outcome_position.round(4).tolist(),
                    "distance_to_median": election_metrics.distance_to_median,
                    "mean_voter_distance": election_metrics.mean_voter_distance,
                    "majority_satisfaction": election_metrics.majority_satisfaction,
                    "ratio": ratio_name,
                    **polarization_metrics,
                    **asymmetry_metrics,
                }
            )

            current_electorate = advance_voters(
                current_electorate,
                winner_position=result.outcome_position,
                dynamics=voter_spec,
                seed=system_seed + 100 + step,
            )
            current_candidates = advance_candidates(
                current_electorate,
                current_candidates,
                dynamics=candidate_spec,
            )

    return pd.DataFrame(rows)


def run_oracle_trajectory(
    electorate: Electorate,
    candidates: CandidateSet,
    oracle_names: list[str] | None = None,
    n_steps: int = 12,
    voter_dynamics: str | VoterDynamicsSpec = "Backlash",
    candidate_dynamics: str | CandidateDynamicsSpec = "Broad coalition chase",
    electorate_name: str | None = None,
    ratio_name: str = "Original",
    seed: int = 0,
) -> pd.DataFrame:
    """Run repeated-election dynamics under unconstrained oracle outcomes."""
    oracle_names = ORACLE_ORDER if oracle_names is None else list(oracle_names)
    voter_spec = (
        build_voter_dynamics(voter_dynamics)
        if isinstance(voter_dynamics, str)
        else voter_dynamics
    )
    candidate_spec = (
        build_candidate_dynamics(candidate_dynamics)
        if isinstance(candidate_dynamics, str)
        else candidate_dynamics
    )

    rows: list[dict[str, float | int | str | list[float]]] = []
    baseline_electorate = _copy_electorate(electorate)

    for oracle_offset, oracle_name in enumerate(oracle_names):
        current_electorate = _copy_electorate(electorate)
        current_candidates = _copy_candidates(candidates)
        oracle_seed = seed + 1009 * oracle_offset

        for step in range(n_steps):
            outcome_position, objective_value = choose_oracle_outcome(
                current_electorate,
                oracle_name=oracle_name,
                voter_dynamics=voter_spec,
            )
            polarization_metrics = compute_polarization_metrics(
                current_electorate,
                current_candidates,
                winner_position=outcome_position,
                seed=oracle_seed + step,
            )
            asymmetry_metrics = compute_camp_asymmetry_metrics(
                current_electorate,
                baseline_electorate,
                electorate_name=electorate_name,
            )
            rows.append(
                {
                    "system": oracle_name,
                    "step": step,
                    "winner": oracle_name,
                    "outcome_position": np.asarray(outcome_position).round(4).tolist(),
                    "distance_to_median": float(
                        np.linalg.norm(outcome_position - current_electorate.geometric_median())
                    ),
                    "mean_voter_distance": float(
                        np.linalg.norm(current_electorate.preferences - outcome_position, axis=1).mean()
                    ),
                    "majority_satisfaction": float("nan"),
                    "oracle_objective_value": float(objective_value),
                    "ratio": ratio_name,
                    **polarization_metrics,
                    **asymmetry_metrics,
                }
            )

            current_electorate = advance_voters(
                current_electorate,
                winner_position=outcome_position,
                dynamics=voter_spec,
                seed=oracle_seed + 100 + step,
            )
            current_candidates = advance_candidates(
                current_electorate,
                current_candidates,
                dynamics=candidate_spec,
            )

    return pd.DataFrame(rows)


def trace_oracle_state(
    electorate: Electorate,
    candidates: CandidateSet,
    oracle_name: str,
    n_steps: int = 12,
    voter_dynamics: str | VoterDynamicsSpec = "Backlash",
    candidate_dynamics: str | CandidateDynamicsSpec = "Broad coalition chase",
    seed: int = 0,
) -> dict[str, object]:
    """Roll one oracle benchmark forward and keep the start/end states."""
    voter_spec = (
        build_voter_dynamics(voter_dynamics)
        if isinstance(voter_dynamics, str)
        else voter_dynamics
    )
    candidate_spec = (
        build_candidate_dynamics(candidate_dynamics)
        if isinstance(candidate_dynamics, str)
        else candidate_dynamics
    )

    current_electorate = _copy_electorate(electorate)
    current_candidates = _copy_candidates(candidates)
    start_electorate = _copy_electorate(electorate)
    start_candidates = _copy_candidates(candidates)
    final_outcome = None

    for step in range(n_steps):
        final_outcome, _ = choose_oracle_outcome(
            current_electorate,
            oracle_name=oracle_name,
            voter_dynamics=voter_spec,
        )
        current_electorate = advance_voters(
            current_electorate,
            winner_position=final_outcome,
            dynamics=voter_spec,
            seed=seed + 100 + step,
        )
        current_candidates = advance_candidates(
            current_electorate,
            current_candidates,
            dynamics=candidate_spec,
        )

    return {
        "start_electorate": start_electorate,
        "start_candidates": start_candidates,
        "end_electorate": current_electorate,
        "end_candidates": current_candidates,
        "final_outcome": np.asarray(final_outcome, dtype=float) if final_outcome is not None else None,
    }


def summarize_trajectory_changes(trajectory_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize how each tracked metric changes from the first to the last round."""
    metric_columns = [
        "distance_to_median",
        "mean_voter_distance",
        "majority_satisfaction",
        "voter_dispersion",
        "voter_pairwise_distance",
        "voter_p90_distance",
        "pc1_bimodality",
        "group_center_gap",
        "candidate_pairwise_distance",
        "winner_to_center_distance",
        "camp_a_displacement",
        "camp_b_displacement",
        "majority_camp_displacement",
        "minority_camp_displacement",
        "displacement_asymmetry_ratio",
        "normalized_displacement_asymmetry",
        "coalition_gap",
        "coalition_gap_change",
        "coalition_midpoint_drift",
        "majority_share",
    ]

    rows = []
    for system, system_df in trajectory_df.groupby("system", sort=False):
        system_df = system_df.sort_values("step").reset_index(drop=True)
        start = system_df.iloc[0]
        end = system_df.iloc[-1]
        row = {
            "system": system,
            "start_winner": start["winner"],
            "end_winner": end["winner"],
            "n_steps": int(system_df["step"].max() + 1),
        }
        for metric in metric_columns:
            row[f"{metric}_start"] = float(start[metric])
            row[f"{metric}_end"] = float(end[metric])
            row[f"{metric}_delta"] = float(end[metric] - start[metric])
        rows.append(row)

    return pd.DataFrame(rows)


def run_polarization_comparison_grid(
    electorate_names: list[str],
    candidate_names: list[str],
    voter_dynamics_names: list[str],
    candidate_dynamics_names: list[str],
    ratio_names: list[str] | None = None,
    system_names: list[str] | None = None,
    n_steps: int = 12,
    n_voters: int = 1400,
    seed: int = 0,
) -> pd.DataFrame:
    """Evaluate many mechanism combinations and summarize start-to-end changes."""
    ratio_names = ["Original"] if ratio_names is None else list(ratio_names)
    system_names = SYSTEM_ORDER if system_names is None else list(system_names)
    rows: list[pd.DataFrame] = []
    case_id = 0

    for electorate_name in electorate_names:
        for ratio_name in ratio_names:
            for candidate_name in candidate_names:
                for voter_name in voter_dynamics_names:
                    for candidate_model in candidate_dynamics_names:
                        case_seed = seed + 1009 * case_id
                        electorate = build_polarization_electorate(
                            electorate_name,
                            seed=case_seed,
                            n_voters=n_voters,
                            ratio_name=ratio_name,
                        )
                        candidates = build_polarization_candidates(candidate_name)
                        trajectory = run_polarization_trajectory(
                            electorate,
                            candidates,
                            system_names=system_names,
                            n_steps=n_steps,
                            voter_dynamics=voter_name,
                            candidate_dynamics=candidate_model,
                            electorate_name=electorate_name,
                            ratio_name=ratio_name,
                            seed=case_seed,
                        )
                        summary = summarize_trajectory_changes(trajectory)
                        summary["case_id"] = case_id
                        summary["electorate"] = electorate_name
                        summary["ratio"] = ratio_name
                        summary["candidate_slate"] = candidate_name
                        summary["voter_dynamics"] = voter_name
                        summary["candidate_dynamics"] = candidate_model
                        rows.append(summary)
                        case_id += 1

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def run_polarization_trajectory_grid(
    electorate_names: list[str],
    candidate_names: list[str],
    voter_dynamics_names: list[str],
    candidate_dynamics_names: list[str],
    ratio_names: list[str] | None = None,
    system_names: list[str] | None = None,
    n_steps: int = 12,
    n_voters: int = 1400,
    seed: int = 0,
) -> pd.DataFrame:
    """Run full step-by-step trajectories across the mechanism grid."""
    ratio_names = ["Original"] if ratio_names is None else list(ratio_names)
    system_names = SYSTEM_ORDER if system_names is None else list(system_names)
    rows: list[pd.DataFrame] = []
    case_id = 0

    for electorate_name in electorate_names:
        for ratio_name in ratio_names:
            for candidate_name in candidate_names:
                for voter_name in voter_dynamics_names:
                    for candidate_model in candidate_dynamics_names:
                        case_seed = seed + 1009 * case_id
                        electorate = build_polarization_electorate(
                            electorate_name,
                            seed=case_seed,
                            n_voters=n_voters,
                            ratio_name=ratio_name,
                        )
                        candidates = build_polarization_candidates(candidate_name)
                        trajectory = run_polarization_trajectory(
                            electorate,
                            candidates,
                            system_names=system_names,
                            n_steps=n_steps,
                            voter_dynamics=voter_name,
                            candidate_dynamics=candidate_model,
                            electorate_name=electorate_name,
                            ratio_name=ratio_name,
                            seed=case_seed,
                        )
                        trajectory["case_id"] = case_id
                        trajectory["electorate"] = electorate_name
                        trajectory["ratio"] = ratio_name
                        trajectory["candidate_slate"] = candidate_name
                        trajectory["voter_dynamics"] = voter_name
                        trajectory["candidate_dynamics"] = candidate_model
                        rows.append(trajectory)
                        case_id += 1

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def compare_trajectory_to_baseline(
    trajectory_grid_df: pd.DataFrame,
    baseline_system: str = "Plurality",
    metrics: list[str] | None = None,
) -> pd.DataFrame:
    """Compute per-step differences from a baseline system within each grid case."""
    metrics = metrics or [
        "voter_pairwise_distance",
        "candidate_pairwise_distance",
        "winner_to_center_distance",
        "distance_to_median",
    ]
    baseline_columns = ["case_id", "step"] + metrics
    baseline_df = (
        trajectory_grid_df[trajectory_grid_df["system"] == baseline_system][baseline_columns]
        .rename(columns={metric: f"{metric}_baseline" for metric in metrics})
    )
    merged = trajectory_grid_df.merge(
        baseline_df,
        on=["case_id", "step"],
        how="left",
        validate="many_to_one",
    )
    for metric in metrics:
        merged[f"{metric}_vs_{baseline_system}"] = (
            merged[metric] - merged[f"{metric}_baseline"]
        )
    return merged


def plot_baseline_difference_trajectories(
    relative_df: pd.DataFrame,
    baseline_system: str = "Plurality",
    metrics: list[str] | None = None,
    systems: list[str] | None = None,
    figsize: tuple[float, float] = (11, 10),
) -> plt.Figure:
    """Plot median difference-from-baseline trajectories with IQR bands across grid cases."""
    metrics = metrics or [
        "voter_pairwise_distance_vs_Plurality",
        "candidate_pairwise_distance_vs_Plurality",
        "winner_to_center_distance_vs_Plurality",
        "distance_to_median_vs_Plurality",
    ]
    systems = systems or [name for name in SYSTEM_ORDER if name != baseline_system]
    palette = dict(zip(systems, plt.cm.Set2(np.linspace(0, 1, len(systems)))))

    fig, axes = plt.subplots(
        len(metrics),
        1,
        figsize=figsize,
        dpi=150,
        sharex=True,
    )
    axes = np.atleast_1d(axes).ravel()

    for ax, metric in zip(axes, metrics):
        for system in systems:
            system_df = relative_df[relative_df["system"] == system]
            grouped = (
                system_df.groupby("step")[metric]
                .agg(
                    q25=lambda s: float(s.quantile(0.25)),
                    median="median",
                    q75=lambda s: float(s.quantile(0.75)),
                )
                .reset_index()
                .sort_values("step")
            )
            x = grouped["step"].to_numpy(dtype=float)
            median = grouped["median"].to_numpy(dtype=float)
            lower = grouped["q25"].to_numpy(dtype=float)
            upper = grouped["q75"].to_numpy(dtype=float)
            color = palette[system]
            ax.plot(
                x,
                median,
                color=color,
                linewidth=2,
                label=DISPLAY_LABELS.get(system, system).replace("\n", " "),
            )
            ax.fill_between(
                x,
                lower,
                upper,
                color=color,
                alpha=0.18,
                linewidth=0,
            )
        ax.axhline(0.0, color="#666666", linewidth=1.0, alpha=0.7)
        ax.set_ylabel(metric.replace("_vs_", "\nvs ").replace("_", " "))
        ax.grid(alpha=0.25)

    axes[0].legend(frameon=False, ncol=min(3, len(systems)))
    axes[-1].set_xlabel("Election round")
    fig.suptitle(f"Median difference from {baseline_system} with IQR bands", y=0.995)
    fig.tight_layout()
    return fig


def plot_polarization_metric_heatmap(
    summary_df: pd.DataFrame,
    metric_delta: str,
    candidate_name: str | None = None,
    figsize: tuple[float, float] = (10, 5),
) -> plt.Figure:
    """Heatmap of average start-to-end metric change by mechanism pair and system."""
    plot_df = summary_df.copy()
    if candidate_name is not None:
        plot_df = plot_df[plot_df["candidate_slate"] == candidate_name]

    grouped = (
        plot_df.groupby(["voter_dynamics", "candidate_dynamics", "system"], as_index=False)[metric_delta]
        .mean()
    )
    grouped["mechanism_pair"] = (
        grouped["voter_dynamics"] + " | " + grouped["candidate_dynamics"]
    )
    row_order = [
        f"{voter_name} | {candidate_name_}"
        for voter_name in VOTER_MODEL_ORDER
        for candidate_name_ in CANDIDATE_MODEL_ORDER
    ]
    heatmap = (
        grouped.pivot(index="mechanism_pair", columns="system", values=metric_delta)
        .reindex(row_order)
        .reindex(columns=SYSTEM_ORDER)
    )

    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    mesh = ax.imshow(heatmap.fillna(0.0).to_numpy(), aspect="auto", cmap="coolwarm")
    xtick_labels = [DISPLAY_LABELS.get(name, name) for name in heatmap.columns]
    ax.set_xticks(range(len(heatmap.columns)), xtick_labels, rotation=0, ha="center")
    ax.set_yticks(range(len(heatmap.index)), heatmap.index)
    ax.set_title(metric_delta.replace("_", " "))
    fig.colorbar(mesh, ax=ax, label="Average start-to-end change")
    fig.tight_layout()
    return fig


def plot_polarization_tradeoff_scatter(
    system_summary: pd.DataFrame,
    x_metric: str = "mean_voter_pairwise_delta",
    y_metric: str = "mean_winner_center_delta",
    figsize: tuple[float, float] = (7.5, 6.0),
) -> plt.Figure:
    """Scatter plot showing the tradeoff between voter depolarization and winner centering."""
    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    colors = plt.cm.Set2(np.linspace(0, 1, max(len(system_summary), 1)))

    for color, (_, row) in zip(colors, system_summary.iterrows()):
        x_value = float(row[x_metric])
        y_value = float(row[y_metric])
        label = DISPLAY_LABELS.get(row["system"], row["system"]).replace("\n", " ")
        ax.scatter(x_value, y_value, s=110, color=color, edgecolor="white", linewidth=0.8)
        ax.annotate(
            label,
            (x_value, y_value),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=8,
        )

    ax.axhline(0.0, color="#666666", linewidth=1.0, alpha=0.7)
    ax.axvline(0.0, color="#666666", linewidth=1.0, alpha=0.7)
    ax.set_xlabel("Mean voter pairwise-distance change")
    ax.set_ylabel("Mean winner-to-center change")
    ax.set_title("Tradeoff between civic depolarization and winner centering")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def plot_ratio_system_heatmap(
    summary_df: pd.DataFrame,
    metric: str,
    figsize: tuple[float, float] = (8.5, 3.8),
) -> plt.Figure:
    """Heatmap of a metric averaged by camp-balance setting and electoral system."""
    grouped = (
        summary_df.groupby(["ratio", "system"], as_index=False)[metric]
        .mean()
    )
    heatmap = (
        grouped.pivot(index="ratio", columns="system", values=metric)
        .reindex(index=RATIO_ORDER)
        .reindex(columns=SYSTEM_ORDER)
    )

    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    mesh = ax.imshow(heatmap.fillna(0.0).to_numpy(), aspect="auto", cmap="coolwarm")
    ax.set_xticks(
        range(len(heatmap.columns)),
        [DISPLAY_LABELS.get(name, name) for name in heatmap.columns],
        rotation=0,
        ha="center",
    )
    ax.set_yticks(
        range(len(heatmap.index)),
        [DISPLAY_LABELS.get(name, name) for name in heatmap.index],
    )
    ax.set_title(metric.replace("_", " "))
    fig.colorbar(mesh, ax=ax, label="Average value")
    fig.tight_layout()
    return fig


def summarize_fractional_sigma_sweep(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate arbitrary fractional sigma runs into a compact sweep summary."""
    sweep_df = summary_df.copy()
    sweep_df["sigma"] = sweep_df["system"].map(parse_fractional_sigma)
    sweep_df = sweep_df[sweep_df["sigma"].notna()].copy()
    if sweep_df.empty:
        return pd.DataFrame(
            columns=[
                "sigma",
                "mean_voter_pairwise_delta",
                "mean_candidate_pairwise_delta",
                "mean_group_gap_delta",
                "mean_normalized_displacement_asymmetry_end",
                "mean_winner_center_delta",
            ]
        )

    return (
        sweep_df.groupby("sigma", as_index=False)
        .agg(
            mean_voter_pairwise_delta=("voter_pairwise_distance_delta", "mean"),
            mean_candidate_pairwise_delta=("candidate_pairwise_distance_delta", "mean"),
            mean_group_gap_delta=("group_center_gap_delta", "mean"),
            mean_normalized_displacement_asymmetry_end=("normalized_displacement_asymmetry_end", "mean"),
            mean_winner_center_delta=("winner_to_center_distance_delta", "mean"),
        )
        .sort_values("sigma")
        .reset_index(drop=True)
    )


def plot_fractional_sigma_sweep(
    sigma_summary: pd.DataFrame,
    x_metric: str = "mean_candidate_pairwise_delta",
    y_metric: str = "mean_voter_pairwise_delta",
    figsize: tuple[float, float] = (7.5, 6.0),
) -> plt.Figure:
    """Plot how a sweep over fractional sigma trades voter and candidate depolarization."""
    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    if sigma_summary.empty:
        ax.set_title("Fractional sigma sweep")
        ax.set_xlabel(x_metric.replace("_", " "))
        ax.set_ylabel(y_metric.replace("_", " "))
        fig.tight_layout()
        return fig

    sigmas = sigma_summary["sigma"].to_numpy(dtype=float)
    colors = plt.cm.viridis(np.linspace(0, 1, len(sigmas)))
    x_values = sigma_summary[x_metric].to_numpy(dtype=float)
    y_values = sigma_summary[y_metric].to_numpy(dtype=float)

    ax.plot(x_values, y_values, color="#666666", linewidth=1.5, alpha=0.7)
    for color, sigma, x_value, y_value in zip(colors, sigmas, x_values, y_values):
        ax.scatter(x_value, y_value, s=120, color=color, edgecolor="white", linewidth=0.8)
        ax.annotate(
            f"{sigma:.2f}",
            (x_value, y_value),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=8,
        )

    ax.axhline(0.0, color="#666666", linewidth=1.0, alpha=0.7)
    ax.axvline(0.0, color="#666666", linewidth=1.0, alpha=0.7)
    ax.set_xlabel(x_metric.replace("_", " "))
    ax.set_ylabel(y_metric.replace("_", " "))
    ax.set_title("Fractional sigma sweep: candidate vs voter polarization change")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def illustrate_voter_mechanisms(
    electorate: Electorate,
    winner_position: np.ndarray,
    model_names: list[str] | None = None,
    n_rounds: int = 1,
    seed: int = 0,
) -> plt.Figure:
    """Show start and end voter maps under repeated applications of each voter rule."""
    model_names = VOTER_MODEL_ORDER if model_names is None else list(model_names)
    fig, axes = plt.subplots(
        len(model_names),
        2,
        figsize=(10, 3.6 * len(model_names)),
        dpi=150,
        sharex=True,
        sharey=True,
    )
    axes = np.atleast_2d(axes)

    base_positions = electorate.preferences
    colors = electorate.group_ids if electorate.group_ids is not None else None

    for row_idx, model_name in enumerate(model_names):
        advanced = _copy_electorate(electorate)
        dynamics = build_voter_dynamics(model_name)
        for round_idx in range(n_rounds):
            advanced = advance_voters(
                advanced,
                winner_position=winner_position,
                dynamics=dynamics,
                seed=seed + 100 * row_idx + round_idx,
            )
        before_ax, after_ax = axes[row_idx]
        before_ax.scatter(
            base_positions[:, 0],
            base_positions[:, 1],
            c=colors,
            cmap="coolwarm",
            s=10,
            alpha=0.35,
        )
        before_ax.scatter(
            winner_position[0],
            winner_position[1],
            marker="*",
            s=180,
            c="black",
        )
        before_ax.set_title(f"{model_name}: start")

        after_ax.scatter(
            advanced.preferences[:, 0],
            advanced.preferences[:, 1],
            c=colors,
            cmap="coolwarm",
            s=10,
            alpha=0.35,
        )
        after_ax.scatter(
            winner_position[0],
            winner_position[1],
            marker="*",
            s=180,
            c="black",
        )
        after_ax.set_title(f"{model_name}: after {n_rounds} rounds")

        for ax in (before_ax, after_ax):
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.0)
            ax.set_xlabel(electorate.dim_names[0])
            ax.set_ylabel(electorate.dim_names[1])
            ax.grid(alpha=0.2)

    fig.tight_layout()
    return fig


def illustrate_candidate_mechanisms(
    electorate: Electorate,
    candidates: CandidateSet,
    model_names: list[str] | None = None,
    n_rounds: int = 1,
) -> plt.Figure:
    """Show start and end candidate maps under repeated applications of each candidate rule."""
    model_names = CANDIDATE_MODEL_ORDER if model_names is None else list(model_names)
    fig, axes = plt.subplots(
        len(model_names),
        2,
        figsize=(10, 3.6 * len(model_names)),
        dpi=150,
        sharex=True,
        sharey=True,
    )
    axes = np.atleast_2d(axes)
    colors = plt.cm.tab10(np.linspace(0, 1, candidates.n_candidates))

    for row_idx, model_name in enumerate(model_names):
        advanced = _copy_candidates(candidates)
        dynamics = build_candidate_dynamics(model_name)
        for _ in range(n_rounds):
            advanced = advance_candidates(
                electorate,
                advanced,
                dynamics=dynamics,
            )
        before_ax, after_ax = axes[row_idx]
        for idx, pos in enumerate(candidates.positions):
            before_ax.scatter(pos[0], pos[1], s=80, c=[colors[idx]])
            before_ax.annotate(candidates.labels[idx], pos, xytext=(4, 4), textcoords="offset points", fontsize=7)
        for idx, pos in enumerate(advanced.positions):
            after_ax.scatter(pos[0], pos[1], s=80, c=[colors[idx]])
            after_ax.annotate(advanced.labels[idx], pos, xytext=(4, 4), textcoords="offset points", fontsize=7)
        before_ax.set_title(f"{model_name}: start")
        after_ax.set_title(f"{model_name}: after {n_rounds} rounds")
        for ax in (before_ax, after_ax):
            ax.scatter(electorate.preferences[:, 0], electorate.preferences[:, 1], s=8, alpha=0.12, c="#999999")
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.0)
            ax.set_xlabel(electorate.dim_names[0])
            ax.set_ylabel(electorate.dim_names[1])
            ax.grid(alpha=0.2)

    fig.tight_layout()
    return fig


def plot_start_end_maps(
    electorate: Electorate,
    candidates: CandidateSet,
    system_names: list[str] | None = None,
    n_steps: int = 12,
    voter_dynamics: str | VoterDynamicsSpec = "Backlash",
    candidate_dynamics: str | CandidateDynamicsSpec = "Broad coalition chase",
    seed: int = 0,
) -> plt.Figure:
    """Show the shared start state and one end-state panel per system."""
    system_names = SYSTEM_ORDER if system_names is None else list(system_names)
    n_panels = len(system_names) + 1
    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=(4.6 * n_panels, 4.2),
        dpi=150,
    )
    axes = np.atleast_1d(axes).ravel()

    plot_electorate(
        electorate,
        candidates,
        title="Round 0",
        ax=axes[0],
    )

    for ax, system_name in zip(axes[1:], system_names):
        system = build_system(system_name)
        current_electorate = _copy_electorate(electorate)
        current_candidates = _copy_candidates(candidates)
        voter_spec = build_voter_dynamics(voter_dynamics) if isinstance(voter_dynamics, str) else voter_dynamics
        candidate_spec = (
            build_candidate_dynamics(candidate_dynamics)
            if isinstance(candidate_dynamics, str)
            else candidate_dynamics
        )

        result = None
        for step in range(n_steps):
            ballots = BallotProfile.from_preferences(current_electorate, current_candidates)
            result = system.run(ballots, current_candidates)
            if step < n_steps - 1:
                current_electorate = advance_voters(
                    current_electorate,
                    winner_position=result.outcome_position,
                    dynamics=voter_spec,
                    seed=seed + 100 * step,
                )
                current_candidates = advance_candidates(
                    current_electorate,
                    current_candidates,
                    dynamics=candidate_spec,
                )

        plot_election_result(
            current_electorate,
            current_candidates,
            result,
            ax=ax,
        )
        ax.set_title(DISPLAY_LABELS.get(system_name, system_name))

    fig.tight_layout()
    return fig


def plot_oracle_start_end_maps(
    state_map: dict[str, dict[str, object]],
    oracle_names: list[str] | None = None,
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """Show shared start and oracle-specific end states."""
    oracle_names = ORACLE_ORDER if oracle_names is None else list(oracle_names)
    n_rows = len(oracle_names)
    fig_width = 10.0 if figsize is None else figsize[0]
    fig_height = (3.4 * n_rows) if figsize is None else figsize[1]
    fig, axes = plt.subplots(
        n_rows,
        2,
        figsize=(fig_width, fig_height),
        dpi=150,
        sharex=True,
        sharey=True,
    )
    axes = np.atleast_2d(axes)

    for row_idx, oracle_name in enumerate(oracle_names):
        state = state_map[oracle_name]
        start_electorate = state["start_electorate"]
        start_candidates = state["start_candidates"]
        end_electorate = state["end_electorate"]
        end_candidates = state["end_candidates"]
        final_outcome = state["final_outcome"]

        for ax, electorate, candidates, title, outcome in (
            (axes[row_idx, 0], start_electorate, start_candidates, f"{oracle_name}: start", None),
            (axes[row_idx, 1], end_electorate, end_candidates, f"{oracle_name}: end", final_outcome),
        ):
            if electorate.group_ids is None:
                ax.scatter(
                    electorate.preferences[:, 0],
                    electorate.preferences[:, 1],
                    s=9,
                    alpha=0.12,
                    color="#4C78A8",
                    edgecolor="none",
                )
            else:
                palette = plt.cm.Set2(np.linspace(0, 1, len(np.unique(electorate.group_ids))))
                for color, group_id in zip(palette, np.unique(electorate.group_ids)):
                    mask = electorate.group_ids == group_id
                    ax.scatter(
                        electorate.preferences[mask, 0],
                        electorate.preferences[mask, 1],
                        s=9,
                        alpha=0.14,
                        color=color,
                        edgecolor="none",
                    )

            ax.scatter(
                candidates.positions[:, 0],
                candidates.positions[:, 1],
                s=95,
                color="#D62728",
                marker="X",
                linewidth=0.6,
                edgecolor="white",
            )
            if outcome is not None:
                ax.scatter(
                    outcome[0],
                    outcome[1],
                    s=150,
                    color="#111111",
                    marker="*",
                    linewidth=0.7,
                    edgecolor="white",
                )
            ax.set_title(title)
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.0)
            ax.set_xlabel(electorate.dim_names[0])
            ax.set_ylabel(electorate.dim_names[1])
            ax.grid(alpha=0.2)

    fig.tight_layout()
    return fig


def plot_polarization_trajectories(
    trajectory_df: pd.DataFrame,
    metrics: list[str] | None = None,
) -> plt.Figure:
    """Plot one or more polarization or welfare summaries over repeated elections."""
    metrics = metrics or [
        "voter_pairwise_distance",
        "pc1_bimodality",
        "distance_to_median",
    ]

    systems = list(dict.fromkeys(trajectory_df["system"]))
    palette = dict(zip(systems, plt.cm.Set2(np.linspace(0, 1, len(systems)))))

    fig, axes = plt.subplots(
        len(metrics),
        1,
        figsize=(10, 3.4 * len(metrics)),
        dpi=150,
        sharex=True,
    )
    axes = np.atleast_1d(axes).ravel()

    for ax, metric in zip(axes, metrics):
        for system in systems:
            system_df = trajectory_df[trajectory_df["system"] == system]
            ax.plot(
                system_df["step"],
                system_df[metric],
                marker="o",
                linewidth=2,
                markersize=4,
                color=palette[system],
                label=DISPLAY_LABELS.get(system, system).replace("\n", " "),
            )
        ax.set_ylabel(metric.replace("_", " "))
        ax.grid(alpha=0.25)

    axes[0].legend(frameon=False, ncol=min(3, len(systems)))
    axes[-1].set_xlabel("Election round")
    fig.tight_layout()
    return fig


__all__ = [
    "CANDIDATE_MODEL_ORDER",
    "CANDIDATE_ORDER",
    "DISPLAY_LABELS",
    "ELECTORATE_ORDER",
    "ORACLE_ORDER",
    "RATIO_ORDER",
    "SYSTEM_ORDER",
    "VOTER_MODEL_ORDER",
    "CandidateDynamicsSpec",
    "VoterDynamicsSpec",
    "advance_candidates",
    "advance_voters",
    "build_candidate_dynamics",
    "fractional_sigma_name",
    "build_polarization_candidates",
    "build_polarization_electorate",
    "build_system",
    "build_voter_dynamics",
    "choose_oracle_outcome",
    "compute_polarization_metrics",
    "compute_camp_asymmetry_metrics",
    "compute_next_step_polarization",
    "compute_voter_variance",
    "compare_trajectory_to_baseline",
    "illustrate_candidate_mechanisms",
    "illustrate_voter_mechanisms",
    "parse_fractional_sigma",
    "plot_baseline_difference_trajectories",
    "plot_fractional_sigma_sweep",
    "plot_oracle_start_end_maps",
    "plot_polarization_trajectories",
    "plot_polarization_metric_heatmap",
    "plot_polarization_tradeoff_scatter",
    "plot_ratio_system_heatmap",
    "plot_start_end_maps",
    "polarization_helper_overview",
    "run_polarization_comparison_grid",
    "run_polarization_trajectory",
    "run_polarization_trajectory_grid",
    "run_oracle_trajectory",
    "summarize_fractional_sigma_sweep",
    "summarize_trajectory_changes",
    "trace_oracle_state",
]
