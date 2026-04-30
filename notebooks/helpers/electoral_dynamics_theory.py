"""Helpers for paper-facing electoral dynamics notebooks."""
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from electoral_sim.ballots import BallotProfile
from electoral_sim.candidates import CandidateSet
from electoral_sim.electorate import Electorate
from electoral_sim.fractional import FractionalBallotContinuous
from electoral_sim.systems import (
    ApprovalVoting,
    CondorcetSchulze,
    InstantRunoff,
    Plurality,
    ScoreVoting,
)
from notebooks.helpers.polarization_dynamics import (
    CandidateDynamicsSpec,
    CANDIDATE_MODEL_ORDER,
    CANDIDATE_ORDER,
    ELECTORATE_ORDER,
    RATIO_ORDER,
    VOTER_MODEL_ORDER,
    advance_candidates,
    advance_voters,
    build_candidate_dynamics,
    compute_camp_asymmetry_metrics,
    build_polarization_candidates,
    build_polarization_electorate,
    build_voter_dynamics,
)


THEORY_BASELINE_SYSTEMS = [
    "Plurality",
    "IRV",
    "Approval",
    "Score",
    "Condorcet",
]

THEORY_MU0_CANDIDATE_MODELS = [
    "Static candidates",
    "Base reinforcement (mu=0)",
]

THEORY_ORACLE_ORDER = [
    "Centrality oracle",
    "Depolarization oracle",
]


@dataclass(frozen=True)
class TheoryMetricBundle:
    winner_radius: float
    mean_winner_distance: float
    voter_variance: float
    supporter_centroid_radius: float
    mean_supporter_centroid_distance: float
    candidate_variance: float


THEORY_CANDIDATE_DYNAMICS_SPECS: dict[str, CandidateDynamicsSpec] = {
    "Base reinforcement (mu=0)": CandidateDynamicsSpec(
        supporter_pull=0.18,
        electorate_pull=0.00,
        differentiation_pull=0.06,
    ),
}


def theory_helper_overview() -> pd.DataFrame:
    """Small reference table for theory-companion notebook helpers."""
    return pd.DataFrame(
        [
            {
                "component": "Winner primitive",
                "options": "winner_radius, mean_winner_distance",
                "purpose": "Directly evaluate the voter-side centrality object R_t from the paper.",
            },
            {
                "component": "Supporter primitive",
                "options": "supporter_centroids, supporter_centroid_radius",
                "purpose": "Measure the candidate-side geometry S_t using rule-specific assignment weights.",
            },
            {
                "component": "Continuous fractional family",
                "options": "Fractional Continuous (sigma=...)",
                "purpose": "Run the convex-combination fractional family used in the paper's within-family theory.",
            },
            {
                "component": "Asymmetry and theory-backed candidate runs",
                "options": "normalized_displacement_asymmetry, Base reinforcement (mu=0)",
                "purpose": "Track camp-level movement asymmetry and isolate candidate runs that satisfy the paper's mu=0 restriction.",
            },
            {
                "component": "Coverage geometry",
                "options": "minimax_center, coverage_gap",
                "purpose": "Numerically study the slate-coverage bound for winner-radius performance.",
            },
            {
                "component": "Oracle objective comparison",
                "options": ", ".join(THEORY_ORACLE_ORDER),
                "purpose": "Contrast one-step winner-centrality optimization with one-step depolarization optimization.",
            },
        ]
    )


def fractional_continuous_name(sigma: float) -> str:
    """Notebook-facing display name for a continuous fractional system."""
    return f"Fractional Continuous (sigma={sigma:.2f})"


def parse_fractional_continuous_sigma(name: str) -> float | None:
    """Extract sigma from a continuous fractional display name when present."""
    match = re.fullmatch(r"Fractional Continuous \(sigma=([0-9]*\.?[0-9]+)\)", name)
    if match is None:
        return None
    return float(match.group(1))


def build_theory_system(name: str):
    """Construct a paper-facing electoral system by display name."""
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

    sigma = parse_fractional_continuous_sigma(name)
    if sigma is not None:
        return FractionalBallotContinuous(sigma=sigma)
    raise ValueError(f"Unknown theory system name: {name}")


def build_theory_candidate_dynamics(name: str | CandidateDynamicsSpec) -> CandidateDynamicsSpec:
    """Return a built-in or theory-only candidate dynamics specification."""
    if isinstance(name, CandidateDynamicsSpec):
        return name
    if name in THEORY_CANDIDATE_DYNAMICS_SPECS:
        return THEORY_CANDIDATE_DYNAMICS_SPECS[name]
    return build_candidate_dynamics(name)


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


def _row_normalize(weights: np.ndarray, fallback_indices: np.ndarray | None = None) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    row_sums = weights.sum(axis=1, keepdims=True)
    zero_rows = row_sums.squeeze() <= 1e-12
    normalized = np.zeros_like(weights, dtype=float)
    np.divide(weights, np.maximum(row_sums, 1e-12), out=normalized, where=row_sums > 0)
    if zero_rows.any():
        if fallback_indices is None:
            fallback_indices = np.zeros(weights.shape[0], dtype=int)
        normalized[zero_rows] = 0.0
        normalized[np.arange(weights.shape[0])[zero_rows], fallback_indices[zero_rows]] = 1.0
    return normalized


def _one_hot(indices: np.ndarray, n_candidates: int) -> np.ndarray:
    indices = np.asarray(indices, dtype=int)
    weights = np.zeros((len(indices), n_candidates), dtype=float)
    weights[np.arange(len(indices)), indices] = 1.0
    return weights


def supporter_weight_matrix(
    system_name: str,
    ballots: BallotProfile,
    candidates: CandidateSet,
    system=None,
) -> np.ndarray:
    """
    Return a rule-facing voter-to-candidate assignment matrix.

    These are analysis weights rather than legal ballot translations.
    They are chosen to mirror the paper's distinction between hard and soft
    supporter assignment maps.
    """
    plurality_fallback = ballots.plurality

    sigma = parse_fractional_continuous_sigma(system_name)
    if sigma is not None:
        if system is None:
            system = FractionalBallotContinuous(sigma=sigma)
        return system.weight_matrix(ballots, candidates)

    if system_name in {"Plurality", "IRV"}:
        return _one_hot(ballots.rankings[:, 0], ballots.n_candidates)

    if system_name == "Approval":
        return _row_normalize(ballots.approvals, fallback_indices=plurality_fallback)

    if system_name == "Score":
        return _row_normalize(np.clip(ballots.scores, 0.0, None), fallback_indices=plurality_fallback)

    if system_name == "Condorcet":
        rank_points = np.zeros_like(ballots.scores)
        descending = np.arange(ballots.n_candidates - 1, -1, -1, dtype=float)
        for voter_idx, ranking in enumerate(ballots.rankings):
            rank_points[voter_idx, ranking] = descending
        return _row_normalize(rank_points, fallback_indices=plurality_fallback)

    raise ValueError(f"Unknown system_name for supporter weights: {system_name}")


def compute_supporter_centroids(
    electorate: Electorate,
    candidates: CandidateSet,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return supporter centroids and total support weight for each candidate."""
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (electorate.n_voters, candidates.n_candidates):
        raise ValueError(
            "weights must have shape "
            f"({electorate.n_voters}, {candidates.n_candidates}), got {weights.shape}."
        )

    support_mass = weights.sum(axis=0)
    centroids = candidates.positions.copy()
    for candidate_idx in range(candidates.n_candidates):
        mass = support_mass[candidate_idx]
        if mass > 1e-12:
            centroids[candidate_idx] = (
                weights[:, candidate_idx][:, None] * electorate.preferences
            ).sum(axis=0) / mass
    return centroids, support_mass


def compute_winner_radius(electorate: Electorate, outcome_position: np.ndarray) -> float:
    """Maximum voter distance to the realized winner position."""
    outcome_position = np.asarray(outcome_position, dtype=float)
    return float(np.linalg.norm(electorate.preferences - outcome_position, axis=1).max())


def compute_mean_winner_distance(electorate: Electorate, outcome_position: np.ndarray) -> float:
    """Average voter distance to the realized winner position."""
    outcome_position = np.asarray(outcome_position, dtype=float)
    return float(np.linalg.norm(electorate.preferences - outcome_position, axis=1).mean())


def compute_voter_variance(electorate: Electorate) -> float:
    """Empirical voter variance D(t)."""
    center = electorate.mean()
    return float(np.mean(np.sum((electorate.preferences - center) ** 2, axis=1)))


def compute_candidate_variance(candidates: CandidateSet) -> float:
    """Empirical candidate variance P(t)."""
    center = candidates.positions.mean(axis=0)
    return float(np.mean(np.sum((candidates.positions - center) ** 2, axis=1)))


def compute_supporter_centroid_radius(
    candidates: CandidateSet,
    supporter_centroids: np.ndarray,
) -> float:
    """Maximum candidate distance to its supporter centroid."""
    return float(np.linalg.norm(candidates.positions - supporter_centroids, axis=1).max())


def compute_mean_supporter_centroid_distance(
    candidates: CandidateSet,
    supporter_centroids: np.ndarray,
) -> float:
    """Average candidate distance to its supporter centroid."""
    return float(np.linalg.norm(candidates.positions - supporter_centroids, axis=1).mean())


def compute_theory_metrics(
    electorate: Electorate,
    candidates: CandidateSet,
    outcome_position: np.ndarray,
    weights: np.ndarray,
) -> TheoryMetricBundle:
    """Bundle the paper-facing geometric quantities for one round."""
    supporter_centroids, _ = compute_supporter_centroids(electorate, candidates, weights)
    return TheoryMetricBundle(
        winner_radius=compute_winner_radius(electorate, outcome_position),
        mean_winner_distance=compute_mean_winner_distance(electorate, outcome_position),
        voter_variance=compute_voter_variance(electorate),
        supporter_centroid_radius=compute_supporter_centroid_radius(candidates, supporter_centroids),
        mean_supporter_centroid_distance=compute_mean_supporter_centroid_distance(candidates, supporter_centroids),
        candidate_variance=compute_candidate_variance(candidates),
    )


def compute_weighted_polarization_cost(
    winner_radius: float,
    supporter_centroid_radius: float,
    alpha: float,
) -> float:
    """Weighted objective L_alpha = alpha R + (1-alpha) S."""
    alpha = float(alpha)
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must lie in [0, 1].")
    return float(alpha * winner_radius + (1.0 - alpha) * supporter_centroid_radius)


def compute_candidate_electorate_center_gap(
    electorate: Electorate,
    candidates: CandidateSet,
) -> float:
    """Distance between the candidate mean position and the electorate mean."""
    return float(np.linalg.norm(candidates.positions.mean(axis=0) - electorate.mean()))


def approximate_minimax_center(
    electorate: Electorate,
    tol: float = 1e-7,
) -> np.ndarray:
    """Numerically approximate the electorate minimax center."""
    preferences = electorate.preferences

    def objective(point: np.ndarray) -> float:
        return float(np.max(np.linalg.norm(preferences - point[None, :], axis=1)))

    result = minimize(
        objective,
        x0=electorate.mean(),
        bounds=[(0.0, 1.0)] * electorate.n_dims,
        tol=tol,
    )
    if not result.success:
        return np.clip(electorate.mean(), 0.0, 1.0)
    return np.clip(np.asarray(result.x, dtype=float), 0.0, 1.0)


def distance_to_candidate_convex_hull(point: np.ndarray, candidates: CandidateSet) -> float:
    """Distance from a point to the convex hull of candidate positions."""
    point = np.asarray(point, dtype=float)
    positions = candidates.positions
    n_candidates = candidates.n_candidates

    def objective(weights: np.ndarray) -> float:
        hull_point = weights @ positions
        return float(np.sum((hull_point - point) ** 2))

    constraints = [{"type": "eq", "fun": lambda weights: weights.sum() - 1.0}]
    bounds = [(0.0, 1.0)] * n_candidates
    x0 = np.full(n_candidates, 1.0 / n_candidates)
    result = minimize(objective, x0=x0, bounds=bounds, constraints=constraints)
    if not result.success:
        hull_point = x0 @ positions
    else:
        hull_point = np.asarray(result.x, dtype=float) @ positions
    return float(np.linalg.norm(hull_point - point))


def compute_coverage_gap(electorate: Electorate, candidates: CandidateSet) -> float:
    """Distance from the electorate minimax center to the candidate convex hull."""
    minimax_center = approximate_minimax_center(electorate)
    return distance_to_candidate_convex_hull(minimax_center, candidates)


def compute_next_step_voter_variance(
    electorate: Electorate,
    winner_position: np.ndarray,
    voter_dynamics: str = "Backlash",
) -> float:
    """
    Evaluate the next-round voter variance from a hypothetical winner position.

    The depolarization oracle uses the deterministic part of the voter update,
    so the objective reflects the paper's contraction object rather than
    one particular noisy realization.
    """
    voter_spec = build_voter_dynamics(voter_dynamics)
    deterministic_spec = replace(voter_spec, noise_scale=0.0)
    updated = advance_voters(
        electorate,
        winner_position=np.asarray(winner_position, dtype=float),
        dynamics=deterministic_spec,
        seed=0,
    )
    return compute_voter_variance(updated)


def _minimize_over_unit_box(
    objective,
    initial_points: list[np.ndarray],
    n_dims: int,
    tol: float = 1e-7,
) -> tuple[np.ndarray, float]:
    bounds = [(0.0, 1.0)] * n_dims
    best_point = np.clip(np.asarray(initial_points[0], dtype=float), 0.0, 1.0)
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


def choose_theory_oracle_outcome(
    electorate: Electorate,
    oracle_name: str,
    voter_dynamics: str = "Backlash",
    tol: float = 1e-7,
) -> tuple[np.ndarray, float]:
    """Choose an oracle outcome position for one round of the theory notebook."""
    mean_center = electorate.mean()
    minimax_center = approximate_minimax_center(electorate, tol=tol)
    initial_points = [mean_center, minimax_center]

    if oracle_name == "Centrality oracle":
        objective = lambda point: compute_winner_radius(electorate, point)
    elif oracle_name == "Depolarization oracle":
        objective = lambda point: compute_next_step_voter_variance(
            electorate,
            point,
            voter_dynamics=voter_dynamics,
        )
    else:
        raise ValueError(f"Unknown oracle_name: {oracle_name}")

    return _minimize_over_unit_box(
        objective=objective,
        initial_points=initial_points,
        n_dims=electorate.n_dims,
        tol=tol,
    )


def trace_theory_oracle_state(
    electorate: Electorate,
    candidates: CandidateSet,
    oracle_name: str,
    n_steps: int = 12,
    voter_dynamics: str = "Backlash",
    candidate_dynamics: str | CandidateDynamicsSpec = "Broad coalition chase",
    seed: int = 0,
) -> dict[str, object]:
    """Roll one oracle forward and keep the start/end states for plotting."""
    voter_spec = build_voter_dynamics(voter_dynamics)
    candidate_spec = build_theory_candidate_dynamics(candidate_dynamics)

    current_electorate = _copy_electorate(electorate)
    current_candidates = _copy_candidates(candidates)
    start_electorate = _copy_electorate(electorate)
    start_candidates = _copy_candidates(candidates)
    final_outcome = None

    for step in range(n_steps):
        final_outcome, _ = choose_theory_oracle_outcome(
            current_electorate,
            oracle_name=oracle_name,
            voter_dynamics=voter_dynamics,
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
        "final_outcome": final_outcome,
    }


def run_theory_oracle_trajectory(
    electorate: Electorate,
    candidates: CandidateSet,
    oracle_names: list[str] | None = None,
    n_steps: int = 12,
    voter_dynamics: str = "Backlash",
    candidate_dynamics: str | CandidateDynamicsSpec = "Broad coalition chase",
    electorate_name: str | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Compare oracle winner rules that optimize different one-step objectives.

    The centrality oracle minimizes the current winner radius R_t. The
    depolarization oracle minimizes the next-step voter variance D^{t+1}
    using the deterministic voter update.
    """
    oracle_names = THEORY_ORACLE_ORDER if oracle_names is None else list(oracle_names)
    voter_spec = build_voter_dynamics(voter_dynamics)
    candidate_spec = build_theory_candidate_dynamics(candidate_dynamics)
    baseline_electorate = _copy_electorate(electorate)

    rows: list[dict[str, float | int | str | list[float]]] = []
    for oracle_offset, oracle_name in enumerate(oracle_names):
        current_electorate = _copy_electorate(electorate)
        current_candidates = _copy_candidates(candidates)
        oracle_seed = seed + 1009 * oracle_offset

        for step in range(n_steps):
            outcome_position, oracle_objective_value = choose_theory_oracle_outcome(
                current_electorate,
                oracle_name=oracle_name,
                voter_dynamics=voter_dynamics,
            )
            nearest_candidate = np.linalg.norm(
                current_electorate.preferences[:, None, :] - current_candidates.positions[None, :, :],
                axis=2,
            ).argmin(axis=1)
            weights = _one_hot(nearest_candidate, current_candidates.n_candidates)
            metrics = compute_theory_metrics(
                current_electorate,
                current_candidates,
                outcome_position,
                weights,
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
                    "winner_radius": metrics.winner_radius,
                    "mean_winner_distance": metrics.mean_winner_distance,
                    "voter_variance": metrics.voter_variance,
                    "supporter_centroid_radius": metrics.supporter_centroid_radius,
                    "mean_supporter_centroid_distance": metrics.mean_supporter_centroid_distance,
                    "candidate_variance": metrics.candidate_variance,
                    "candidate_electorate_center_gap": compute_candidate_electorate_center_gap(
                        current_electorate,
                        current_candidates,
                    ),
                    "oracle_objective_value": float(oracle_objective_value),
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


def run_theory_oracle_replicates(
    electorate_name: str,
    candidate_name: str,
    ratio_name: str = "Original",
    oracle_names: list[str] | None = None,
    n_steps: int = 12,
    voter_dynamics: str = "Backlash",
    candidate_dynamics: str | CandidateDynamicsSpec = "Broad coalition chase",
    n_runs: int = 20,
    n_voters: int = 1400,
    seed: int = 0,
) -> pd.DataFrame:
    """Run repeated oracle trajectories across multiple random draws."""
    oracle_names = THEORY_ORACLE_ORDER if oracle_names is None else list(oracle_names)
    rows: list[pd.DataFrame] = []

    for run_id in range(n_runs):
        run_seed = seed + 2027 * run_id
        electorate = build_polarization_electorate(
            electorate_name,
            seed=run_seed,
            n_voters=n_voters,
            ratio_name=ratio_name,
        )
        candidates = build_polarization_candidates(candidate_name)
        trajectory = run_theory_oracle_trajectory(
            electorate,
            candidates,
            oracle_names=oracle_names,
            n_steps=n_steps,
            voter_dynamics=voter_dynamics,
            candidate_dynamics=candidate_dynamics,
            electorate_name=electorate_name,
            seed=run_seed,
        )
        trajectory["run_id"] = run_id
        trajectory["electorate"] = electorate_name
        trajectory["ratio"] = ratio_name
        trajectory["candidate_slate"] = candidate_name
        trajectory["voter_dynamics"] = voter_dynamics
        trajectory["candidate_dynamics"] = (
            candidate_dynamics if isinstance(candidate_dynamics, str) else "custom"
        )
        rows.append(trajectory)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def run_theory_trajectory(
    electorate: Electorate,
    candidates: CandidateSet,
    system_names: list[str],
    n_steps: int = 12,
    voter_dynamics: str = "Backlash",
    candidate_dynamics: str | CandidateDynamicsSpec = "Broad coalition chase",
    approval_threshold: float | None = None,
    electorate_name: str | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Run a repeated-election trajectory with paper-facing geometric metrics."""
    voter_spec = build_voter_dynamics(voter_dynamics)
    candidate_spec = build_theory_candidate_dynamics(candidate_dynamics)
    baseline_electorate = _copy_electorate(electorate)

    rows: list[dict[str, float | int | str | list[float]]] = []
    for system_offset, system_name in enumerate(system_names):
        system = build_theory_system(system_name)
        current_electorate = _copy_electorate(electorate)
        current_candidates = _copy_candidates(candidates)
        system_seed = seed + 1009 * system_offset

        for step in range(n_steps):
            ballots = BallotProfile.from_preferences(
                current_electorate,
                current_candidates,
                approval_threshold=approval_threshold,
            )
            result = system.run(ballots, current_candidates)
            weights = supporter_weight_matrix(
                system_name,
                ballots,
                current_candidates,
                system=system,
            )
            metrics = compute_theory_metrics(
                current_electorate,
                current_candidates,
                result.outcome_position,
                weights,
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
                    "winner_radius": metrics.winner_radius,
                    "mean_winner_distance": metrics.mean_winner_distance,
                    "voter_variance": metrics.voter_variance,
                    "supporter_centroid_radius": metrics.supporter_centroid_radius,
                    "mean_supporter_centroid_distance": metrics.mean_supporter_centroid_distance,
                    "candidate_variance": metrics.candidate_variance,
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


def summarize_theory_trajectory_changes(trajectory_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize start/end changes for paper-facing metrics."""
    metric_columns = [
        "winner_radius",
        "mean_winner_distance",
        "voter_variance",
        "supporter_centroid_radius",
        "mean_supporter_centroid_distance",
        "candidate_variance",
        "candidate_electorate_center_gap",
        "oracle_objective_value",
        "normalized_displacement_asymmetry",
        "majority_camp_displacement",
        "minority_camp_displacement",
        "coalition_gap",
        "coalition_gap_change",
    ]

    rows: list[dict[str, float | int | str]] = []
    for system, system_df in trajectory_df.groupby("system", sort=False):
        system_df = system_df.sort_values("step").reset_index(drop=True)
        start = system_df.iloc[0]
        end = system_df.iloc[-1]
        row: dict[str, float | int | str] = {
            "system": system,
            "start_winner": start["winner"],
            "end_winner": end["winner"],
            "n_steps": int(system_df["step"].max() + 1),
        }
        for metric in metric_columns:
            if metric not in system_df.columns:
                continue
            row[f"{metric}_start"] = float(start[metric])
            row[f"{metric}_end"] = float(end[metric])
            row[f"{metric}_delta"] = float(end[metric] - start[metric])
        rows.append(row)

    return pd.DataFrame(rows)


def run_theory_grid(
    electorate_names: list[str] | None = None,
    ratio_names: list[str] | None = None,
    candidate_names: list[str] | None = None,
    voter_dynamics_names: list[str] | None = None,
    candidate_dynamics_names: list[str] | None = None,
    system_names: list[str] | None = None,
    n_steps: int = 12,
    n_voters: int = 1400,
    seed: int = 0,
) -> pd.DataFrame:
    """Run the paper-facing repeated-election grid with direct R_t and S_t metrics."""
    electorate_names = ELECTORATE_ORDER if electorate_names is None else list(electorate_names)
    ratio_names = RATIO_ORDER if ratio_names is None else list(ratio_names)
    candidate_names = CANDIDATE_ORDER[:2] if candidate_names is None else list(candidate_names)
    voter_dynamics_names = VOTER_MODEL_ORDER if voter_dynamics_names is None else list(voter_dynamics_names)
    candidate_dynamics_names = (
        CANDIDATE_MODEL_ORDER if candidate_dynamics_names is None else list(candidate_dynamics_names)
    )
    system_names = THEORY_BASELINE_SYSTEMS if system_names is None else list(system_names)

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
                        trajectory = run_theory_trajectory(
                            electorate,
                            candidates,
                            system_names=system_names,
                            n_steps=n_steps,
                            voter_dynamics=voter_name,
                            candidate_dynamics=candidate_model,
                            electorate_name=electorate_name,
                            seed=case_seed,
                        )
                        summary = summarize_theory_trajectory_changes(trajectory)
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


def summarize_asymmetry_by_system(
    summary_df: pd.DataFrame,
    value_column: str = "normalized_displacement_asymmetry_end",
) -> pd.DataFrame:
    """Aggregate camp-asymmetry outcomes by system and camp-balance setting."""
    if value_column not in summary_df.columns:
        return pd.DataFrame(columns=["ratio", "system", value_column, "case_count"])
    return (
        summary_df.groupby(["ratio", "system"], as_index=False)
        .agg(
            **{
                value_column: (value_column, "mean"),
                "case_count": (value_column, "size"),
            }
        )
        .sort_values(["ratio", "system"])
        .reset_index(drop=True)
    )


def summarize_fractional_continuous_sweep(
    summary_df: pd.DataFrame,
    alphas: tuple[float, ...] = (0.25, 0.5, 0.75),
) -> pd.DataFrame:
    """Aggregate continuous-fractional sigma runs and add weighted objectives."""
    sweep_df = summary_df.copy()
    sweep_df["sigma"] = sweep_df["system"].map(parse_fractional_continuous_sigma)
    sweep_df = sweep_df[sweep_df["sigma"].notna()].copy()
    if sweep_df.empty:
        columns = [
            "sigma",
            "mean_winner_radius_end",
            "mean_supporter_centroid_radius_end",
            "mean_voter_variance_delta",
            "mean_candidate_variance_delta",
        ]
        columns += [f"mean_weighted_cost_alpha_{alpha:.2f}" for alpha in alphas]
        return pd.DataFrame(columns=columns)

    summary = (
        sweep_df.groupby("sigma", as_index=False)
        .agg(
            mean_winner_radius_end=("winner_radius_end", "mean"),
            mean_supporter_centroid_radius_end=("supporter_centroid_radius_end", "mean"),
            mean_voter_variance_delta=("voter_variance_delta", "mean"),
            mean_candidate_variance_delta=("candidate_variance_delta", "mean"),
        )
        .sort_values("sigma")
        .reset_index(drop=True)
    )

    for alpha in alphas:
        summary[f"mean_weighted_cost_alpha_{alpha:.2f}"] = (
            alpha * summary["mean_winner_radius_end"]
            + (1.0 - alpha) * summary["mean_supporter_centroid_radius_end"]
        )
    return summary


def build_fractional_tradeoff_cases() -> list[dict[str, object]]:
    """Hand-built cases chosen to make interior sigma tradeoffs more plausible."""
    return [
        {
            "case_name": "Bridge gap with centered ladder",
            "electorate_name": "Bridge conflict",
            "ratio_name": "Original",
            "candidate_name": "Centrist ladder",
            "voter_dynamics": "Backlash",
            "candidate_dynamics": "Static candidates",
        },
        {
            "case_name": "Asymmetric resentment with polarized elites",
            "electorate_name": "Asymmetric resentment",
            "ratio_name": "70:30",
            "candidate_name": "Polarized elites",
            "voter_dynamics": "Sorting pressure",
            "candidate_dynamics": "Static candidates",
        },
        {
            "case_name": "Two blocs with one-sided manual slate",
            "electorate_name": "Two blocs",
            "ratio_name": "50:50",
            "candidate_positions": np.array(
                [
                    [0.08, 0.70],
                    [0.18, 0.62],
                    [0.30, 0.56],
                    [0.42, 0.52],
                    [0.58, 0.46],
                ],
                dtype=float,
            ),
            "candidate_labels": ["L1", "L2", "L3", "L4", "Bridge"],
            "voter_dynamics": "Sorting pressure",
            "candidate_dynamics": "Static candidates",
        },
        {
            "case_name": "Bridge conflict with extreme-only slate",
            "electorate_name": "Bridge conflict",
            "ratio_name": "Original",
            "candidate_positions": np.array(
                [
                    [0.08, 0.72],
                    [0.16, 0.63],
                    [0.84, 0.37],
                    [0.92, 0.28],
                ],
                dtype=float,
            ),
            "candidate_labels": ["L-edge", "L-base", "R-base", "R-edge"],
            "voter_dynamics": "Backlash",
            "candidate_dynamics": "Static candidates",
        },
    ]


def _build_tradeoff_case_objects(
    case: dict[str, object],
    n_voters: int,
    seed: int,
) -> tuple[Electorate, CandidateSet]:
    electorate = build_polarization_electorate(
        str(case["electorate_name"]),
        seed=seed,
        n_voters=n_voters,
        ratio_name=str(case.get("ratio_name", "Original")),
    )
    if "candidate_positions" in case:
        positions = np.asarray(case["candidate_positions"], dtype=float)
        labels = list(case.get("candidate_labels", [f"C{i}" for i in range(len(positions))]))
        candidates = CandidateSet(positions=positions, labels=labels)
    else:
        candidates = build_polarization_candidates(str(case["candidate_name"]))
    return electorate, candidates


def search_fractional_interior_optima(
    cases: list[dict[str, object]] | None = None,
    sigma_values: list[float] | np.ndarray | None = None,
    alphas: tuple[float, ...] = (0.05, 0.15, 0.25, 0.5, 0.75, 0.85, 0.95),
    n_steps: int = 10,
    n_voters: int = 1000,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Targeted search for cases where the weighted fractional objective has an interior minimum."""
    cases = build_fractional_tradeoff_cases() if cases is None else list(cases)
    sigma_values = (
        np.array([0.03, 0.05, 0.08, 0.12, 0.15, 0.22, 0.30, 0.45, 0.70, 1.00, 1.50, 2.50], dtype=float)
        if sigma_values is None
        else np.asarray(sigma_values, dtype=float)
    )

    rows: list[dict[str, object]] = []
    minima_rows: list[dict[str, object]] = []
    for case_idx, case in enumerate(cases):
        case_seed = seed + 2027 * case_idx
        electorate, candidates = _build_tradeoff_case_objects(case, n_voters=n_voters, seed=case_seed)
        coverage_gap = compute_coverage_gap(electorate, candidates)
        systems = [fractional_continuous_name(float(sigma)) for sigma in sigma_values]
        summary = run_theory_grid(
            electorate_names=[str(case["electorate_name"])],
            ratio_names=[str(case.get("ratio_name", "Original"))],
            candidate_names=[str(case["candidate_name"])],
            voter_dynamics_names=[str(case.get("voter_dynamics", "Backlash"))],
            candidate_dynamics_names=[str(case.get("candidate_dynamics", "Static candidates"))],
            system_names=systems,
            n_steps=n_steps,
            n_voters=n_voters,
            seed=case_seed,
        ) if "candidate_name" in case else None
        if summary is None:
            trajectory = run_theory_trajectory(
                electorate,
                candidates,
                systems,
                n_steps=n_steps,
                voter_dynamics=str(case.get("voter_dynamics", "Backlash")),
                candidate_dynamics=str(case.get("candidate_dynamics", "Static candidates")),
                electorate_name=str(case["electorate_name"]),
                seed=case_seed,
            )
            summary = summarize_theory_trajectory_changes(trajectory)
        for _, row in summary.iterrows():
            sigma = parse_fractional_continuous_sigma(str(row["system"]))
            if sigma is None:
                continue
            record = {
                "case_name": str(case["case_name"]),
                "electorate": str(case["electorate_name"]),
                "ratio": str(case.get("ratio_name", "Original")),
                "voter_dynamics": str(case.get("voter_dynamics", "Backlash")),
                "candidate_dynamics": str(case.get("candidate_dynamics", "Static candidates")),
                "coverage_gap": coverage_gap,
                "sigma": float(sigma),
                "winner_radius_end": float(row["winner_radius_end"]),
                "supporter_centroid_radius_end": float(row["supporter_centroid_radius_end"]),
                "normalized_displacement_asymmetry_end": float(row.get("normalized_displacement_asymmetry_end", np.nan)),
            }
            for alpha in alphas:
                record[f"weighted_cost_alpha_{alpha:.2f}"] = compute_weighted_polarization_cost(
                    record["winner_radius_end"],
                    record["supporter_centroid_radius_end"],
                    alpha=alpha,
                )
            rows.append(record)

    search_df = pd.DataFrame(rows).sort_values(["case_name", "sigma"]).reset_index(drop=True)
    for case_name, case_df in search_df.groupby("case_name", sort=False):
        case_df = case_df.sort_values("sigma").reset_index(drop=True)
        for alpha in alphas:
            column = f"weighted_cost_alpha_{alpha:.2f}"
            values = case_df[column].to_numpy(dtype=float)
            best_idx = int(np.argmin(values))
            interior = 0 < best_idx < len(values) - 1
            left_slope = values[best_idx] - values[best_idx - 1] if interior else np.nan
            right_slope = values[best_idx + 1] - values[best_idx] if interior else np.nan
            minima_rows.append(
                {
                    "case_name": case_name,
                    "alpha": float(alpha),
                    "best_sigma": float(case_df.loc[best_idx, "sigma"]),
                    "best_cost": float(values[best_idx]),
                    "is_interior_minimum": bool(interior and left_slope < 0.0 and right_slope > 0.0),
                    "left_step_change": float(left_slope) if not np.isnan(left_slope) else np.nan,
                    "right_step_change": float(right_slope) if not np.isnan(right_slope) else np.nan,
                }
            )
    minima_df = pd.DataFrame(minima_rows)
    return search_df, minima_df


def plot_fractional_weighted_objective_curves(
    sigma_summary: pd.DataFrame,
    alphas: tuple[float, ...] = (0.25, 0.5, 0.75),
    figsize: tuple[float, float] = (8.5, 4.5),
) -> plt.Figure:
    """Plot weighted objectives L_alpha across the continuous-fractional sigma family."""
    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    palette = plt.cm.Set2(np.linspace(0, 1, len(alphas)))
    for color, alpha in zip(palette, alphas):
        column = f"mean_weighted_cost_alpha_{alpha:.2f}"
        if column not in sigma_summary:
            continue
        ax.plot(
            sigma_summary["sigma"],
            sigma_summary[column],
            marker="o",
            linewidth=2,
            markersize=4,
            color=color,
            label=f"alpha={alpha:.2f}",
        )
    ax.set_xlabel("sigma")
    ax.set_ylabel("weighted cost")
    ax.set_title("Continuous fractional weighted objective")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def plot_theory_uncertainty_trajectories(
    trajectory_df: pd.DataFrame,
    metrics: list[str],
    system_order: list[str] | None = None,
    n_cols: int = 2,
    figsize: tuple[float, float] = (11, 7),
) -> plt.Figure:
    """Plot mean trajectories with IQR bands across repeated runs."""
    system_order = system_order or list(dict.fromkeys(trajectory_df["system"]))
    palette = dict(zip(system_order, plt.cm.Set2(np.linspace(0, 1, len(system_order)))))
    n_cols = max(1, int(n_cols))
    n_rows = int(np.ceil(len(metrics) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, dpi=150, sharex=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, metric in zip(axes, metrics):
        for system_name in system_order:
            system_df = trajectory_df[trajectory_df["system"] == system_name]
            grouped = (
                system_df.groupby("step")[metric]
                .agg(
                    q25=lambda s: float(s.quantile(0.25)),
                    mean="mean",
                    q75=lambda s: float(s.quantile(0.75)),
                )
                .reset_index()
                .sort_values("step")
            )
            x = grouped["step"].to_numpy(dtype=float)
            mean = grouped["mean"].to_numpy(dtype=float)
            lower = grouped["q25"].to_numpy(dtype=float)
            upper = grouped["q75"].to_numpy(dtype=float)
            color = palette[system_name]
            ax.plot(x, mean, color=color, linewidth=2, label=system_name)
            ax.fill_between(x, lower, upper, color=color, alpha=0.18, linewidth=0)
        ax.set_title(metric.replace("_", " "))
        ax.set_xlabel("round")
        ax.grid(alpha=0.25)

    for ax in axes[len(metrics):]:
        ax.axis("off")
    axes[0].legend(frameon=False, loc="best")
    fig.tight_layout()
    return fig


__all__ = [
    "CANDIDATE_MODEL_ORDER",
    "CANDIDATE_ORDER",
    "ELECTORATE_ORDER",
    "RATIO_ORDER",
    "THEORY_BASELINE_SYSTEMS",
    "TheoryMetricBundle",
    "THEORY_ORACLE_ORDER",
    "VOTER_MODEL_ORDER",
    "approximate_minimax_center",
    "build_theory_system",
    "choose_theory_oracle_outcome",
    "compute_candidate_variance",
    "compute_candidate_electorate_center_gap",
    "compute_coverage_gap",
    "compute_mean_supporter_centroid_distance",
    "compute_next_step_voter_variance",
    "compute_mean_winner_distance",
    "compute_supporter_centroid_radius",
    "compute_supporter_centroids",
    "compute_theory_metrics",
    "compute_voter_variance",
    "compute_weighted_polarization_cost",
    "compute_winner_radius",
    "distance_to_candidate_convex_hull",
    "fractional_continuous_name",
    "parse_fractional_continuous_sigma",
    "plot_fractional_weighted_objective_curves",
    "plot_theory_uncertainty_trajectories",
    "build_fractional_tradeoff_cases",
    "build_theory_candidate_dynamics",
    "run_theory_grid",
    "run_theory_oracle_replicates",
    "run_theory_oracle_trajectory",
    "run_theory_trajectory",
    "search_fractional_interior_optima",
    "summarize_asymmetry_by_system",
    "summarize_fractional_continuous_sweep",
    "summarize_theory_trajectory_changes",
    "supporter_weight_matrix",
    "THEORY_MU0_CANDIDATE_MODELS",
    "trace_theory_oracle_state",
    "theory_helper_overview",
]
