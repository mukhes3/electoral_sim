"""Robustness metrics for truthful vs reported-outcome comparisons."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from electoral_sim.candidates import CandidateSet
from electoral_sim.electorate import Electorate
from electoral_sim.types import ElectionResult


@dataclass
class FractionalRobustnessMetrics:
    """Comparison metrics for truthful and reporting-distorted outcomes."""

    outcome_shift: float
    seat_share_shift: float
    winner_changed: bool
    distance_to_median_delta: float
    mean_voter_distance_delta: float
    majority_satisfaction_delta: float
    target_mean_utility_gain: float = float("nan")
    other_voters_mean_utility_gain: float = float("nan")


def _seat_share_vector(result: ElectionResult, candidates: CandidateSet) -> np.ndarray:
    shares = np.zeros(candidates.n_candidates, dtype=float)
    for idx, share in result.seat_shares.items():
        shares[int(idx)] = float(share)
    return shares


def _mean_distance_to_outcome(
    electorate: Electorate,
    outcome: np.ndarray,
    voter_mask: np.ndarray,
) -> float:
    dists = np.linalg.norm(electorate.preferences[voter_mask] - outcome, axis=1)
    return float(dists.mean()) if len(dists) else float("nan")


def compute_fractional_robustness_metrics(
    truthful_result: ElectionResult,
    reported_result: ElectionResult,
    electorate: Electorate,
    candidates: CandidateSet,
    target_mask: np.ndarray | None = None,
) -> FractionalRobustnessMetrics:
    """
    Compare a truthful fractional outcome against a reported-position outcome.

    Positive utility gain means the group is, on average, closer to the
    reported outcome than to the truthful one.
    """
    from electoral_sim.metrics import compute_metrics

    truthful_metrics = compute_metrics(truthful_result, electorate, candidates)
    reported_metrics = compute_metrics(reported_result, electorate, candidates)

    target_gain = float("nan")
    other_gain = float("nan")
    if target_mask is not None:
        target_mask = np.asarray(target_mask, dtype=bool)
        if target_mask.shape != (electorate.n_voters,):
            raise ValueError(
                f"target_mask must have shape ({electorate.n_voters},), got {target_mask.shape}."
            )
        target_truthful = _mean_distance_to_outcome(
            electorate,
            truthful_result.outcome_position,
            target_mask,
        )
        target_reported = _mean_distance_to_outcome(
            electorate,
            reported_result.outcome_position,
            target_mask,
        )
        if not np.isnan(target_truthful) and not np.isnan(target_reported):
            target_gain = target_truthful - target_reported

        other_mask = ~target_mask
        other_truthful = _mean_distance_to_outcome(
            electorate,
            truthful_result.outcome_position,
            other_mask,
        )
        other_reported = _mean_distance_to_outcome(
            electorate,
            reported_result.outcome_position,
            other_mask,
        )
        if not np.isnan(other_truthful) and not np.isnan(other_reported):
            other_gain = other_truthful - other_reported

    return FractionalRobustnessMetrics(
        outcome_shift=float(
            np.linalg.norm(
                truthful_result.outcome_position - reported_result.outcome_position
            )
        ),
        seat_share_shift=float(
            np.abs(
                _seat_share_vector(truthful_result, candidates)
                - _seat_share_vector(reported_result, candidates)
            ).sum()
        ),
        winner_changed=truthful_result.winner_indices != reported_result.winner_indices,
        distance_to_median_delta=(
            reported_metrics.distance_to_median - truthful_metrics.distance_to_median
        ),
        mean_voter_distance_delta=(
            reported_metrics.mean_voter_distance - truthful_metrics.mean_voter_distance
        ),
        majority_satisfaction_delta=(
            reported_metrics.majority_satisfaction - truthful_metrics.majority_satisfaction
        ),
        target_mean_utility_gain=target_gain,
        other_voters_mean_utility_gain=other_gain,
    )
