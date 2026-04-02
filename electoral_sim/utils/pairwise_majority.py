"""Convenience helpers for two-candidate majority contests."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from electoral_sim.candidates import CandidateSet
from electoral_sim.electorate import Electorate


@dataclass
class PairwiseMajorityResult:
    """Summary of a two-candidate majority contest."""

    candidate_labels: tuple[str, str]
    vote_counts: tuple[float, float]
    vote_shares: tuple[float, float]
    winner_label: str | None
    winner_local_index: int | None
    margin: float
    tied_voter_count: int


def pairwise_majority_vote(
    electorate: Electorate,
    candidate_positions: np.ndarray,
    candidate_labels: tuple[str, str] = ("A", "B"),
) -> PairwiseMajorityResult:
    """
    Run a majority contest between exactly two candidate positions.

    Voters who are exactly equidistant from both candidates split their vote
    evenly across the two sides.
    """
    positions = np.asarray(candidate_positions, dtype=float)
    if positions.shape != (2, electorate.n_dims):
        raise ValueError(
            "candidate_positions must have shape "
            f"(2, {electorate.n_dims}), got {positions.shape}."
        )

    dists = np.linalg.norm(
        electorate.preferences[:, None, :] - positions[None, :, :],
        axis=2,
    )
    tol = 1e-12
    first_better = dists[:, 0] + tol < dists[:, 1]
    second_better = dists[:, 1] + tol < dists[:, 0]
    tied = ~(first_better | second_better)

    votes_a = float(first_better.sum()) + 0.5 * float(tied.sum())
    votes_b = float(second_better.sum()) + 0.5 * float(tied.sum())
    total_votes = max(electorate.n_voters, 1)
    shares = (votes_a / total_votes, votes_b / total_votes)
    margin = shares[0] - shares[1]

    if abs(margin) <= tol:
        winner_label = None
        winner_local_index = None
    elif margin > 0:
        winner_label = candidate_labels[0]
        winner_local_index = 0
    else:
        winner_label = candidate_labels[1]
        winner_local_index = 1

    return PairwiseMajorityResult(
        candidate_labels=tuple(candidate_labels),
        vote_counts=(votes_a, votes_b),
        vote_shares=shares,
        winner_label=winner_label,
        winner_local_index=winner_local_index,
        margin=margin,
        tied_voter_count=int(tied.sum()),
    )


def pairwise_majority_between(
    electorate: Electorate,
    candidates: CandidateSet,
    candidate_a: int,
    candidate_b: int,
) -> PairwiseMajorityResult:
    """Run a majority contest between two candidates from a candidate set."""
    indices = [int(candidate_a), int(candidate_b)]
    positions = candidates.positions[indices]
    labels = (candidates.labels[indices[0]], candidates.labels[indices[1]])
    return pairwise_majority_vote(
        electorate=electorate,
        candidate_positions=positions,
        candidate_labels=labels,
    )
