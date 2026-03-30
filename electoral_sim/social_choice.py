"""Teaching-oriented helpers for social choice diagnostics and notebooks."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from electoral_sim.ballots import BallotProfile
from electoral_sim.candidates import CandidateSet


@dataclass
class SocialRanking:
    """A labeled candidate ordering together with the underlying ranking values."""

    method_name: str
    candidate_order: list[int]
    labels: list[str]
    values: np.ndarray
    value_label: str

    @property
    def ordered_labels(self) -> list[str]:
        return [self.labels[idx] for idx in self.candidate_order]


@dataclass
class PairwiseSummary:
    """Pairwise-majority summary for notebook-style diagnostics."""

    labels: list[str]
    preference_counts: np.ndarray
    preference_shares: np.ndarray
    margin_matrix: np.ndarray
    condorcet_winner: int | None
    majority_cycle: list[int] | None


def _labels(candidates: CandidateSet | None, n_candidates: int) -> list[str]:
    if candidates is None:
        return [f"C{i}" for i in range(n_candidates)]
    return list(candidates.labels)


def _rank_from_values(values: np.ndarray, method_name: str, value_label: str, labels: list[str]) -> SocialRanking:
    order = np.argsort(-np.asarray(values), kind="stable")
    return SocialRanking(
        method_name=method_name,
        candidate_order=[int(idx) for idx in order],
        labels=labels,
        values=np.asarray(values, dtype=float).copy(),
        value_label=value_label,
    )


def plurality_social_ranking(
    ballots: BallotProfile,
    candidates: CandidateSet | None = None,
) -> SocialRanking:
    """Rank candidates by plurality first-preference counts."""
    labels = _labels(candidates, ballots.n_candidates)
    return _rank_from_values(
        ballots.plurality_counts(),
        method_name="Plurality social ranking",
        value_label="plurality_votes",
        labels=labels,
    )


def borda_social_ranking(
    ballots: BallotProfile,
    candidates: CandidateSet | None = None,
) -> SocialRanking:
    """Rank candidates by Borda score."""
    labels = _labels(candidates, ballots.n_candidates)
    return _rank_from_values(
        ballots.borda_scores(),
        method_name="Borda social ranking",
        value_label="borda_points",
        labels=labels,
    )


def approval_social_ranking(
    ballots: BallotProfile,
    candidates: CandidateSet | None = None,
) -> SocialRanking:
    """Rank candidates by approval counts."""
    labels = _labels(candidates, ballots.n_candidates)
    counts = ballots.active_approvals().sum(axis=0) if ballots.n_active_voters > 0 else np.zeros(ballots.n_candidates)
    return _rank_from_values(
        counts,
        method_name="Approval social ranking",
        value_label="approval_votes",
        labels=labels,
    )


def score_social_ranking(
    ballots: BallotProfile,
    candidates: CandidateSet | None = None,
) -> SocialRanking:
    """Rank candidates by mean score."""
    labels = _labels(candidates, ballots.n_candidates)
    mean_scores = ballots.active_scores().mean(axis=0) if ballots.n_active_voters > 0 else np.zeros(ballots.n_candidates)
    return _rank_from_values(
        mean_scores,
        method_name="Score social ranking",
        value_label="mean_score",
        labels=labels,
    )


def copeland_social_ranking(
    ballots: BallotProfile,
    candidates: CandidateSet | None = None,
) -> SocialRanking:
    """
    Rank candidates by Copeland score:
    1 point for each pairwise win, 0.5 for each tie, 0 for each loss.
    """
    labels = _labels(candidates, ballots.n_candidates)
    pairwise = ballots.pairwise_matrix()
    n = ballots.n_candidates
    scores = np.zeros(n)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if pairwise[i, j] > pairwise[j, i]:
                scores[i] += 1.0
            elif np.isclose(pairwise[i, j], pairwise[j, i]):
                scores[i] += 0.5
    return _rank_from_values(
        scores,
        method_name="Copeland social ranking",
        value_label="copeland_score",
        labels=labels,
    )


def condorcet_winner(ballots: BallotProfile) -> int | None:
    """Return the direct Condorcet winner, if one exists."""
    pairwise = ballots.pairwise_matrix()
    for i in range(ballots.n_candidates):
        if all(pairwise[i, j] > 0.5 for j in range(ballots.n_candidates) if j != i):
            return int(i)
    return None


def find_majority_cycle(ballots: BallotProfile) -> list[int] | None:
    """
    Return one directed majority cycle if one exists, otherwise None.

    The returned cycle repeats the starting candidate at the end,
    e.g. [0, 1, 2, 0].
    """
    margins = ballots.pairwise_margin_matrix()
    n = ballots.n_candidates
    adjacency = {i: [j for j in range(n) if i != j and margins[i, j] > 0] for i in range(n)}

    def dfs(node: int, stack: list[int], seen_in_stack: set[int]) -> list[int] | None:
        for nxt in adjacency[node]:
            if nxt == stack[0] and len(stack) >= 3:
                return stack + [nxt]
            if nxt in seen_in_stack:
                continue
            cycle = dfs(nxt, stack + [nxt], seen_in_stack | {nxt})
            if cycle is not None:
                return cycle
        return None

    for start in range(n):
        cycle = dfs(start, [start], {start})
        if cycle is not None:
            return cycle
    return None


def pairwise_summary(
    ballots: BallotProfile,
    candidates: CandidateSet | None = None,
) -> PairwiseSummary:
    """Return a labeled summary of pairwise-majority structure."""
    shares = ballots.pairwise_matrix()
    return PairwiseSummary(
        labels=_labels(candidates, ballots.n_candidates),
        preference_counts=ballots.pairwise_preference_counts(),
        preference_shares=shares,
        margin_matrix=shares - shares.T,
        condorcet_winner=condorcet_winner(ballots),
        majority_cycle=find_majority_cycle(ballots),
    )


def social_prefers(ranking: SocialRanking, candidate_a: int, candidate_b: int) -> bool:
    """Return True if the ranking places candidate_a above candidate_b."""
    order_pos = {candidate: idx for idx, candidate in enumerate(ranking.candidate_order)}
    return order_pos[candidate_a] < order_pos[candidate_b]
