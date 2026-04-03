"""Group-level welfare metrics for labeled electorates."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from electoral_sim.candidates import CandidateSet
from electoral_sim.electorate import Electorate
from electoral_sim.types import ElectionResult


@dataclass
class GroupOutcomeMetrics:
    """Outcome metrics for a single labeled voter group."""

    group_id: int
    group_name: str
    n_voters: int
    population_share: float
    mean_voter_distance: float
    worst_case_distance: float
    majority_satisfaction: float
    welfare: float


@dataclass
class GroupMetricsSummary:
    """Grouped outcome metrics plus simple disparity summaries."""

    system_name: str
    groups: list[GroupOutcomeMetrics]
    max_mean_distance_gap: float
    max_majority_satisfaction_gap: float
    min_group_welfare: float


def compute_group_metrics(
    result: ElectionResult,
    electorate: Electorate,
    candidates: CandidateSet,
) -> GroupMetricsSummary:
    """
    Compute per-group welfare metrics for a labeled electorate.

    The group welfare is defined as the negative mean distance from members of a
    group to the election outcome, so larger values are better.
    """
    if not electorate.has_groups:
        raise ValueError("compute_group_metrics requires electorate.group_ids to be present")

    outcome = result.outcome_position
    cand_dists = None
    if candidates.n_candidates > 1:
        cand_dists = np.linalg.norm(
            electorate.preferences[:, None, :] - candidates.positions[None, :, :],
            axis=2,
        )

    group_results: list[GroupOutcomeMetrics] = []
    for group_id, mask in electorate.group_indices().items():
        prefs = electorate.preferences[mask]
        voter_dists = np.linalg.norm(prefs - outcome, axis=1)
        mean_distance = float(voter_dists.mean())
        worst_case_distance = float(voter_dists.max())

        if cand_dists is None:
            majority_satisfaction = 1.0
        else:
            nearest_candidate_dist = cand_dists[mask].min(axis=1)
            majority_satisfaction = float(
                (voter_dists <= nearest_candidate_dist + 1e-9).mean()
            )

        group_results.append(
            GroupOutcomeMetrics(
                group_id=group_id,
                group_name=electorate.group_names[group_id],
                n_voters=int(mask.sum()),
                population_share=float(mask.mean()),
                mean_voter_distance=mean_distance,
                worst_case_distance=worst_case_distance,
                majority_satisfaction=majority_satisfaction,
                welfare=-mean_distance,
            )
        )

    mean_distances = np.array([group.mean_voter_distance for group in group_results], dtype=float)
    majority_rates = np.array([group.majority_satisfaction for group in group_results], dtype=float)
    welfares = np.array([group.welfare for group in group_results], dtype=float)

    return GroupMetricsSummary(
        system_name=result.system_name,
        groups=group_results,
        max_mean_distance_gap=float(mean_distances.max() - mean_distances.min()),
        max_majority_satisfaction_gap=float(majority_rates.max() - majority_rates.min()),
        min_group_welfare=float(welfares.min()),
    )
