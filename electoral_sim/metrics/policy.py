"""Policy-layer welfare summaries built on top of realized policy outcomes."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from electoral_sim.candidates import CandidateSet
from electoral_sim.electorate import Electorate
from electoral_sim.policy import (
    PolicyConsequenceSpec,
    PolicyDefinitionLike,
    PolicyOutcome,
    extract_policy_outcome,
    policy_utility_components,
)
from electoral_sim.types import ElectionResult


@dataclass
class GroupPolicyMetrics:
    """Policy consequences for a single labeled group."""

    group_id: int
    group_name: str
    n_voters: int
    population_share: float
    mean_policy_distance: float
    mean_total_utility: float
    mean_distance_utility: float
    mean_public_goods_utility: float
    mean_group_adjustment_utility: float
    mean_threshold_utility: float


@dataclass
class PolicyMetricsSummary:
    """Aggregate and group-level policy welfare summaries."""

    system_name: str
    policy_rule: str
    policy_vector: np.ndarray
    mean_policy_distance: float
    mean_total_utility: float
    utility_std: float
    mean_distance_utility: float
    mean_public_goods_utility: float
    mean_group_adjustment_utility: float
    mean_threshold_utility: float
    groups: list[GroupPolicyMetrics]
    majority_group_utility: float
    minority_group_utility: float
    majority_minority_utility_gap: float
    max_group_utility_gap: float


def compute_policy_metrics(
    result: ElectionResult | PolicyOutcome,
    electorate: Electorate,
    candidates: CandidateSet | None = None,
    policy_rule: PolicyDefinitionLike = "outcome",
    consequence_spec: PolicyConsequenceSpec | None = None,
) -> PolicyMetricsSummary:
    """
    Compute policy welfare summaries from an election result or direct policy outcome.

    By default, the realized policy is just ``result.outcome_position``. This keeps
    the existing simulator semantics unchanged while making the policy layer explicit.
    """

    if isinstance(result, PolicyOutcome):
        policy = result
        system_name = result.system_name
    else:
        policy = extract_policy_outcome(result, candidates=candidates, policy_rule=policy_rule)
        system_name = result.system_name

    components = policy_utility_components(policy, electorate, spec=consequence_spec)
    total_utility = components.total_utility
    threshold_utility = (
        np.zeros(electorate.n_voters, dtype=float)
        if components.threshold_utility is None
        else components.threshold_utility
    )

    groups: list[GroupPolicyMetrics] = []
    if electorate.has_groups:
        for group_id, mask in electorate.group_indices().items():
            groups.append(
                GroupPolicyMetrics(
                    group_id=int(group_id),
                    group_name=electorate.group_names[int(group_id)],
                    n_voters=int(mask.sum()),
                    population_share=float(mask.mean()),
                    mean_policy_distance=float(components.policy_distance[mask].mean()),
                    mean_total_utility=float(total_utility[mask].mean()),
                    mean_distance_utility=float(components.distance_utility[mask].mean()),
                    mean_public_goods_utility=float(components.public_goods_utility[mask].mean()),
                    mean_group_adjustment_utility=float(
                        components.group_adjustment_utility[mask].mean()
                    ),
                    mean_threshold_utility=float(threshold_utility[mask].mean()),
                )
            )

    majority_group_utility = float("nan")
    minority_group_utility = float("nan")
    majority_minority_utility_gap = float("nan")
    max_group_utility_gap = float("nan")
    if groups:
        groups_by_share = sorted(groups, key=lambda g: (g.population_share, g.n_voters))
        minority = groups_by_share[0]
        majority = groups_by_share[-1]
        group_utilities = np.array([group.mean_total_utility for group in groups], dtype=float)
        majority_group_utility = float(majority.mean_total_utility)
        minority_group_utility = float(minority.mean_total_utility)
        majority_minority_utility_gap = majority_group_utility - minority_group_utility
        max_group_utility_gap = float(group_utilities.max() - group_utilities.min())

    return PolicyMetricsSummary(
        system_name=system_name,
        policy_rule=policy.rule,
        policy_vector=policy.vector.copy(),
        mean_policy_distance=float(components.policy_distance.mean()),
        mean_total_utility=float(total_utility.mean()),
        utility_std=float(total_utility.std()),
        mean_distance_utility=float(components.distance_utility.mean()),
        mean_public_goods_utility=float(components.public_goods_utility.mean()),
        mean_group_adjustment_utility=float(components.group_adjustment_utility.mean()),
        mean_threshold_utility=float(threshold_utility.mean()),
        groups=groups,
        majority_group_utility=majority_group_utility,
        minority_group_utility=minority_group_utility,
        majority_minority_utility_gap=majority_minority_utility_gap,
        max_group_utility_gap=max_group_utility_gap,
    )


__all__ = [
    "GroupPolicyMetrics",
    "PolicyMetricsSummary",
    "compute_policy_metrics",
]
