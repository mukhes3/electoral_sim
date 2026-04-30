"""Policy-layer primitives built on top of election outcomes."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, TypeAlias

import numpy as np

from electoral_sim.candidates import CandidateSet
from electoral_sim.electorate import Electorate
from electoral_sim.types import ElectionResult


class PolicyResolver(Protocol):
    """Callable protocol for custom policy-resolution rules."""

    def __call__(
        self,
        result: ElectionResult,
        candidates: CandidateSet | None = None,
    ) -> "PolicyOutcome | np.ndarray":
        ...


class PolicyUtilityFunction(Protocol):
    """Callable protocol for custom policy-utility definitions."""

    def __call__(
        self,
        policy: "PolicyOutcome | np.ndarray",
        electorate: Electorate,
        spec: "PolicyConsequenceSpec",
    ) -> "PolicyUtilityComponents":
        ...


@dataclass(frozen=True)
class PolicyOutcome:
    """
    Realized policy associated with an election result.

    Parameters
    ----------
    vector : np.ndarray
        Realized policy vector in the same dimensional space as voters.
    system_name : str
        Electoral system that produced the policy.
    rule : str
        Policy-resolution rule used to obtain ``vector`` from the election result.
    metadata : dict
        Free-form auxiliary information for notebook-level interpretation.
    """

    vector: np.ndarray
    system_name: str
    rule: str = "outcome"
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "vector", np.asarray(self.vector, dtype=float).copy())
        if self.vector.ndim != 1:
            raise ValueError("PolicyOutcome.vector must be a 1D array.")


@dataclass(frozen=True)
class PolicyDefinition:
    """
    Pluggable definition of how an election result maps to a realized policy.

    This keeps the policy layer backward compatible while allowing notebooks to
    experiment with richer interpretations than ``outcome_position`` alone.
    """

    name: str
    resolver: PolicyResolver
    metadata: dict = field(default_factory=dict)


PolicyDefinitionLike: TypeAlias = str | PolicyDefinition | PolicyResolver


@dataclass(frozen=True)
class PolicyConsequenceSpec:
    """
    Simple, backward-compatible policy consequence model.

    The default model treats policy utility as negative weighted distance from a
    voter's ideal point to the realized policy. Optional public-goods and
    group-specific additive terms let notebooks introduce more substantive
    consequences without abandoning the existing spatial representation.
    """

    distance_weight: float = 1.0
    dimension_weights: np.ndarray | None = None
    public_goods_dim: int | None = None
    public_goods_weight: float = 0.0
    group_adjustments: dict[int, float] | None = None
    threshold_effects: tuple["PolicyThresholdEffect", ...] = ()
    utility_function: PolicyUtilityFunction | None = None


@dataclass(frozen=True)
class PolicyThresholdEffect:
    """
    Threshold or floor effect for one policy dimension.

    The effect is triggered when the realized policy on ``dimension`` falls
    below a voter-specific threshold. Thresholds can be shared by everyone or
    vary by group. When triggered, the distance component can either be scaled
    or replaced by a constant level, and an additional utility offset can be
    applied.

    This makes it possible to express ideas like:
    - below a public-goods floor, all nearby policies are still bad
    - a minority group requires a minimum redistribution level before
      ideological proximity starts to matter
    """

    dimension: int
    threshold: float | None = None
    threshold_by_group: dict[int, float] | None = None
    distance_multiplier_below: float = 0.0
    distance_utility_below: float | None = None
    utility_offset_below: float = 0.0


@dataclass(frozen=True)
class PolicyUtilityComponents:
    """Per-voter utility decomposition under a realized policy."""

    total_utility: np.ndarray
    distance_utility: np.ndarray
    public_goods_utility: np.ndarray
    group_adjustment_utility: np.ndarray
    policy_distance: np.ndarray
    threshold_utility: np.ndarray | None = None


@dataclass(frozen=True)
class PolicyFeedbackSpec:
    """
    Deterministic policy-feedback rule for repeated-election experiments.

    ``alignment_strength`` moves everyone somewhat toward the realized policy.
    ``utility_sensitivity`` then scales that movement by relative utility:
    voters who benefited more move more toward the policy, while voters who
    benefited less can move away from it when the term dominates.
    """

    alignment_strength: float = 0.03
    utility_sensitivity: float = 0.08
    mean_reversion: float = 0.0
    noise_scale: float = 0.0


def extract_policy_outcome(
    result: ElectionResult,
    candidates: CandidateSet | None = None,
    policy_rule: PolicyDefinitionLike = "outcome",
) -> PolicyOutcome:
    """
    Resolve a realized policy vector from an ``ElectionResult``.

    Supported rules
    ---------------
    ``"outcome"``
        Use ``result.outcome_position``. This is the fully backward-compatible default.
    ``"centroid"``
        Use ``result.centroid_position``.
    ``"median_legislator"``
        Use ``result.median_legislator_position``.
    ``"seat_share_mean"``
        Recompute the seat-share-weighted mean using ``candidates.positions``.
    """

    if isinstance(policy_rule, PolicyDefinition):
        resolved = policy_rule.resolver(result, candidates=candidates)
        return _coerce_policy_outcome(
            resolved,
            system_name=result.system_name,
            rule=policy_rule.name,
            metadata={
                "is_pr": result.is_pr,
                "winner_indices": list(result.winner_indices),
                **dict(policy_rule.metadata),
            },
        )

    if callable(policy_rule) and not isinstance(policy_rule, str):
        resolved = policy_rule(result, candidates=candidates)
        return _coerce_policy_outcome(
            resolved,
            system_name=result.system_name,
            rule=getattr(policy_rule, "__name__", "custom"),
            metadata={"is_pr": result.is_pr, "winner_indices": list(result.winner_indices)},
        )

    if policy_rule == "outcome":
        vector = result.outcome_position
    elif policy_rule == "centroid":
        vector = result.centroid_position
    elif policy_rule == "median_legislator":
        vector = result.median_legislator_position
    elif policy_rule == "seat_share_mean":
        if candidates is None:
            raise ValueError("candidates are required when policy_rule='seat_share_mean'")
        vector = np.zeros(candidates.n_dims, dtype=float)
        for candidate_idx, share in result.seat_shares.items():
            vector += float(share) * candidates.positions[int(candidate_idx)]
    else:
        valid = ["outcome", "centroid", "median_legislator", "seat_share_mean"]
        raise ValueError(f"Unknown policy_rule: {policy_rule}. Expected one of {valid}.")

    return PolicyOutcome(
        vector=np.asarray(vector, dtype=float),
        system_name=result.system_name,
        rule=policy_rule,
        metadata={"is_pr": result.is_pr, "winner_indices": list(result.winner_indices)},
    )


def policy_utility_components(
    policy: PolicyOutcome | np.ndarray,
    electorate: Electorate,
    spec: PolicyConsequenceSpec | None = None,
) -> PolicyUtilityComponents:
    """Compute per-voter utility decomposition for a realized policy."""

    spec = PolicyConsequenceSpec() if spec is None else spec
    policy_vector = _policy_vector(policy)
    if policy_vector.shape != (electorate.n_dims,):
        raise ValueError(
            f"policy vector must have shape ({electorate.n_dims},), got {policy_vector.shape}"
        )

    if spec.utility_function is not None:
        components = spec.utility_function(policy, electorate, spec)
        policy_distance = np.asarray(components.policy_distance, dtype=float)
        distance_utility = np.asarray(components.distance_utility, dtype=float)
        public_goods_utility = np.asarray(components.public_goods_utility, dtype=float)
        group_adjustment_utility = np.asarray(components.group_adjustment_utility, dtype=float)
        total_utility = np.asarray(components.total_utility, dtype=float)
        initial_threshold_utility = (
            np.zeros(electorate.n_voters, dtype=float)
            if components.threshold_utility is None
            else np.asarray(components.threshold_utility, dtype=float).copy()
        )
        threshold_utility = initial_threshold_utility.copy()
    else:
        weights = _dimension_weights(spec.dimension_weights, electorate.n_dims)
        deltas = electorate.preferences - policy_vector[None, :]
        policy_distance = np.sqrt((weights[None, :] * np.square(deltas)).sum(axis=1))
        distance_utility = -float(spec.distance_weight) * policy_distance

        public_goods_utility = np.zeros(electorate.n_voters, dtype=float)
        if spec.public_goods_dim is not None and spec.public_goods_weight != 0.0:
            dim = int(spec.public_goods_dim)
            if dim < 0 or dim >= electorate.n_dims:
                raise ValueError(
                    f"public_goods_dim must be in [0, {electorate.n_dims - 1}], got {dim}"
                )
            public_goods_utility += float(spec.public_goods_weight) * float(policy_vector[dim])

        group_adjustment_utility = np.zeros(electorate.n_voters, dtype=float)
        if spec.group_adjustments:
            if electorate.group_ids is None:
                raise ValueError("group_adjustments require electorate.group_ids to be present")
            for group_id, adjustment in spec.group_adjustments.items():
                group_adjustment_utility[electorate.group_ids == int(group_id)] += float(adjustment)
        total_utility = distance_utility + public_goods_utility + group_adjustment_utility
        initial_threshold_utility = np.zeros(electorate.n_voters, dtype=float)
        threshold_utility = np.zeros(electorate.n_voters, dtype=float)

    for effect in spec.threshold_effects:
        thresholds = _threshold_vector(effect, electorate)
        dim = int(effect.dimension)
        if dim < 0 or dim >= electorate.n_dims:
            raise ValueError(
                f"threshold effect dimension must be in [0, {electorate.n_dims - 1}], got {dim}"
            )
        below_mask = float(policy_vector[dim]) < thresholds
        if not below_mask.any():
            continue
        if effect.distance_utility_below is None:
            replacement = effect.distance_multiplier_below * distance_utility[below_mask]
        else:
            replacement = np.full(
                int(below_mask.sum()),
                float(effect.distance_utility_below),
                dtype=float,
            )
        threshold_utility[below_mask] += replacement - distance_utility[below_mask]
        threshold_utility[below_mask] += float(effect.utility_offset_below)

    base_total = distance_utility + public_goods_utility + group_adjustment_utility
    residual_total = total_utility - base_total - initial_threshold_utility
    total_utility = base_total + threshold_utility + residual_total
    return PolicyUtilityComponents(
        total_utility=total_utility,
        distance_utility=distance_utility,
        public_goods_utility=public_goods_utility,
        group_adjustment_utility=group_adjustment_utility,
        policy_distance=policy_distance,
        threshold_utility=threshold_utility,
    )


def apply_policy_feedback(
    electorate: Electorate,
    policy: PolicyOutcome | np.ndarray,
    feedback: PolicyFeedbackSpec | None = None,
    consequence_spec: PolicyConsequenceSpec | None = None,
    utilities: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> Electorate:
    """
    Update voter positions after policy consequences are realized.

    The update is intentionally simple and deterministic by default so the
    first generation of policy notebooks remains interpretable.
    """

    feedback = PolicyFeedbackSpec() if feedback is None else feedback
    rng = np.random.default_rng() if rng is None else rng

    policy_vector = _policy_vector(policy)
    if utilities is None:
        utilities = policy_utility_components(
            policy_vector,
            electorate,
            spec=consequence_spec,
        ).total_utility
    else:
        utilities = np.asarray(utilities, dtype=float)
    if utilities.shape != (electorate.n_voters,):
        raise ValueError(
            f"utilities must have shape ({electorate.n_voters},), got {utilities.shape}"
        )

    centered_utilities = utilities - utilities.mean()
    scale = max(float(np.abs(centered_utilities).max()), 1e-9)
    normalized_utilities = centered_utilities / scale

    prefs = electorate.preferences.copy()
    to_policy = policy_vector[None, :] - prefs
    updated = prefs + float(feedback.alignment_strength) * to_policy
    updated += float(feedback.utility_sensitivity) * normalized_utilities[:, None] * to_policy

    if feedback.mean_reversion != 0.0:
        center = electorate.mean()
        updated += float(feedback.mean_reversion) * (center[None, :] - prefs)

    if feedback.noise_scale > 0.0:
        updated += rng.normal(scale=float(feedback.noise_scale), size=updated.shape)

    return Electorate(
        preferences=np.clip(updated, 0.0, 1.0),
        dim_names=list(electorate.dim_names) if electorate.dim_names is not None else None,
        group_ids=None if electorate.group_ids is None else electorate.group_ids.copy(),
        group_names=None if electorate.group_names is None else dict(electorate.group_names),
    )


def _policy_vector(policy: PolicyOutcome | np.ndarray) -> np.ndarray:
    if isinstance(policy, PolicyOutcome):
        return policy.vector
    return np.asarray(policy, dtype=float)


def _coerce_policy_outcome(
    value: PolicyOutcome | np.ndarray,
    system_name: str,
    rule: str,
    metadata: dict | None = None,
) -> PolicyOutcome:
    if isinstance(value, PolicyOutcome):
        merged_metadata = dict(metadata or {})
        merged_metadata.update(value.metadata)
        return PolicyOutcome(
            vector=value.vector,
            system_name=value.system_name or system_name,
            rule=value.rule or rule,
            metadata=merged_metadata,
        )
    return PolicyOutcome(
        vector=np.asarray(value, dtype=float),
        system_name=system_name,
        rule=rule,
        metadata=dict(metadata or {}),
    )


def _dimension_weights(
    weights: np.ndarray | None,
    n_dims: int,
) -> np.ndarray:
    if weights is None:
        return np.ones(n_dims, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (n_dims,):
        raise ValueError(f"dimension_weights must have shape ({n_dims},), got {weights.shape}")
    if np.any(weights < 0.0):
        raise ValueError("dimension_weights must be non-negative")
    return weights


def _threshold_vector(
    effect: PolicyThresholdEffect,
    electorate: Electorate,
) -> np.ndarray:
    if effect.threshold is None and effect.threshold_by_group is None:
        raise ValueError("A threshold effect must provide threshold or threshold_by_group.")

    base_threshold = 0.0 if effect.threshold is None else float(effect.threshold)
    thresholds = np.full(electorate.n_voters, base_threshold, dtype=float)

    if effect.threshold_by_group is not None:
        if electorate.group_ids is None:
            raise ValueError("threshold_by_group requires electorate.group_ids to be present")
        for group_id, threshold in effect.threshold_by_group.items():
            thresholds[electorate.group_ids == int(group_id)] = float(threshold)
    return thresholds


__all__ = [
    "PolicyConsequenceSpec",
    "PolicyDefinition",
    "PolicyDefinitionLike",
    "PolicyFeedbackSpec",
    "PolicyOutcome",
    "PolicyResolver",
    "PolicyThresholdEffect",
    "PolicyUtilityFunction",
    "PolicyUtilityComponents",
    "apply_policy_feedback",
    "extract_policy_outcome",
    "policy_utility_components",
]
