import numpy as np

from electoral_sim.candidates import CandidateSet
from electoral_sim.electorate import Electorate
from electoral_sim.metrics import compute_policy_metrics
from electoral_sim.policy import (
    PolicyConsequenceSpec,
    PolicyDefinition,
    PolicyFeedbackSpec,
    PolicyOutcome,
    PolicyThresholdEffect,
    PolicyUtilityComponents,
    apply_policy_feedback,
    extract_policy_outcome,
    policy_utility_components,
)
from electoral_sim.types import ElectionResult


def _grouped_electorate() -> Electorate:
    return Electorate(
        preferences=np.array(
            [
                [0.15, 0.30],
                [0.20, 0.35],
                [0.80, 0.70],
                [0.85, 0.75],
            ],
            dtype=float,
        ),
        dim_names=["economic", "social"],
        group_ids=np.array([0, 0, 1, 1], dtype=int),
        group_names={0: "Left bloc", 1: "Right bloc"},
    )


def _candidates() -> CandidateSet:
    return CandidateSet(
        positions=np.array(
            [
                [0.20, 0.40],
                [0.80, 0.60],
            ],
            dtype=float,
        ),
        labels=["A", "B"],
    )


def test_extract_policy_outcome_supports_pr_resolution_rules():
    result = ElectionResult(
        outcome_position=np.array([0.30, 0.40]),
        centroid_position=np.array([0.45, 0.55]),
        median_legislator_position=np.array([0.60, 0.70]),
        winner_indices=[0, 1],
        seat_shares={0: 0.25, 1: 0.75},
        elimination_order=[],
        system_name="Party-List PR",
        is_pr=True,
    )
    candidates = _candidates()

    assert np.allclose(extract_policy_outcome(result).vector, [0.30, 0.40])
    assert np.allclose(
        extract_policy_outcome(result, policy_rule="centroid").vector,
        [0.45, 0.55],
    )
    assert np.allclose(
        extract_policy_outcome(result, policy_rule="median_legislator").vector,
        [0.60, 0.70],
    )
    assert np.allclose(
        extract_policy_outcome(result, candidates=candidates, policy_rule="seat_share_mean").vector,
        0.25 * candidates.positions[0] + 0.75 * candidates.positions[1],
    )

    custom_definition = PolicyDefinition(
        name="blended_compromise",
        resolver=lambda election_result, candidates=None: (
            0.5 * election_result.outcome_position + 0.5 * election_result.centroid_position
        ),
    )
    assert np.allclose(
        extract_policy_outcome(result, policy_rule=custom_definition).vector,
        [0.375, 0.475],
    )


def test_policy_utility_components_and_group_metrics_are_computed():
    electorate = _grouped_electorate()
    policy = PolicyOutcome(vector=np.array([0.25, 0.55]), system_name="Score Voting")
    spec = PolicyConsequenceSpec(
        distance_weight=1.5,
        public_goods_dim=1,
        public_goods_weight=0.4,
        group_adjustments={0: 0.2, 1: -0.1},
    )

    components = policy_utility_components(policy, electorate, spec=spec)
    assert components.total_utility.shape == (electorate.n_voters,)
    assert np.allclose(
        components.total_utility,
        components.distance_utility
        + components.public_goods_utility
        + components.group_adjustment_utility,
    )

    summary = compute_policy_metrics(policy, electorate, consequence_spec=spec)
    assert summary.system_name == "Score Voting"
    assert len(summary.groups) == 2
    assert summary.mean_public_goods_utility > 0.0
    assert np.isfinite(summary.majority_minority_utility_gap)
    assert summary.max_group_utility_gap >= 0.0


def test_custom_policy_utility_definition_is_supported():
    electorate = _grouped_electorate()
    policy = PolicyOutcome(vector=np.array([0.30, 0.65]), system_name="Approval Voting")

    def custom_utility(policy_value, electorate_value, spec):
        policy_vector = policy_value.vector if isinstance(policy_value, PolicyOutcome) else np.asarray(policy_value)
        economic_gain = policy_vector[0] - electorate_value.preferences[:, 0]
        civic_gain = 0.5 * policy_vector[1] * np.ones(electorate_value.n_voters, dtype=float)
        zeros = np.zeros(electorate_value.n_voters, dtype=float)
        return PolicyUtilityComponents(
            total_utility=economic_gain + civic_gain,
            distance_utility=economic_gain,
            public_goods_utility=civic_gain,
            group_adjustment_utility=zeros,
            policy_distance=np.abs(policy_vector[0] - electorate_value.preferences[:, 0]),
        )

    spec = PolicyConsequenceSpec(utility_function=custom_utility)
    components = policy_utility_components(policy, electorate, spec=spec)
    assert np.allclose(
        components.total_utility,
        components.distance_utility + components.public_goods_utility,
    )

    summary = compute_policy_metrics(policy, electorate, consequence_spec=spec)
    assert summary.mean_public_goods_utility > 0.0
    assert summary.utility_std >= 0.0


def test_threshold_effect_can_flatten_distance_utility_below_floor():
    electorate = _grouped_electorate()
    policy = PolicyOutcome(vector=np.array([0.35, 0.55]), system_name="Plurality")
    spec = PolicyConsequenceSpec(
        threshold_effects=(
            PolicyThresholdEffect(
                dimension=0,
                threshold=0.50,
                distance_utility_below=-2.5,
            ),
        )
    )

    components = policy_utility_components(policy, electorate, spec=spec)
    assert np.allclose(components.distance_utility, -components.policy_distance)
    assert np.allclose(components.threshold_utility, -2.5 - components.distance_utility)
    assert np.allclose(components.total_utility, -2.5)

    summary = compute_policy_metrics(policy, electorate, consequence_spec=spec)
    assert np.isclose(summary.mean_threshold_utility, (-2.5 - components.distance_utility).mean())


def test_threshold_effect_can_vary_by_group():
    electorate = _grouped_electorate()
    policy = PolicyOutcome(vector=np.array([0.45, 0.55]), system_name="Approval Voting")
    spec = PolicyConsequenceSpec(
        threshold_effects=(
            PolicyThresholdEffect(
                dimension=0,
                threshold=0.40,
                threshold_by_group={1: 0.60},
                distance_utility_below=-1.8,
                utility_offset_below=-0.2,
            ),
        )
    )

    components = policy_utility_components(policy, electorate, spec=spec)
    # Majority group clears the shared floor, minority group does not.
    assert np.allclose(components.threshold_utility[:2], 0.0)
    expected_minority_adjustment = (-1.8 - components.distance_utility[2:]) - 0.2
    assert np.allclose(components.threshold_utility[2:], expected_minority_adjustment)

    summary = compute_policy_metrics(policy, electorate, consequence_spec=spec)
    assert summary.groups[0].mean_threshold_utility != summary.groups[1].mean_threshold_utility


def test_threshold_effects_also_apply_with_custom_utility_functions():
    electorate = _grouped_electorate()
    policy = PolicyOutcome(vector=np.array([0.45, 0.65]), system_name="Score Voting")

    def custom_utility(policy_value, electorate_value, spec):
        policy_vector = policy_value.vector if isinstance(policy_value, PolicyOutcome) else np.asarray(policy_value)
        distance = np.abs(policy_vector[0] - electorate_value.preferences[:, 0])
        distance_utility = -distance
        public_goods = np.full(electorate_value.n_voters, 0.2 * policy_vector[1], dtype=float)
        zeros = np.zeros(electorate_value.n_voters, dtype=float)
        return PolicyUtilityComponents(
            total_utility=distance_utility + public_goods,
            distance_utility=distance_utility,
            public_goods_utility=public_goods,
            group_adjustment_utility=zeros,
            policy_distance=distance,
        )

    spec = PolicyConsequenceSpec(
        utility_function=custom_utility,
        threshold_effects=(
            PolicyThresholdEffect(
                dimension=0,
                threshold=0.40,
                threshold_by_group={1: 0.60},
                distance_utility_below=-1.7,
            ),
        ),
    )

    components = policy_utility_components(policy, electorate, spec=spec)
    assert np.allclose(components.threshold_utility[:2], 0.0)
    assert np.all(components.threshold_utility[2:] < 0.0)


def test_apply_policy_feedback_preserves_labels_and_moves_relative_to_utility():
    electorate = _grouped_electorate()
    policy = PolicyOutcome(vector=np.array([0.25, 0.35]), system_name="Plurality")
    utilities = np.array([1.0, 0.8, -0.8, -1.0], dtype=float)

    updated = apply_policy_feedback(
        electorate,
        policy,
        feedback=PolicyFeedbackSpec(
            alignment_strength=0.02,
            utility_sensitivity=0.10,
            mean_reversion=0.0,
            noise_scale=0.0,
        ),
        utilities=utilities,
    )

    assert updated.group_names == electorate.group_names
    assert np.array_equal(updated.group_ids, electorate.group_ids)
    assert np.all(updated.preferences >= 0.0)
    assert np.all(updated.preferences <= 1.0)

    start_dists = np.linalg.norm(electorate.preferences - policy.vector, axis=1)
    end_dists = np.linalg.norm(updated.preferences - policy.vector, axis=1)
    assert end_dists[0] < start_dists[0]
    assert end_dists[1] < start_dists[1]
    assert end_dists[3] > start_dists[3]
