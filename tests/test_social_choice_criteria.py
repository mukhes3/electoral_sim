import numpy as np

from electoral_sim.ballots import BallotProfile
from electoral_sim.candidates import fixed_candidates
from electoral_sim.social_choice import borda_social_ranking, plurality_social_ranking
from electoral_sim.utils import (
    check_iia,
    check_non_dictatorship,
    check_unanimity,
    compare_iia,
    find_iia_violations,
    unanimous_preference_pairs,
)


def make_consensus_case():
    rankings = np.array(
        [
            [0, 1, 2],
            [0, 1, 2],
            [0, 1, 2],
            [0, 1, 2],
        ],
        dtype=int,
    )
    ballots = BallotProfile(
        plurality=rankings[:, 0].copy(),
        rankings=rankings,
        scores=np.zeros((4, 3), dtype=float),
        approvals=np.ones((4, 3), dtype=int),
        distances=np.zeros((4, 3), dtype=float),
        approval_threshold=1.0,
        n_voters=4,
        n_candidates=3,
        active_voter_mask=np.ones(4, dtype=bool),
    )
    candidates = fixed_candidates(
        [[0.20, 0.50], [0.50, 0.50], [0.80, 0.50]],
        ["A", "B", "C"],
    )
    return ballots, candidates


def make_borda_iia_case():
    rankings = np.array(
        [
            [1, 2, 0],
            [1, 2, 0],
            [2, 0, 1],
            [2, 0, 1],
            [2, 0, 1],
        ],
        dtype=int,
    )
    ballots = BallotProfile(
        plurality=rankings[:, 0].copy(),
        rankings=rankings,
        scores=np.zeros((5, 3), dtype=float),
        approvals=np.ones((5, 3), dtype=int),
        distances=np.zeros((5, 3), dtype=float),
        approval_threshold=1.0,
        n_voters=5,
        n_candidates=3,
        active_voter_mask=np.ones(5, dtype=bool),
    )
    candidates = fixed_candidates(
        [[0.15, 0.50], [0.50, 0.50], [0.85, 0.50]],
        ["A", "B", "C"],
    )
    return ballots, candidates


def test_unanimity_check_and_profile_dictatorship_diagnostic():
    ballots, candidates = make_consensus_case()
    ranking = plurality_social_ranking(ballots, candidates)

    unanimous_pairs = unanimous_preference_pairs(ballots)
    unanimity = check_unanimity(ballots, ranking, candidates)
    non_dictatorship = check_non_dictatorship(ballots, ranking)

    assert set(unanimous_pairs) == {(0, 1), (0, 2), (1, 2)}
    assert unanimity.satisfied is True
    assert unanimity.violating_pairs == []
    assert non_dictatorship.satisfied is False
    assert non_dictatorship.matching_voter_indices == [0, 1, 2, 3]


def test_compare_iia_detects_borda_violation_when_third_candidate_removed():
    ballots, candidates = make_borda_iia_case()

    comparison = compare_iia(
        ballots=ballots,
        ranking_builder=borda_social_ranking,
        candidate_a=0,
        candidate_b=1,
        removed_candidates=[2],
        candidates=candidates,
    )

    assert comparison.satisfied is False
    assert comparison.candidate_labels == ("A", "B")
    assert comparison.removed_labels == ["C"]
    assert comparison.full_preference is False
    assert comparison.restricted_preference is True


def test_check_iia_finds_borda_violation():
    ballots, candidates = make_borda_iia_case()

    criterion = check_iia(ballots, borda_social_ranking, candidates)
    violations = find_iia_violations(ballots, borda_social_ranking, candidates)

    assert criterion.satisfied is False
    assert ("A", "B") in criterion.violating_labels
    assert len(violations) >= 1
    assert any(v.candidate_pair == (0, 1) and v.removed_candidates == [2] for v in violations)
