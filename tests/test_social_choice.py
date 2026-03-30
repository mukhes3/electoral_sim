import numpy as np

from electoral_sim.ballots import BallotProfile
from electoral_sim.candidates import fixed_candidates
from electoral_sim.electorate import Electorate
from electoral_sim.social_choice import (
    approval_social_ranking,
    borda_social_ranking,
    condorcet_winner,
    copeland_social_ranking,
    find_majority_cycle,
    pairwise_summary,
    plurality_social_ranking,
    score_social_ranking,
    social_prefers,
)


def make_cycle_case():
    """Construct a BallotProfile with the classic A>B>C>A majority cycle."""
    rankings = np.array(
        [
            [0, 1, 2],
            [0, 1, 2],
            [0, 1, 2],
            [1, 2, 0],
            [1, 2, 0],
            [1, 2, 0],
            [2, 0, 1],
            [2, 0, 1],
            [2, 0, 1],
        ],
        dtype=int,
    )
    plurality = rankings[:, 0].copy()
    ballots = BallotProfile(
        plurality=plurality,
        rankings=rankings,
        scores=np.zeros((9, 3), dtype=float),
        approvals=np.ones((9, 3), dtype=int),
        distances=np.zeros((9, 3), dtype=float),
        approval_threshold=1.0,
        n_voters=9,
        n_candidates=3,
        active_voter_mask=np.ones(9, dtype=bool),
    )
    candidates = fixed_candidates(
        [[0.00, 0.00], [1.00, 0.00], [0.50, 0.90]],
        ["A", "B", "C"],
    )
    return ballots, candidates


def make_linear_case():
    electorate = Electorate(
        np.array(
            [
                [0.00, 0.50],
                [0.10, 0.50],
                [0.20, 0.50],
                [0.45, 0.50],
                [0.50, 0.50],
                [0.55, 0.50],
                [0.80, 0.50],
                [0.90, 0.50],
            ]
        )
    )
    candidates = fixed_candidates(
        [[0.10, 0.50], [0.50, 0.50], [0.90, 0.50]],
        ["Left", "Center", "Right"],
    )
    return electorate, candidates


def test_restrict_to_candidates_preserves_relative_ranking():
    electorate, candidates = make_linear_case()
    ballots = BallotProfile.from_preferences(electorate, candidates)

    restricted = ballots.restrict_to_candidates([0, 2])

    assert restricted.n_candidates == 2
    assert np.all(np.isin(restricted.rankings, [-1, 0, 1]))
    assert restricted.plurality.shape == ballots.plurality.shape
    assert restricted.active_voter_mask.sum() == ballots.active_voter_mask.sum()


def test_majority_cycle_detection_finds_cycle():
    ballots, candidates = make_cycle_case()

    cycle = find_majority_cycle(ballots)
    summary = pairwise_summary(ballots, candidates)

    assert cycle is not None
    assert cycle[0] == cycle[-1]
    assert len(cycle) >= 4
    assert summary.condorcet_winner is None


def test_condorcet_winner_detected_in_linear_case():
    electorate, candidates = make_linear_case()
    ballots = BallotProfile.from_preferences(electorate, candidates)

    winner = condorcet_winner(ballots)
    summary = pairwise_summary(ballots, candidates)

    assert winner == 1
    assert summary.condorcet_winner == 1
    assert summary.majority_cycle is None


def test_social_rankings_return_labeled_orders():
    electorate, candidates = make_linear_case()
    ballots = BallotProfile.from_preferences(electorate, candidates)

    plurality = plurality_social_ranking(ballots, candidates)
    borda = borda_social_ranking(ballots, candidates)
    approval = approval_social_ranking(ballots, candidates)
    score = score_social_ranking(ballots, candidates)
    copeland = copeland_social_ranking(ballots, candidates)

    assert set(plurality.candidate_order) == {0, 1, 2}
    assert borda.ordered_labels[0] == "Center"
    assert score.ordered_labels[0] == "Center"
    assert copeland.ordered_labels[0] == "Center"
    assert set(approval.candidate_order) == {0, 1, 2}

    assert social_prefers(borda, 1, 0) is True
