import numpy as np

from electoral_sim.candidates import fixed_candidates
from electoral_sim.electorate import Electorate
from electoral_sim.utils import pairwise_majority_between, pairwise_majority_vote


def test_pairwise_majority_vote_finds_majority_winner_in_one_dimension():
    electorate = Electorate(
        np.array([[0.10], [0.20], [0.40], [0.60], [0.90]], dtype=float),
        dim_names=["x"],
    )

    result = pairwise_majority_vote(
        electorate,
        candidate_positions=np.array([[0.25], [0.75]], dtype=float),
        candidate_labels=("Left", "Right"),
    )

    assert result.winner_label == "Left"
    assert result.winner_local_index == 0
    assert result.vote_counts == (3.0, 2.0)
    assert np.isclose(sum(result.vote_shares), 1.0)
    assert result.tied_voter_count == 0


def test_pairwise_majority_vote_splits_tied_voters_evenly():
    electorate = Electorate(
        np.array([[0.25], [0.50], [0.75]], dtype=float),
        dim_names=["x"],
    )

    result = pairwise_majority_vote(
        electorate,
        candidate_positions=np.array([[0.25], [0.75]], dtype=float),
        candidate_labels=("A", "B"),
    )

    assert result.vote_counts == (1.5, 1.5)
    assert result.winner_label is None
    assert result.winner_local_index is None
    assert result.tied_voter_count == 1


def test_pairwise_majority_between_uses_candidate_labels():
    electorate = Electorate(
        np.array([[0.2, 0.5], [0.3, 0.5], [0.9, 0.5]], dtype=float),
        dim_names=["x", "y"],
    )
    candidates = fixed_candidates(
        [[0.25, 0.5], [0.80, 0.5], [0.50, 0.5]],
        ["Left", "Right", "Center"],
    )

    result = pairwise_majority_between(electorate, candidates, 0, 1)

    assert result.candidate_labels == ("Left", "Right")
    assert result.winner_label == "Left"
