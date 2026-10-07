import numpy as np
import pytest

from electoral_sim.ballots import BallotProfile
from electoral_sim.candidates import fixed_candidates
from electoral_sim.electorate import Electorate
from electoral_sim.liquid import DelegationProfile, LiquidApprovalVoting, LiquidScoreVoting
from electoral_sim.metrics import run_simulation
from electoral_sim.systems import ApprovalVoting, ScoreVoting


def election():
    electorate = Electorate(np.array([[0.0], [0.1], [1.0]]))
    candidates = fixed_candidates([[0.0], [1.0]])
    ballots = BallotProfile.from_preferences(electorate, candidates, 0.2)
    return electorate, candidates, ballots


def test_chains_and_merging_preserve_power():
    result = DelegationProfile([1, 2, -1, 1, -1]).resolve()
    np.testing.assert_array_equal(result.destinations, [2, 2, 2, 2, 4])
    np.testing.assert_array_equal(result.effective_weights, [0, 0, 4, 0, 1])
    assert result.participating_power == result.represented_power == 5
    assert result.unrepresented_power == 0


def test_cycle_members_cast_own_ballots_and_incoming_chains_stop_at_entry():
    result = DelegationProfile([1, 2, 1, 0, 2, 5]).resolve()
    np.testing.assert_array_equal(result.destinations, [1, 1, 2, 1, 2, 5])
    np.testing.assert_array_equal(result.effective_weights, [0, 3, 2, 0, 0, 1])
    assert result.cycles == ((1, 2),)


def test_disjoint_cycles():
    result = DelegationProfile([1, 0, 3, 2]).resolve()
    np.testing.assert_array_equal(result.effective_weights, np.ones(4))
    assert len(result.cycles) == 2


def test_abstention_breaks_chains_without_creating_power():
    result = DelegationProfile([1, 2, -1, -1]).resolve([True, False, True, True])
    np.testing.assert_array_equal(result.destinations, [-1, -1, 2, 3])
    np.testing.assert_array_equal(result.effective_weights, [0, 0, 1, 1])
    assert result.participating_power == 3
    assert result.represented_power == 2
    assert result.unrepresented_power == 1


def test_long_chain_avoids_recursion_limit():
    delegates = np.arange(1, 10001)
    delegates[-1] = -1
    result = DelegationProfile(delegates).resolve()
    assert result.effective_weights[-1] == 10000


@pytest.mark.parametrize("delegates", [[], [[-1]], [0.0], [True], [-2], [1], ["0"]])
def test_invalid_targets_rejected(delegates):
    with pytest.raises(ValueError):
        DelegationProfile(delegates)


@pytest.mark.parametrize("mask", [[True], [1, 0], [[True, False]]])
def test_invalid_active_masks_rejected(mask):
    with pytest.raises(ValueError):
        DelegationProfile([-1, -1]).resolve(mask)


def test_target_input_and_property_do_not_mutate_profile():
    targets = np.array([1, -1])
    profile = DelegationProfile(targets)
    targets[0] = -1
    profile.delegates[0] = -1
    np.testing.assert_array_equal(profile.resolve().effective_weights, [0, 2])


@pytest.mark.parametrize("ordinary,liquid,field", [
    (ApprovalVoting, LiquidApprovalVoting, "approval_counts"),
    (ScoreVoting, LiquidScoreVoting, "mean_scores"),
])
@pytest.mark.parametrize("mask", [[True, True, True], [True, False, True]])
def test_no_delegation_matches_direct_voting(ordinary, liquid, field, mask):
    _, candidates, ballots = election()
    ballots.active_voter_mask = np.array(mask)
    direct = ordinary().run(ballots, candidates)
    delegated = liquid(DelegationProfile([-1, 1, -1])).run(ballots, candidates)
    assert delegated.winner_indices == direct.winner_indices
    np.testing.assert_allclose(delegated.metadata[field], direct.metadata[field])
    np.testing.assert_array_equal(delegated.outcome_position, direct.outcome_position)


@pytest.mark.parametrize("system,field", [
    (LiquidApprovalVoting, "approval_counts"),
    (LiquidScoreVoting, "mean_scores"),
])
def test_delegation_changes_winner_using_terminal_ballot(system, field):
    _, candidates, ballots = election()
    original = ballots.scores.copy()
    result = system(DelegationProfile([1, 2, -1])).run(ballots, candidates)
    assert result.winner_indices == [1]
    np.testing.assert_array_equal(result.outcome_position, [1.0])
    expected = [0, 3] if field == "approval_counts" else [0, 1]
    np.testing.assert_allclose(result.metadata[field], expected)
    np.testing.assert_array_equal(ballots.scores, original)
    np.testing.assert_array_equal(ballots.active_voter_mask, [True, True, True])


@pytest.mark.parametrize("system", [LiquidApprovalVoting, LiquidScoreVoting])
def test_no_represented_power_rejected(system):
    _, candidates, ballots = election()
    ballots.active_voter_mask[:] = False
    with pytest.raises(ValueError, match="no represented"):
        system(DelegationProfile([-1, -1, -1])).run(ballots, candidates)
    ballots.active_voter_mask[0] = True
    with pytest.raises(ValueError, match="no represented"):
        system(DelegationProfile([1, -1, -1])).run(ballots, candidates)


def test_failed_delegation_excluded_from_denominator():
    _, candidates, ballots = election()
    ballots.active_voter_mask[:] = [True, False, True]
    profile = DelegationProfile([1, -1, -1])
    for system, field in [(LiquidApprovalVoting, "approval_rates"),
                          (LiquidScoreVoting, "mean_scores")]:
        result = system(profile).run(ballots, candidates)
        np.testing.assert_allclose(result.metadata[field], [0, 1])
        assert result.metadata["unrepresented_power"] == 1


def test_profile_size_must_match_ballots():
    _, candidates, ballots = election()
    with pytest.raises(ValueError, match="voter count"):
        LiquidScoreVoting(DelegationProfile([-1])).run(ballots, candidates)


def test_existing_simulation_pipeline_evaluates_true_electorate():
    electorate, candidates, _ = election()
    metrics = run_simulation(electorate, candidates, [
        LiquidScoreVoting(DelegationProfile([1, 2, -1]))
    ])
    assert metrics[0].system_name == "Liquid Score Voting"
    assert metrics[0].mean_voter_distance == pytest.approx(1.9 / 3)


def test_random_graphs_match_independent_per_voter_walk():
    rng = np.random.default_rng(173)
    for _ in range(100):
        n = 30
        targets = rng.integers(-1, n, size=n)
        active = rng.random(n) > 0.2
        expected = np.full(n, -1)
        for voter in range(n):
            node = voter
            seen = set()
            while active[node]:
                if node in seen or targets[node] in (-1, node):
                    expected[voter] = node
                    break
                seen.add(node)
                node = targets[node]
        result = DelegationProfile(targets).resolve(active)
        np.testing.assert_array_equal(result.destinations, expected)
        assert result.represented_power + result.unrepresented_power == active.sum()
        assert np.all(result.effective_weights[~active] == 0)


@pytest.mark.parametrize("liquid,ordinary,field", [
    (LiquidApprovalVoting, ApprovalVoting, "approval_counts"),
    (LiquidScoreVoting, ScoreVoting, "mean_scores"),
])
def test_weighted_tallies_match_explicit_ballot_replication(liquid, ordinary, field):
    _, candidates, ballots = election()
    # Supplied ballots may be strategic and need not match spatial distances.
    ballots.scores[:] = [[0.1, 0.6], [0.8, 0.3], [0.2, 0.9]]
    ballots.approvals[:] = [[0, 1], [1, 1], [0, 1]]
    profile = DelegationProfile([1, -1, -1])
    rows = np.array([1, 1, 2])
    replicated = BallotProfile(
        plurality=ballots.plurality[rows], rankings=ballots.rankings[rows],
        scores=ballots.scores[rows], approvals=ballots.approvals[rows],
        distances=ballots.distances[rows], approval_threshold=0.2,
        n_voters=3, n_candidates=2,
    )
    actual = liquid(profile).run(ballots, candidates)
    expected = ordinary().run(replicated, candidates)
    np.testing.assert_allclose(actual.metadata[field], expected.metadata[field])
    assert actual.winner_indices == expected.winner_indices
