import numpy as np

from electoral_sim.ballots import BallotProfile
from electoral_sim.candidates import fixed_candidates
from electoral_sim.electorate import Electorate
from electoral_sim.fractional import FractionalBallotContinuous, FractionalBallotDiscrete
from electoral_sim.fractional_reporting import run_fractional_reporting_simulation
from electoral_sim.metrics import compute_fractional_robustness_metrics
from electoral_sim.reporting import CoalitionMisreporting, HonestReporting


def make_fractional_case():
    electorate = Electorate(
        np.array(
            [
                [0.12, 0.50],
                [0.18, 0.52],
                [0.22, 0.48],
                [0.78, 0.50],
                [0.84, 0.52],
            ]
        ),
        dim_names=["economic", "social"],
    )
    candidates = fixed_candidates(
        [[0.15, 0.50], [0.85, 0.50]],
        ["Left", "Right"],
    )
    ballots = BallotProfile.from_preferences(electorate, candidates)
    return electorate, candidates, ballots


def test_fractional_systems_remain_backward_compatible_when_no_reported_positions_are_used():
    electorate, candidates, ballots = make_fractional_case()

    discrete = FractionalBallotDiscrete(sigma=0.1)
    continuous = FractionalBallotContinuous(sigma=0.1)

    old_discrete = discrete.run(ballots, candidates)
    explicit_none_discrete = discrete.run(ballots, candidates, reported_preferences=None)
    old_continuous = continuous.run(ballots, candidates)
    explicit_none_continuous = continuous.run(ballots, candidates, reported_preferences=None)

    assert np.allclose(old_discrete.outcome_position, explicit_none_discrete.outcome_position)
    assert old_discrete.winner_indices == explicit_none_discrete.winner_indices
    assert np.allclose(old_continuous.outcome_position, explicit_none_continuous.outcome_position)
    assert old_continuous.seat_shares == explicit_none_continuous.seat_shares
    assert old_discrete.metadata["used_reported_preferences"] is False
    assert old_continuous.metadata["used_reported_preferences"] is False
    assert electorate.n_voters == ballots.n_voters  # smoke-check fixture consistency


def test_honest_reporting_reproduces_truthful_fractional_outcome():
    electorate, candidates, _ = make_fractional_case()
    system = FractionalBallotContinuous(sigma=0.2)

    run = run_fractional_reporting_simulation(
        electorate,
        candidates,
        system=system,
        reporting_model=HonestReporting(),
    )

    assert np.allclose(run.truthful_result.outcome_position, run.reported_result.outcome_position)
    assert run.robustness.outcome_shift == 0.0
    assert run.robustness.seat_share_shift == 0.0
    assert run.robustness.winner_changed is False


def test_reported_positions_change_fractional_outcomes_and_metadata():
    electorate, candidates, ballots = make_fractional_case()
    system = FractionalBallotContinuous(sigma=0.1)
    reported_preferences = np.repeat([[0.90, 0.50]], repeats=electorate.n_voters, axis=0)

    truthful_result = system.run(ballots, candidates)
    reported_result = system.run(
        ballots,
        candidates,
        reported_preferences=reported_preferences,
    )

    assert not np.allclose(truthful_result.outcome_position, reported_result.outcome_position)
    assert reported_result.metadata["used_reported_preferences"] is True


def test_robustness_metrics_detect_winner_flip_under_strong_coalition_misreporting():
    electorate, candidates, _ = make_fractional_case()
    system = FractionalBallotDiscrete(sigma=0.05)
    all_voters = np.ones(electorate.n_voters, dtype=bool)
    reporting_model = CoalitionMisreporting(
        target_position=candidates.positions[1],
        strength=1.0,
        voter_mask=all_voters,
    )

    run = run_fractional_reporting_simulation(
        electorate,
        candidates,
        system=system,
        reporting_model=reporting_model,
        target_mask=all_voters,
    )

    assert run.truthful_result.winner_indices == [0]
    assert run.reported_result.winner_indices == [1]
    assert run.robustness.winner_changed is True
    assert run.robustness.outcome_shift > 0


def test_compute_fractional_robustness_metrics_is_zero_for_identical_results():
    electorate, candidates, ballots = make_fractional_case()
    system = FractionalBallotContinuous(sigma=0.2)
    result = system.run(ballots, candidates)

    robustness = compute_fractional_robustness_metrics(
        result,
        result,
        electorate,
        candidates,
    )

    assert robustness.outcome_shift == 0.0
    assert robustness.seat_share_shift == 0.0
    assert robustness.winner_changed is False
    assert robustness.distance_to_median_delta == 0.0
    assert robustness.mean_voter_distance_delta == 0.0
