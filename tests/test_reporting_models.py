import numpy as np

from electoral_sim.candidates import fixed_candidates
from electoral_sim.electorate import Electorate
from electoral_sim.reporting import (
    BiasedNoiseReporting,
    CoalitionMisreporting,
    DirectionalExaggerationReporting,
    GaussianNoiseReporting,
    HonestReporting,
    nearest_to_candidate,
)


def make_reporting_case():
    electorate = Electorate(
        np.array(
            [
                [0.20, 0.50],
                [0.25, 0.52],
                [0.70, 0.48],
                [0.78, 0.50],
            ]
        ),
        dim_names=["economic", "social"],
    )
    candidates = fixed_candidates(
        [[0.18, 0.50], [0.80, 0.50]],
        ["Left", "Right"],
    )
    return electorate, candidates


def test_honest_reporting_returns_true_positions():
    electorate, candidates = make_reporting_case()

    reported = HonestReporting().report_positions(electorate, candidates)

    assert np.allclose(reported, electorate.preferences)
    assert reported is not electorate.preferences


def test_gaussian_noise_reporting_preserves_shape_and_bounds():
    electorate, candidates = make_reporting_case()
    model = GaussianNoiseReporting(noise_std=0.1, rng=np.random.default_rng(42))

    reported = model.report_positions(electorate, candidates)

    assert reported.shape == electorate.preferences.shape
    assert np.all(reported >= 0.0) and np.all(reported <= 1.0)
    assert not np.allclose(reported, electorate.preferences)


def test_biased_noise_reporting_only_changes_selected_voters():
    electorate, candidates = make_reporting_case()
    model = BiasedNoiseReporting(
        bias=np.array([0.10, -0.05]),
        voter_mask=np.array([True, False, False, True]),
    )

    reported = model.report_positions(electorate, candidates)

    assert np.allclose(reported[1], electorate.preferences[1])
    assert np.allclose(reported[2], electorate.preferences[2])
    assert not np.allclose(reported[0], electorate.preferences[0])
    assert not np.allclose(reported[3], electorate.preferences[3])


def test_directional_exaggeration_moves_selected_voters_away_from_rival():
    electorate, candidates = make_reporting_case()
    model = DirectionalExaggerationReporting(
        strength=0.15,
        away_from_candidate_idx=1,
        selector=nearest_to_candidate(0),
    )

    reported = model.report_positions(electorate, candidates)
    right_pos = candidates.positions[1]

    original_dists = np.linalg.norm(electorate.preferences[:2] - right_pos, axis=1)
    reported_dists = np.linalg.norm(reported[:2] - right_pos, axis=1)

    assert np.all(reported_dists >= original_dists)
    assert np.allclose(reported[2:], electorate.preferences[2:])


def test_coalition_misreporting_can_snap_selected_voters_to_target():
    electorate, candidates = make_reporting_case()
    target = np.array([0.90, 0.90])
    model = CoalitionMisreporting(
        target_position=target,
        strength=1.0,
        voter_mask=np.array([False, True, True, False]),
    )

    reported = model.report_positions(electorate, candidates)

    assert np.allclose(reported[1], target)
    assert np.allclose(reported[2], target)
    assert np.allclose(reported[0], electorate.preferences[0])
    assert np.allclose(reported[3], electorate.preferences[3])
