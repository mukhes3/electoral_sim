import numpy as np
import pandas as pd
import pytest

from notebooks.helpers.liquid_experiments import (
    ExperimentConfig, absence_order, assign_delegates, make_case,
    run_participation_comparison, run_system_comparison, summarize_by_seed,
)


@pytest.fixture(scope="module")
def config():
    return ExperimentConfig(seeds=(3, 7), n_voters=40, network_draws=2,
                            families=("Asymmetric",), slates=("Broad coverage",),
                            delegation_fractions=(0.0, 1.0), absence_fractions=(0.0, 0.5))


def test_same_electorate_across_slates():
    first, c1 = make_case("Polarized", "Broad coverage", 13, 50)
    second, c2 = make_case("Polarized", "No central candidate", 13, 50)
    np.testing.assert_array_equal(first.preferences, second.preferences)
    np.testing.assert_array_equal(first.group_ids, second.group_ids)
    np.testing.assert_array_equal(c2.positions, c1.positions[[0, 1, 3, 4]])


@pytest.mark.parametrize("mechanism", ["Nearby", "Random", "Popular"])
def test_selected_voters_represented_by_available_casters(mechanism):
    preferences = np.array([[0.0, 0.0], [0.2, 0.2], [0.8, 0.8], [1.0, 1.0]])
    p = assign_delegates(preferences, [0, 3], [1, 2], mechanism, np.random.default_rng(4))
    r = p.resolve()
    assert r.represented_power == 4
    assert r.unrepresented_power == 0
    assert set(r.destinations) <= {1, 2}
    assert r.effective_weights[[0, 3]].sum() == 0


def test_concentrated_absence_prioritizes_target_bloc():
    e, _ = make_case("Asymmetric", "Broad coverage", 9, 80)
    order = absence_order(e, "Bloc-concentrated absence", np.random.default_rng(6))
    target = max(np.unique(e.group_ids), key=lambda g: e.preferences[e.group_ids == g, 0].mean())
    count = (e.group_ids == target).sum()
    assert np.all(e.group_ids[order[:count]] == target)
    assert sorted(order) == list(range(80))


def test_system_baselines_and_pairing(config):
    direct, liquid = run_system_comparison(config)
    assert len(direct) == 2 * 7  # Direct results are not duplicated by network draws.
    assert np.allclose(liquid[liquid.fraction == 0].distance_delta, 0)
    assert np.all(liquid[liquid.fraction == 0].winner_changed == 0)
    assert np.all(liquid.represented_power == config.n_voters)
    assert np.all(liquid.unrepresented_power == 0)
    assert np.allclose(liquid.prefer_liquid + liquid.prefer_direct + liquid.indifferent, 1)
    assert np.all(liquid[liquid.system == "Score"].distance_delta >= -1e-12)
    # Identical seeds reproduce the entire experiment.
    d2, l2 = run_system_comparison(config)
    pd.testing.assert_frame_equal(direct, d2)
    pd.testing.assert_frame_equal(liquid, l2)


def test_participation_accounting_and_reference(config):
    result = run_participation_comparison(config)
    assert np.allclose(result[result.fraction == 0].reference_distance, 0)
    assert np.all(result[result.condition == "Full participation"].winner_matches_full == 1)
    delegated = result[result.condition == "Delegation"]
    assert np.all(delegated.represented_power == config.n_voters)
    assert np.all(delegated.unrepresented_power == 0)
    absent = result[result.condition == "Abstention"]
    np.testing.assert_allclose(absent.represented_power + absent.absent_power, config.n_voters)
    assert np.all(result[result.system == "Score"].distance_delta_full >= -1e-12)
    keys = ["seed", "draw", "pattern", "fraction", "system"]
    # Every mechanism uses the same abstention winner and same unavailable subset size.
    assert np.all(result.groupby(keys).abstention_winner.nunique() == 1)
    assert np.all(result.groupby(keys).actual_fraction.nunique() == 1)


def test_seed_summary_does_not_treat_repeated_draws_as_independent():
    frame = pd.DataFrame({"family": ["A"] * 3, "slate": ["S"] * 3,
                          "seed": [0, 0, 1], "system": ["X"] * 3, "loss": [0, 0, 2]})
    summary = summarize_by_seed(frame, ["system"], ["loss"], bootstrap_draws=500)
    assert summary.iloc[0]["mean"] == 1  # Not the raw-row mean of 2/3.
    assert summary.iloc[0].n_seeds == 2
    repeated = pd.concat([frame, frame[frame.seed == 0]] * 4, ignore_index=True)
    pd.testing.assert_frame_equal(summary, summarize_by_seed(repeated, ["system"], ["loss"], 500))


def test_invalid_absence_fraction_rejected():
    with pytest.raises(ValueError):
        ExperimentConfig(absence_fractions=(1.0,))
