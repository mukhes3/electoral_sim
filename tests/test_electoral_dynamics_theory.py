from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from electoral_sim.ballots import BallotProfile
from notebooks.helpers import (
    THEORY_BASELINE_SYSTEMS,
    THEORY_CANDIDATE_MODEL_ORDER,
    THEORY_CANDIDATE_ORDER,
    THEORY_ELECTORATE_ORDER,
    THEORY_MU0_CANDIDATE_MODELS,
    THEORY_ORACLE_ORDER,
    THEORY_RATIO_ORDER,
    THEORY_VOTER_MODEL_ORDER,
    build_fractional_tradeoff_cases,
    build_theory_candidate_dynamics,
    choose_theory_oracle_outcome,
    compute_candidate_electorate_center_gap,
    compute_next_step_voter_variance,
    fractional_continuous_name,
    plot_theory_uncertainty_trajectories,
    run_theory_oracle_replicates,
    run_theory_oracle_trajectory,
    search_fractional_interior_optima,
    summarize_asymmetry_by_system,
    summarize_theory_fractional_continuous_sweep,
    trace_theory_oracle_state,
)
from notebooks.helpers.electoral_dynamics_theory import (
    approximate_minimax_center,
    build_theory_system,
    compute_candidate_variance,
    compute_coverage_gap,
    compute_mean_supporter_centroid_distance,
    compute_mean_winner_distance,
    compute_supporter_centroid_radius,
    compute_supporter_centroids,
    compute_theory_metrics,
    compute_voter_variance,
    compute_weighted_polarization_cost,
    compute_winner_radius,
    distance_to_candidate_convex_hull,
    parse_fractional_continuous_sigma,
    plot_fractional_weighted_objective_curves,
    run_theory_grid,
    run_theory_trajectory,
    summarize_theory_trajectory_changes,
    supporter_weight_matrix,
    theory_helper_overview,
)
from notebooks.helpers.polarization_dynamics import (
    build_polarization_candidates,
    build_polarization_electorate,
)


def _example_inputs(seed: int = 7, n_voters: int = 240):
    electorate = build_polarization_electorate("Bridge conflict", seed=seed, n_voters=n_voters)
    candidates = build_polarization_candidates("Centrist ladder")
    ballots = BallotProfile.from_preferences(electorate, candidates)
    return electorate, candidates, ballots


def test_theory_helpers_are_exported_with_prefixed_orders():
    assert THEORY_ELECTORATE_ORDER == ["Two blocs", "Bridge conflict", "Asymmetric resentment"]
    assert THEORY_RATIO_ORDER == ["Original", "70:30", "50:50"]
    assert "Broad coalition chase" in THEORY_CANDIDATE_MODEL_ORDER
    assert "Backlash" in THEORY_VOTER_MODEL_ORDER
    assert "Centrist ladder" in THEORY_CANDIDATE_ORDER
    assert THEORY_BASELINE_SYSTEMS == ["Plurality", "IRV", "Approval", "Score", "Condorcet"]
    assert THEORY_MU0_CANDIDATE_MODELS == ["Static candidates", "Base reinforcement (mu=0)"]
    assert THEORY_ORACLE_ORDER == ["Centrality oracle", "Depolarization oracle"]


def test_theory_helper_overview_mentions_core_paper_primitives():
    overview = theory_helper_overview()
    assert isinstance(overview, pd.DataFrame)
    assert "Winner primitive" in overview["component"].tolist()
    assert "Supporter primitive" in overview["component"].tolist()
    assert "Asymmetry and theory-backed candidate runs" in overview["component"].tolist()
    assert "Coverage geometry" in overview["component"].tolist()
    assert "Oracle objective comparison" in overview["component"].tolist()


def test_fractional_continuous_name_and_parser_round_trip():
    name = fractional_continuous_name(0.45)
    assert name == "Fractional Continuous (sigma=0.45)"
    assert parse_fractional_continuous_sigma(name) == 0.45
    assert parse_fractional_continuous_sigma("Plurality") is None


def test_build_theory_system_supports_continuous_fractional_display_name():
    system = build_theory_system("Fractional Continuous (sigma=0.45)")
    assert system.__class__.__name__ == "FractionalBallotContinuous"
    assert system.sigma == 0.45


def test_build_theory_candidate_dynamics_supports_mu_zero_variant():
    dynamics = build_theory_candidate_dynamics("Base reinforcement (mu=0)")
    assert dynamics.supporter_pull > 0.0
    assert dynamics.electorate_pull == 0.0
    assert dynamics.differentiation_pull > 0.0


def test_supporter_weight_matrix_rows_sum_to_one_across_system_proxies():
    electorate, candidates, ballots = _example_inputs()
    system_names = THEORY_BASELINE_SYSTEMS + [fractional_continuous_name(0.35)]

    for system_name in system_names:
        system = build_theory_system(system_name)
        weights = supporter_weight_matrix(system_name, ballots, candidates, system=system)
        assert weights.shape == (electorate.n_voters, candidates.n_candidates)
        assert np.allclose(weights.sum(axis=1), 1.0)
        assert np.all(weights >= 0.0)


def test_supporter_centroids_and_distances_have_expected_shapes():
    electorate, candidates, ballots = _example_inputs()
    weights = supporter_weight_matrix("Plurality", ballots, candidates)

    centroids, support_mass = compute_supporter_centroids(electorate, candidates, weights)

    assert centroids.shape == candidates.positions.shape
    assert support_mass.shape == (candidates.n_candidates,)
    assert np.isclose(support_mass.sum(), electorate.n_voters)
    assert compute_supporter_centroid_radius(candidates, centroids) >= 0.0
    assert compute_mean_supporter_centroid_distance(candidates, centroids) >= 0.0


def test_theory_metric_primitives_return_nonnegative_values():
    electorate, candidates, ballots = _example_inputs()
    outcome = candidates.positions[2]
    weights = supporter_weight_matrix("Score", ballots, candidates)
    metrics = compute_theory_metrics(electorate, candidates, outcome, weights)

    assert compute_winner_radius(electorate, outcome) >= compute_mean_winner_distance(electorate, outcome)
    assert compute_voter_variance(electorate) >= 0.0
    assert compute_candidate_variance(candidates) >= 0.0
    assert metrics.winner_radius >= 0.0
    assert metrics.supporter_centroid_radius >= 0.0
    assert metrics.candidate_variance >= 0.0


def test_weighted_polarization_cost_checks_alpha_bounds():
    assert np.isclose(compute_weighted_polarization_cost(0.8, 0.2, alpha=0.25), 0.35)
    try:
        compute_weighted_polarization_cost(1.0, 1.0, alpha=1.5)
    except ValueError:
        pass
    else:
        raise AssertionError("alpha outside [0, 1] should raise ValueError")


def test_oracle_helpers_return_valid_positions_and_values():
    electorate, candidates, _ = _example_inputs()

    centrality_point, centrality_value = choose_theory_oracle_outcome(
        electorate,
        oracle_name="Centrality oracle",
        voter_dynamics="Backlash",
    )
    depolarization_point, depolarization_value = choose_theory_oracle_outcome(
        electorate,
        oracle_name="Depolarization oracle",
        voter_dynamics="Backlash",
    )

    assert centrality_point.shape == (electorate.n_dims,)
    assert depolarization_point.shape == (electorate.n_dims,)
    assert np.all((0.0 <= centrality_point) & (centrality_point <= 1.0))
    assert np.all((0.0 <= depolarization_point) & (depolarization_point <= 1.0))
    assert centrality_value >= 0.0
    assert depolarization_value >= 0.0
    assert compute_next_step_voter_variance(electorate, depolarization_point, "Backlash") >= 0.0
    assert compute_candidate_electorate_center_gap(electorate, candidates) >= 0.0


def test_oracle_trajectory_and_state_trace_smoke():
    electorate = build_polarization_electorate("Bridge conflict", seed=11, n_voters=180)
    candidates = build_polarization_candidates("Centrist ladder")

    trajectory = run_theory_oracle_trajectory(
        electorate,
        candidates,
        n_steps=4,
        voter_dynamics="Backlash",
        candidate_dynamics="Broad coalition chase",
        electorate_name="Bridge conflict",
        seed=5,
    )

    assert len(trajectory) == 8
    assert set(trajectory["system"]) == set(THEORY_ORACLE_ORDER)
    assert {"oracle_objective_value", "candidate_electorate_center_gap"} <= set(trajectory.columns)

    state = trace_theory_oracle_state(
        electorate,
        candidates,
        oracle_name="Centrality oracle",
        n_steps=3,
        voter_dynamics="Backlash",
        candidate_dynamics="Broad coalition chase",
        seed=2,
    )
    assert state["end_electorate"].preferences.shape == electorate.preferences.shape
    assert state["end_candidates"].positions.shape == candidates.positions.shape
    assert state["final_outcome"].shape == (electorate.n_dims,)


def test_oracle_replicates_and_uncertainty_plot_smoke():
    replicated = run_theory_oracle_replicates(
        electorate_name="Bridge conflict",
        candidate_name="Polarized elites",
        ratio_name="70:30",
        n_steps=3,
        voter_dynamics="Sorting pressure",
        candidate_dynamics="Base reinforcement (mu=0)",
        n_runs=4,
        n_voters=160,
        seed=31,
    )

    assert set(replicated["system"]) == set(THEORY_ORACLE_ORDER)
    assert set(replicated["run_id"]) == {0, 1, 2, 3}
    fig = plot_theory_uncertainty_trajectories(
        replicated,
        metrics=["winner_radius", "voter_variance", "candidate_electorate_center_gap"],
        system_order=THEORY_ORACLE_ORDER,
    )
    assert len(fig.axes) >= 3
    plt.close(fig)


def test_minimax_center_and_convex_hull_distance_smoke():
    electorate, candidates, _ = _example_inputs()
    center = approximate_minimax_center(electorate)

    assert center.shape == (electorate.n_dims,)
    assert np.all(center >= 0.0)
    assert np.all(center <= 1.0)

    hull_distance = distance_to_candidate_convex_hull(center, candidates)
    coverage_gap = compute_coverage_gap(electorate, candidates)

    assert hull_distance >= 0.0
    assert coverage_gap >= 0.0


def test_run_theory_trajectory_and_summary_smoke():
    electorate = build_polarization_electorate("Two blocs", seed=3, n_voters=180)
    candidates = build_polarization_candidates("Polarized elites")
    system_names = ["Plurality", fractional_continuous_name(0.30)]

    trajectory = run_theory_trajectory(
        electorate,
        candidates,
        system_names=system_names,
        n_steps=4,
        voter_dynamics="Consensus pull",
        candidate_dynamics="Broad coalition chase",
        electorate_name="Two blocs",
        seed=12,
    )

    assert len(trajectory) == 8
    assert set(trajectory["system"]) == set(system_names)
    assert set(trajectory["step"]) == {0, 1, 2, 3}
    assert "winner_radius" in trajectory.columns
    assert "supporter_centroid_radius" in trajectory.columns
    assert "normalized_displacement_asymmetry" in trajectory.columns

    summary = summarize_theory_trajectory_changes(trajectory)
    assert set(summary["system"]) == set(system_names)
    assert "winner_radius_delta" in summary.columns
    assert "candidate_variance_delta" in summary.columns
    assert "normalized_displacement_asymmetry_end" in summary.columns


def test_run_theory_grid_and_fractional_summary_smoke():
    systems = ["Plurality", fractional_continuous_name(0.30), fractional_continuous_name(0.80)]
    summary = run_theory_grid(
        electorate_names=["Two blocs"],
        ratio_names=["50:50"],
        candidate_names=["Centrist ladder"],
        voter_dynamics_names=["Consensus pull"],
        candidate_dynamics_names=["Static candidates"],
        system_names=systems,
        n_steps=3,
        n_voters=180,
        seed=9,
    )

    assert len(summary) == 3
    assert set(summary["system"]) == set(systems)
    assert set(summary["ratio"]) == {"50:50"}

    sigma_summary = summarize_theory_fractional_continuous_sweep(summary)
    assert len(sigma_summary) == 2
    assert "mean_weighted_cost_alpha_0.50" in sigma_summary.columns

    asymmetry_summary = summarize_asymmetry_by_system(summary)
    assert len(asymmetry_summary) == 3
    assert "normalized_displacement_asymmetry_end" in asymmetry_summary.columns


def test_fractional_tradeoff_case_builder_and_search_smoke():
    cases = build_fractional_tradeoff_cases()
    assert len(cases) >= 3

    search_df, minima_df = search_fractional_interior_optima(
        cases=cases[:2],
        sigma_values=[0.05, 0.15, 0.4, 1.0],
        alphas=(0.1, 0.9),
        n_steps=3,
        n_voters=180,
        seed=21,
    )

    assert not search_df.empty
    assert {"case_name", "sigma", "weighted_cost_alpha_0.10", "weighted_cost_alpha_0.90"} <= set(search_df.columns)
    assert not minima_df.empty
    assert {"case_name", "alpha", "best_sigma", "is_interior_minimum"} <= set(minima_df.columns)


def test_plot_fractional_weighted_objective_curves_returns_figure():
    sigma_summary = pd.DataFrame(
        {
            "sigma": [0.2, 0.6, 1.0],
            "mean_weighted_cost_alpha_0.25": [0.8, 0.7, 0.75],
            "mean_weighted_cost_alpha_0.50": [0.9, 0.78, 0.79],
            "mean_weighted_cost_alpha_0.75": [1.0, 0.84, 0.83],
        }
    )

    fig = plot_fractional_weighted_objective_curves(sigma_summary)
    assert len(fig.axes) == 1
    plt.close(fig)
