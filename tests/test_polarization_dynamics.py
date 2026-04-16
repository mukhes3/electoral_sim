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

from notebooks.helpers.polarization_dynamics import (
    ORACLE_ORDER,
    RATIO_ORDER,
    advance_candidates,
    advance_voters,
    build_candidate_dynamics,
    build_system,
    choose_oracle_outcome,
    compute_next_step_polarization,
    fractional_sigma_name,
    build_polarization_candidates,
    build_polarization_electorate,
    build_voter_dynamics,
    compare_trajectory_to_baseline,
    compute_camp_asymmetry_metrics,
    compute_polarization_metrics,
    illustrate_candidate_mechanisms,
    illustrate_voter_mechanisms,
    parse_fractional_sigma,
    plot_baseline_difference_trajectories,
    plot_fractional_sigma_sweep,
    plot_oracle_start_end_maps,
    plot_polarization_trajectories,
    plot_polarization_metric_heatmap,
    plot_polarization_tradeoff_scatter,
    plot_ratio_system_heatmap,
    plot_start_end_maps,
    polarization_helper_overview,
    run_polarization_comparison_grid,
    run_oracle_trajectory,
    run_polarization_trajectory,
    run_polarization_trajectory_grid,
    summarize_fractional_sigma_sweep,
    summarize_trajectory_changes,
    trace_oracle_state,
)


def test_polarization_helper_overview_mentions_core_components():
    overview = polarization_helper_overview()
    assert isinstance(overview, pd.DataFrame)
    assert "Voter dynamics" in overview["component"].tolist()
    assert "Polarization metrics" in overview["component"].tolist()
    assert "Camp balance" in overview["component"].tolist()
    assert "Oracle benchmarks" in overview["component"].tolist()


def test_ratio_order_lists_added_balance_variants():
    assert RATIO_ORDER == ["Original", "70:30", "50:50"]
    assert ORACLE_ORDER == ["Geometric median oracle", "Depolarization oracle"]


def test_fractional_sigma_name_and_parser_support_arbitrary_sigma_values():
    name = fractional_sigma_name(0.75)
    assert name == "Fractional (sigma=0.75)"
    assert parse_fractional_sigma(name) == 0.75
    assert parse_fractional_sigma("Fractional") == 0.3
    assert parse_fractional_sigma("Plurality") is None


def test_fractional_system_builder_uses_continuous_variant():
    system = build_system("Fractional (sigma=0.45)")
    assert system.__class__.__name__ == "FractionalBallotDiscrete"
    assert np.isclose(system.sigma, 0.45)


def test_build_polarization_electorate_supports_ratio_variants():
    electorate = build_polarization_electorate(
        "Asymmetric resentment",
        seed=17,
        n_voters=6000,
        ratio_name="70:30",
    )
    labels = electorate.group_labels()
    left_ids = [group_id for group_id, name in labels.items() if name in {"Left bloc", "Center-left"}]
    right_ids = [group_id for group_id, name in labels.items() if name in {"Center-right", "Right edge"}]
    left_share = np.isin(electorate.group_ids, left_ids).mean()
    right_share = np.isin(electorate.group_ids, right_ids).mean()
    assert abs(left_share - 0.70) < 0.03
    assert abs(right_share - 0.30) < 0.03


def test_compute_polarization_metrics_returns_expected_fields():
    electorate = build_polarization_electorate("Bridge conflict", seed=3, n_voters=400)
    candidates = build_polarization_candidates("Centrist ladder")

    metrics = compute_polarization_metrics(
        electorate,
        candidates=candidates,
        winner_position=candidates.positions[2],
        seed=11,
    )

    for key in [
        "voter_dispersion",
        "voter_pairwise_distance",
        "voter_p90_distance",
        "pc1_bimodality",
        "candidate_pairwise_distance",
        "group_center_gap",
        "winner_to_center_distance",
    ]:
        assert key in metrics
        assert metrics[key] >= 0.0


def test_oracle_helpers_return_valid_outputs():
    electorate = build_polarization_electorate("Bridge conflict", seed=21, n_voters=320)

    median_point, median_value = choose_oracle_outcome(
        electorate,
        oracle_name="Geometric median oracle",
        voter_dynamics="Backlash",
    )
    depol_point, depol_value = choose_oracle_outcome(
        electorate,
        oracle_name="Depolarization oracle",
        voter_dynamics="Backlash",
    )

    assert median_point.shape == (electorate.n_dims,)
    assert depol_point.shape == (electorate.n_dims,)
    assert np.all((0.0 <= median_point) & (median_point <= 1.0))
    assert np.all((0.0 <= depol_point) & (depol_point <= 1.0))
    assert median_value >= 0.0
    assert depol_value >= 0.0
    assert compute_next_step_polarization(electorate, depol_point, "Backlash") >= 0.0


def test_advance_voters_preserves_shape_bounds_and_groups():
    electorate = build_polarization_electorate("Two blocs", seed=5, n_voters=300)
    dynamics = build_voter_dynamics("Backlash")

    advanced = advance_voters(
        electorate,
        winner_position=[0.8, 0.4],
        dynamics=dynamics,
        seed=7,
    )

    assert advanced.preferences.shape == electorate.preferences.shape
    assert advanced.group_labels() == electorate.group_labels()
    assert advanced.preferences.min() >= 0.0
    assert advanced.preferences.max() <= 1.0


def test_advance_candidates_keeps_positions_in_unit_square():
    electorate = build_polarization_electorate("Asymmetric resentment", seed=9, n_voters=450)
    candidates = build_polarization_candidates("Asymmetric insurgency")
    dynamics = build_candidate_dynamics("Broad coalition chase")

    advanced = advance_candidates(electorate, candidates, dynamics)

    assert advanced.positions.shape == candidates.positions.shape
    assert advanced.positions.min() >= 0.0
    assert advanced.positions.max() <= 1.0


def test_run_polarization_trajectory_smoke():
    electorate = build_polarization_electorate("Bridge conflict", seed=4, n_voters=350)
    candidates = build_polarization_candidates("Centrist ladder")

    trajectory = run_polarization_trajectory(
        electorate,
        candidates,
        system_names=["Plurality", "Approval"],
        n_steps=4,
        voter_dynamics="Sorting pressure",
        candidate_dynamics="Base reinforcement",
        seed=12,
    )

    assert len(trajectory) == 8
    assert set(trajectory["system"]) == {"Plurality", "Approval"}
    assert set(trajectory["step"]) == {0, 1, 2, 3}
    assert "voter_pairwise_distance" in trajectory.columns
    assert "distance_to_median" in trajectory.columns
    assert "normalized_displacement_asymmetry" in trajectory.columns


def test_run_oracle_trajectory_and_state_trace_smoke():
    electorate = build_polarization_electorate("Bridge conflict", seed=24, n_voters=280)
    candidates = build_polarization_candidates("Polarized elites")

    trajectory = run_oracle_trajectory(
        electorate,
        candidates,
        n_steps=4,
        voter_dynamics="Sorting pressure",
        candidate_dynamics="Base reinforcement",
        electorate_name="Bridge conflict",
        seed=12,
    )

    assert len(trajectory) == 8
    assert set(trajectory["system"]) == set(ORACLE_ORDER)
    assert "oracle_objective_value" in trajectory.columns
    assert "distance_to_median" in trajectory.columns

    state_map = {
        name: trace_oracle_state(
            electorate,
            candidates,
            oracle_name=name,
            n_steps=3,
            voter_dynamics="Sorting pressure",
            candidate_dynamics="Base reinforcement",
            seed=5,
        )
        for name in ORACLE_ORDER
    }
    fig = plot_oracle_start_end_maps(state_map, ORACLE_ORDER)
    assert len(fig.axes) == 4
    plt.close(fig)


def test_summarize_trajectory_changes_returns_start_end_deltas():
    electorate = build_polarization_electorate("Bridge conflict", seed=4, n_voters=300)
    candidates = build_polarization_candidates("Centrist ladder")
    trajectory = run_polarization_trajectory(
        electorate,
        candidates,
        system_names=["Plurality", "Fractional (sigma=0.3)", "Fractional (sigma=1.0)"],
        n_steps=4,
        seed=3,
    )

    summary = summarize_trajectory_changes(trajectory)

    assert set(summary["system"]) == {
        "Plurality",
        "Fractional (sigma=0.3)",
        "Fractional (sigma=1.0)",
    }
    assert "voter_pairwise_distance_delta" in summary.columns
    assert "distance_to_median_delta" in summary.columns


def test_run_polarization_comparison_grid_smoke():
    grid = run_polarization_comparison_grid(
        electorate_names=["Two blocs"],
        candidate_names=["Centrist ladder"],
        voter_dynamics_names=["Consensus pull"],
        candidate_dynamics_names=["Static candidates", "Broad coalition chase"],
        ratio_names=["Original", "50:50"],
        system_names=["Plurality", "IRV"],
        n_steps=3,
        n_voters=240,
        seed=10,
    )

    assert len(grid) == 8
    assert set(grid["system"]) == {"Plurality", "IRV"}
    assert set(grid["candidate_dynamics"]) == {"Static candidates", "Broad coalition chase"}
    assert set(grid["ratio"]) == {"Original", "50:50"}


def test_run_polarization_trajectory_grid_and_baseline_comparison_smoke():
    trajectory_grid = run_polarization_trajectory_grid(
        electorate_names=["Two blocs"],
        candidate_names=["Centrist ladder"],
        voter_dynamics_names=["Consensus pull"],
        candidate_dynamics_names=["Static candidates"],
        ratio_names=["Original", "70:30"],
        system_names=["Plurality", "IRV"],
        n_steps=3,
        n_voters=240,
        seed=14,
    )

    assert set(trajectory_grid["system"]) == {"Plurality", "IRV"}
    assert set(trajectory_grid["step"]) == {0, 1, 2}
    assert "case_id" in trajectory_grid.columns
    assert set(trajectory_grid["ratio"]) == {"Original", "70:30"}

    relative = compare_trajectory_to_baseline(
        trajectory_grid,
        baseline_system="Plurality",
        metrics=["voter_pairwise_distance", "winner_to_center_distance"],
    )
    assert "voter_pairwise_distance_vs_Plurality" in relative.columns
    assert "winner_to_center_distance_vs_Plurality" in relative.columns
    plurality_rows = relative[relative["system"] == "Plurality"]
    assert (plurality_rows["voter_pairwise_distance_vs_Plurality"].abs() < 1e-12).all()


def test_plot_polarization_trajectories_returns_figure():
    electorate = build_polarization_electorate("Two blocs", seed=2, n_voters=250)
    candidates = build_polarization_candidates("Polarized elites")
    trajectory = run_polarization_trajectory(
        electorate,
        candidates,
        system_names=["Plurality", "IRV"],
        n_steps=3,
        seed=8,
    )

    fig = plot_polarization_trajectories(
        trajectory,
        metrics=["voter_pairwise_distance", "pc1_bimodality"],
    )

    assert len(fig.axes) == 2
    plt.close(fig)


def test_mechanism_illustration_helpers_return_figures():
    electorate = build_polarization_electorate("Two blocs", seed=2, n_voters=200)
    candidates = build_polarization_candidates("Polarized elites")

    fig = illustrate_voter_mechanisms(
        electorate,
        winner_position=candidates.positions[2],
        model_names=["Consensus pull", "Backlash"],
        seed=1,
    )
    assert len(fig.axes) == 4
    plt.close(fig)

    fig = illustrate_candidate_mechanisms(
        electorate,
        candidates,
        model_names=["Static candidates", "Base reinforcement"],
    )
    assert len(fig.axes) == 4
    plt.close(fig)


def test_start_end_and_heatmap_helpers_return_figures():
    electorate = build_polarization_electorate("Bridge conflict", seed=8, n_voters=260)
    candidates = build_polarization_candidates("Centrist ladder")

    fig = plot_start_end_maps(
        electorate,
        candidates,
        system_names=["Plurality", "Fractional (sigma=0.3)", "Fractional (sigma=1.0)"],
        n_steps=3,
        seed=6,
    )
    assert len(fig.axes) >= 3
    plt.close(fig)

    grid = run_polarization_comparison_grid(
        electorate_names=["Two blocs"],
        candidate_names=["Centrist ladder"],
        voter_dynamics_names=["Backlash", "Sorting pressure"],
        candidate_dynamics_names=["Static candidates"],
        system_names=["Plurality", "Fractional (sigma=0.3)", "Fractional (sigma=1.0)"],
        n_steps=3,
        n_voters=240,
        seed=11,
    )
    fig = plot_polarization_metric_heatmap(
        grid,
        metric_delta="voter_pairwise_distance_delta",
    )
    assert len(fig.axes) >= 1
    plt.close(fig)

    summary = (
        grid.groupby("system", as_index=False)
        .agg(
            mean_voter_pairwise_delta=("voter_pairwise_distance_delta", "mean"),
            mean_winner_center_delta=("winner_to_center_distance_delta", "mean"),
        )
    )
    fig = plot_polarization_tradeoff_scatter(summary)
    assert len(fig.axes) == 1
    plt.close(fig)

    fig = plot_ratio_system_heatmap(
        grid,
        metric="normalized_displacement_asymmetry_delta",
    )
    assert len(fig.axes) >= 1
    plt.close(fig)

    trajectory_grid = run_polarization_trajectory_grid(
        electorate_names=["Two blocs"],
        candidate_names=["Centrist ladder"],
        voter_dynamics_names=["Backlash", "Sorting pressure"],
        candidate_dynamics_names=["Static candidates"],
        system_names=["Plurality", "IRV", "Fractional (sigma=0.3)"],
        n_steps=3,
        n_voters=200,
        seed=12,
    )
    relative = compare_trajectory_to_baseline(trajectory_grid, baseline_system="Plurality")
    fig = plot_baseline_difference_trajectories(
        relative,
        baseline_system="Plurality",
        metrics=[
            "voter_pairwise_distance_vs_Plurality",
            "winner_to_center_distance_vs_Plurality",
        ],
        systems=["IRV", "Fractional (sigma=0.3)"],
    )
    assert len(fig.axes) == 2
    plt.close(fig)


def test_fractional_alias_and_explicit_variants_are_supported():
    electorate = build_polarization_electorate("Two blocs", seed=13, n_voters=220)
    candidates = build_polarization_candidates("Centrist ladder")

    trajectory = run_polarization_trajectory(
        electorate,
        candidates,
        system_names=["Fractional", "Fractional (sigma=0.3)", "Fractional (sigma=1.0)"],
        n_steps=2,
        seed=9,
    )

    assert set(trajectory["system"]) == {
        "Fractional",
        "Fractional (sigma=0.3)",
        "Fractional (sigma=1.0)",
    }


def test_arbitrary_fractional_sigma_name_runs_through_builder_and_sweep_summary():
    electorate = build_polarization_electorate("Two blocs", seed=19, n_voters=220)
    candidates = build_polarization_candidates("Centrist ladder")

    trajectory = run_polarization_trajectory(
        electorate,
        candidates,
        system_names=["Plurality", "Fractional (sigma=0.45)", "Fractional (sigma=1.25)"],
        n_steps=2,
        seed=20,
    )
    assert set(trajectory["system"]) == {
        "Plurality",
        "Fractional (sigma=0.45)",
        "Fractional (sigma=1.25)",
    }

    summary = summarize_trajectory_changes(trajectory)
    sweep = summarize_fractional_sigma_sweep(summary)
    assert sweep["sigma"].tolist() == [0.45, 1.25]

    fig = plot_fractional_sigma_sweep(sweep)
    assert len(fig.axes) == 1
    plt.close(fig)


def test_compute_camp_asymmetry_metrics_identifies_larger_and_smaller_camps():
    electorate = build_polarization_electorate(
        "Asymmetric resentment",
        seed=33,
        n_voters=1500,
        ratio_name="70:30",
    )
    advanced = advance_voters(
        electorate,
        winner_position=[0.75, 0.40],
        dynamics=build_voter_dynamics("Backlash"),
        seed=34,
    )
    metrics = compute_camp_asymmetry_metrics(
        advanced,
        electorate,
        electorate_name="Asymmetric resentment",
    )
    assert metrics["majority_share"] > 0.65
    assert np.isfinite(metrics["normalized_displacement_asymmetry"])
