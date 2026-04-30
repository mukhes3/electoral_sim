import pandas as pd
from pathlib import Path
import sys

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from notebooks.helpers.representation_vs_policy_consequences import (
    CASE_ORDER,
    POLICY_DEFINITION_ORDER,
    SYSTEM_ORDER,
    build_exposure_sensitive_spec,
    build_threshold_sensitive_spec,
    case_comparison_table,
    case_reference_table,
    compare_consequence_models,
    compare_policy_definitions,
    consequence_model_reference_table,
    hidden_harm_cases,
    plot_case_model_heatmaps,
    plot_case_policy_points,
    plot_consequence_model_comparison,
    plot_hidden_harm_scatter,
    plot_policy_definition_map,
    plot_policy_heatmaps,
    plot_representation_policy_contrasts,
    representation_policy_helper_overview,
    run_case_grid,
    run_case_systems,
    summarize_case_takeaway,
    summarize_overall_conclusion,
    system_reference_table,
)


def test_helper_overview_mentions_cases_and_policy_definitions():
    overview = representation_policy_helper_overview()
    assert "Static cases" in overview["component"].tolist()
    assert "Policy definitions" in overview["component"].tolist()

    cases = case_reference_table()
    systems = system_reference_table()
    models = consequence_model_reference_table()
    assert set(cases["case"]) == set(CASE_ORDER)
    assert set(systems["system"]) == set(SYSTEM_ORDER)
    assert {"Distance-only", "Exposure-sensitive", "Threshold-sensitive"} == set(models["utility_type"])


def test_run_case_systems_returns_expected_columns():
    df = run_case_systems(
        "Moderate burden shift",
        consequence_spec=build_exposure_sensitive_spec(),
        seed=4,
        n_voters=900,
    )
    assert set(df["system"]) == set(SYSTEM_ORDER)
    required = {
        "representation_aggregate_welfare",
        "representation_minority_welfare",
        "policy_aggregate_utility",
        "policy_minority_utility",
        "policy_majority_minority_gap",
    }
    assert required <= set(df.columns)
    assert "policy_threshold_component" in df.columns


def test_compare_policy_definitions_smoke():
    df = compare_policy_definitions(
        case_name="Fragmented legislature",
        consequence_spec=build_exposure_sensitive_spec(),
        seed=5,
        n_voters=900,
    )
    assert set(df["definition"]) == set(POLICY_DEFINITION_ORDER)
    assert set(df["policy_rule"]) == {"outcome", "centroid", "blended_compromise"}


def test_compare_consequence_models_smoke():
    df = compare_consequence_models(
        case_name="Moderate burden shift",
        seed=5,
        n_voters=900,
    )
    assert {"Distance-only", "Exposure-sensitive", "Threshold-sensitive"} == set(df["consequence_model"])
    threshold_rows = df[df["consequence_model"] == "Threshold-sensitive"]
    assert "policy_threshold_component" in threshold_rows.columns
    assert threshold_rows["policy_threshold_component"].abs().max() >= 0.0

    table = case_comparison_table(df, "Moderate burden shift")
    assert "Aggregate utility" in table.columns.get_level_values(0)
    takeaway = summarize_case_takeaway(df, "Moderate burden shift")
    assert "plurality" in takeaway.lower() or "all six systems" in takeaway.lower()


def test_grid_summary_and_plot_helpers_smoke():
    df = run_case_grid(
        case_names=CASE_ORDER,
        consequence_spec=build_exposure_sensitive_spec(),
        seed=6,
        n_voters=800,
    )
    ranked = hidden_harm_cases(df)
    assert len(ranked) == len(df)

    fig1 = plot_representation_policy_contrasts(df, case_name="Fragmented legislature")
    fig2 = plot_policy_heatmaps(df)
    fig3 = plot_hidden_harm_scatter(df)
    fig3a = plot_case_model_heatmaps(
        compare_consequence_models(
            case_name="Moderate burden shift",
            seed=6,
            n_voters=800,
        ),
        case_name="Moderate burden shift",
    )
    fig3aa = plot_case_policy_points(
        compare_consequence_models(
            case_name="Moderate burden shift",
            seed=6,
            n_voters=800,
        ),
        case_name="Moderate burden shift",
        seed=6,
        n_voters=800,
    )
    fig3b = plot_consequence_model_comparison(
        compare_consequence_models(
            case_name="Moderate burden shift",
            seed=6,
            n_voters=800,
        ),
        case_name="Moderate burden shift",
    )
    fig4 = plot_policy_definition_map(
        compare_policy_definitions(
            case_name="Fragmented legislature",
            consequence_spec=build_exposure_sensitive_spec(),
            seed=6,
            n_voters=800,
        ),
        case_name="Fragmented legislature",
        seed=6,
        n_voters=800,
    )
    assert fig1 is not None and fig2 is not None and fig3 is not None and fig3b is not None and fig4 is not None
    conclusion = summarize_overall_conclusion(
        pd.concat(
            [
                compare_consequence_models(case_name=case_name, seed=6, n_voters=800)
                for case_name in CASE_ORDER
            ],
            ignore_index=True,
        )
    )
    assert conclusion
    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)
    plt.close(fig3a)
    plt.close(fig3aa)
    plt.close(fig3b)
    plt.close(fig4)
