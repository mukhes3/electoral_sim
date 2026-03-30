"""Utility helpers for the electoral simulator."""
from electoral_sim.utils.viz_electorate import (
    plot_electorate,
    plot_election_result,
    plot_all_systems_spatial,
)
from electoral_sim.utils.viz_metrics import (
    plot_metric_bar,
    plot_grouped_metrics,
    plot_scenario_heatmap,
    plot_radar,
    plot_monte_carlo_distributions,
)
from electoral_sim.utils.social_choice_criteria import (
    IIAComparison,
    PairwiseCriterionCheck,
    ProfileDictatorshipDiagnostic,
    check_iia,
    check_non_dictatorship,
    check_unanimity,
    compare_iia,
    find_dictatorial_voters,
    find_iia_violations,
    unanimous_preference_pairs,
)

__all__ = [
    "plot_electorate",
    "plot_election_result",
    "plot_all_systems_spatial",
    "plot_metric_bar",
    "plot_grouped_metrics",
    "plot_scenario_heatmap",
    "plot_radar",
    "plot_monte_carlo_distributions",
    "PairwiseCriterionCheck",
    "ProfileDictatorshipDiagnostic",
    "IIAComparison",
    "unanimous_preference_pairs",
    "check_unanimity",
    "find_dictatorial_voters",
    "check_non_dictatorship",
    "compare_iia",
    "find_iia_violations",
    "check_iia",
]
