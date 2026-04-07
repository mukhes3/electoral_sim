from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from notebooks.helpers.party_primary_strategy_space import (
    build_candidate_slate,
    build_grouped_electorate,
    build_party_positions,
    build_turnout_setup,
    full_space_helper_overview,
    mixed_lhs_helper_overview,
    run_design_case,
    run_full_space_grid,
    run_mixed_lhs_grid,
    sample_mixed_latin_hypercube_design,
)
from electoral_sim.primaries import build_party_specs_from_positions


def test_helper_overview_mentions_full_design_axes():
    overview = full_space_helper_overview()
    assert isinstance(overview, pd.DataFrame)
    assert "Primary pipelines" in overview["dimension"].tolist()
    assert "Strategy profiles" in overview["dimension"].tolist()


def test_mixed_lhs_overview_mentions_intensity_axes():
    overview = mixed_lhs_helper_overview()
    assert "Turnout intensity" in overview["dimension"].tolist()
    assert "Strategy intensity" in overview["dimension"].tolist()


def test_turnout_setup_builds_masks_from_party_positions():
    electorate = build_grouped_electorate("Aligned polarization", "60:40", seed=3, n_voters=300)
    parties = build_party_positions()
    candidates = build_candidate_slate("Balanced ladder")
    party_specs = build_party_specs_from_positions(candidates, parties)

    turnout = build_turnout_setup(
        "Even turnout",
        electorate,
        candidates,
        parties,
        party_specs,
        seed=7,
    )

    assert turnout["general_mask"].shape == (electorate.n_voters,)
    assert set(turnout["closed_memberships"]) == {"Left Party", "Right Party"}


def test_build_grouped_electorate_keeps_both_groups_in_small_extreme_case():
    electorate = build_grouped_electorate("Aligned polarization", "99:1", seed=4, n_voters=120)
    assert set(electorate.group_labels().values()) == {"Majority group", "Minority group"}


def test_build_grouped_electorate_accepts_arbitrary_ratio_labels():
    electorate = build_grouped_electorate("Shared center", "68:32", seed=8, n_voters=250)
    shares = {
        name: (electorate.group_ids == group_id).mean()
        for group_id, name in electorate.group_labels().items()
    }
    assert "Majority group" in shares
    assert "Minority group" in shares
    assert 0.55 <= shares["Majority group"] <= 0.80


def test_run_design_case_smoke():
    row = run_design_case(
        electorate_name="Shared center",
        primary_name="Closed plurality",
        turnout_name="Even turnout",
        strategy_name="Sincere",
        ratio_name="60:40",
        slate_name="Balanced ladder",
        system_name="Plurality",
        seed=11,
        n_voters=400,
    )

    assert row["primary"] == "Closed plurality"
    assert row["system"] == "Plurality"
    assert "minority_group_welfare" in row


def test_run_full_space_grid_subset_returns_dataframe():
    df = run_full_space_grid(
        electorate_names=["Aligned polarization"],
        primary_names=["No primary", "Top-4 open"],
        turnout_names=["Even turnout"],
        strategy_names=["Sincere"],
        ratio_names=["90:10"],
        slate_names=["Bridge-heavy"],
        system_names=["IRV"],
        seed=5,
        n_voters=350,
    )

    assert list(df["primary"]) == ["No primary", "Top-4 open"]
    assert "aggregate_welfare_delta_vs_no_primary" in df.columns


def test_sample_mixed_latin_hypercube_design_returns_expected_columns():
    design = sample_mixed_latin_hypercube_design(n_cases=8, seed=9)
    assert len(design) == 8
    for column in [
        "case_id",
        "majority_share",
        "electorate",
        "primary",
        "turnout",
        "turnout_strength",
        "strategy",
        "strategy_strength",
        "candidate_slate",
    ]:
        assert column in design.columns


def test_run_mixed_lhs_grid_expands_each_case_across_systems():
    design = sample_mixed_latin_hypercube_design(n_cases=4, seed=11)
    df = run_mixed_lhs_grid(
        design,
        system_names=["Plurality", "IRV"],
        seed=13,
        n_voters=250,
    )

    assert len(df) == 8
    assert set(df["system"]) == {"Plurality", "IRV"}
    assert set(df["case_id"]) == {0, 1, 2, 3}
