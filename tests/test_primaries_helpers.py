import numpy as np

from electoral_sim.candidates import CandidateSet
from electoral_sim.electorate import Electorate
from electoral_sim.primaries import (
    PartySpec,
    PrimaryType,
    run_party_primary,
    run_open_primary_top_k,
    run_primary_monte_carlo,
    run_two_party_primary,
    summarize_primary_result,
)
from electoral_sim.strategies import SincereStrategy
from electoral_sim.systems import InstantRunoff, Plurality


def make_two_party_case():
    preferences = np.vstack(
        [
            np.tile([0.12], (25, 1)),
            np.tile([0.32], (25, 1)),
            np.tile([0.68], (25, 1)),
            np.tile([0.88], (25, 1)),
        ]
    )
    electorate = Electorate(preferences, dim_names=["economic"])
    candidates = CandidateSet(
        np.array([[0.10], [0.35], [0.65], [0.90]]),
        ["L-Base", "L-Moderate", "R-Moderate", "R-Base"],
    )
    parties = [
        PartySpec("Left Party", [0, 1], Plurality()),
        PartySpec("Right Party", [2, 3], Plurality()),
    ]
    return electorate, candidates, parties


def make_open_primary_case():
    preferences = np.vstack(
        [
            np.tile([0.10], (28, 1)),
            np.tile([0.28], (24, 1)),
            np.tile([0.48], (18, 1)),
            np.tile([0.72], (16, 1)),
            np.tile([0.90], (14, 1)),
        ]
    )
    electorate = Electorate(preferences, dim_names=["economic"])
    candidates = CandidateSet(
        np.array([[0.10], [0.30], [0.50], [0.70], [0.90]]),
        ["A", "B", "C", "D", "E"],
    )
    parties = [
        PartySpec("Left Party", [0, 1], Plurality()),
        PartySpec("Right Party", [3, 4], Plurality()),
    ]
    return electorate, candidates, parties


def test_run_two_party_primary_accepts_custom_memberships_and_stage_callables():
    electorate, candidates, parties = make_two_party_case()
    memberships = {
        "Left Party": np.array([True] * 20 + [False] * 80),
        "Right Party": np.array([False] * 80 + [True] * 20),
    }

    primary_calls = []
    general_calls = []

    def primary_strategy(party, primary_electorate, party_candidates):
        primary_calls.append((party.name, primary_electorate.n_voters, party_candidates.n_candidates))
        return SincereStrategy()

    def general_strategy(stage_electorate, stage_candidates, _stage_meta):
        general_calls.append((stage_electorate.n_voters, tuple(stage_candidates.labels)))
        return SincereStrategy()

    result = run_two_party_primary(
        electorate,
        candidates,
        parties,
        general_system=Plurality(),
        primary_type=PrimaryType.CLOSED,
        memberships=memberships,
        primary_strategy=primary_strategy,
        general_strategy=general_strategy,
    )

    assert [pr.n_primary_voters for pr in result.primary_results] == [20, 20]
    assert [call[0] for call in primary_calls] == ["Left Party", "Right Party"]
    assert len(general_calls) == 2

    row = summarize_primary_result(result, pipeline_name="Closed primary")
    assert row["pipeline_type"] == "two_party_primary"
    assert row["primary_voters_by_party"] == {"Left Party": 20, "Right Party": 20}


def test_run_party_primary_defaults_match_explicit_sincere_strategy():
    electorate, candidates, parties = make_two_party_case()
    memberships = {
        "Left Party": np.array([True] * 50 + [False] * 50),
        "Right Party": np.array([False] * 50 + [True] * 50),
    }
    party = parties[0]

    default_result = run_party_primary(
        electorate,
        candidates,
        party,
        memberships[party.name],
        PrimaryType.CLOSED,
    )
    explicit_result = run_party_primary(
        electorate,
        candidates,
        party,
        memberships[party.name],
        PrimaryType.CLOSED,
        strategy=SincereStrategy(),
    )

    assert default_result.nominee_index == explicit_result.nominee_index
    assert default_result.primary_system_name == explicit_result.primary_system_name
    assert default_result.n_primary_voters == explicit_result.n_primary_voters
    assert default_result.primary_vote_share == explicit_result.primary_vote_share
    assert default_result.eliminated_indices == explicit_result.eliminated_indices
    assert (
        default_result.primary_metrics.distance_to_median
        == explicit_result.primary_metrics.distance_to_median
    )


def test_run_two_party_primary_defaults_match_explicit_sincere_strategies():
    electorate, candidates, parties = make_two_party_case()

    default_result = run_two_party_primary(
        electorate,
        candidates,
        parties,
        general_system=Plurality(),
        primary_type=PrimaryType.CLOSED,
    )
    explicit_result = run_two_party_primary(
        electorate,
        candidates,
        parties,
        general_system=Plurality(),
        primary_type=PrimaryType.CLOSED,
        primary_strategy=lambda *_args: SincereStrategy(),
        general_strategy=lambda *_args: SincereStrategy(),
    )

    assert [pr.nominee_index for pr in default_result.primary_results] == [
        pr.nominee_index for pr in explicit_result.primary_results
    ]
    assert default_result.general_result.winner_indices == explicit_result.general_result.winner_indices
    assert np.allclose(
        default_result.general_result.outcome_position,
        explicit_result.general_result.outcome_position,
    )
    assert (
        default_result.general_metrics.distance_to_median
        == explicit_result.general_metrics.distance_to_median
    )
    assert (
        default_result.baseline_metrics.distance_to_median
        == explicit_result.baseline_metrics.distance_to_median
    )
    assert default_result.primary_divergence == explicit_result.primary_divergence


def test_run_open_primary_top_k_returns_finalists_and_summary():
    electorate, candidates, _ = make_open_primary_case()

    result = run_open_primary_top_k(
        electorate,
        candidates,
        general_system=InstantRunoff(),
        top_k=3,
    )

    assert result.primary_result.finalist_indices == [0, 1, 2]
    assert result.primary_result.finalist_labels == ["A", "B", "C"]
    assert result.primary_result.top_k == 3
    assert result.primary_result.n_primary_voters == electorate.n_voters

    row = summarize_primary_result(result, pipeline_name="Top-3 open primary")
    assert row["pipeline_type"] == "open_primary_top_k"
    assert row["finalist_labels"] == ["A", "B", "C"]
    assert row["winner"] in {"A", "B", "C"}


def test_run_primary_monte_carlo_returns_trial_rows_for_multiple_pipelines():
    def scenario_factory(rng=None):
        _ = rng
        return make_two_party_case()

    pipelines = {
        "Closed primary": lambda electorate, candidates, parties: run_two_party_primary(
            electorate,
            candidates,
            parties,
            general_system=Plurality(),
            primary_type=PrimaryType.CLOSED,
        ),
        "Top-3 open primary": lambda electorate, candidates, _parties: run_open_primary_top_k(
            electorate,
            candidates,
            general_system=InstantRunoff(),
            top_k=3,
        ),
    }

    rows = run_primary_monte_carlo(
        scenario_factory,
        pipelines,
        n_trials=3,
        rng=np.random.default_rng(7),
    )

    assert len(rows) == 6
    assert {row["pipeline"] for row in rows} == {"Closed primary", "Top-3 open primary"}
    assert {row["trial"] for row in rows} == {0, 1, 2}
    assert all("distance_to_median" in row for row in rows)
