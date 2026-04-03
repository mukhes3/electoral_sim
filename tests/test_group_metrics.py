import numpy as np
import pytest

from electoral_sim.candidates import fixed_candidates
from electoral_sim.electorate import Electorate, gaussian_mixture_electorate
from electoral_sim.metrics import compute_group_metrics
from electoral_sim.types import ElectionResult


def test_gaussian_mixture_remains_unlabeled_when_no_groups_are_provided():
    electorate = gaussian_mixture_electorate(
        200,
        components=[
            {"weight": 0.5, "mean": [0.3, 0.5], "cov": [[0.01, 0.0], [0.0, 0.01]]},
            {"weight": 0.5, "mean": [0.7, 0.5], "cov": [[0.01, 0.0], [0.0, 0.01]]},
        ],
        rng=np.random.default_rng(1),
    )

    assert electorate.group_ids is None
    assert electorate.group_names is None
    assert electorate.has_groups is False


def test_gaussian_mixture_can_assign_the_same_group_to_multiple_components():
    electorate = gaussian_mixture_electorate(
        300,
        components=[
            {
                "weight": 0.3,
                "mean": [0.2, 0.4],
                "cov": [[0.01, 0.0], [0.0, 0.01]],
                "group": "Group A",
            },
            {
                "weight": 0.2,
                "mean": [0.4, 0.6],
                "cov": [[0.01, 0.0], [0.0, 0.01]],
                "group": "Group A",
            },
            {
                "weight": 0.5,
                "mean": [0.75, 0.5],
                "cov": [[0.01, 0.0], [0.0, 0.01]],
                "group": "Group B",
            },
        ],
        rng=np.random.default_rng(2),
    )

    assert electorate.has_groups is True
    assert set(electorate.group_names.values()) == {"Group A", "Group B"}
    assert len(np.unique(electorate.group_ids)) == 2

    subsample = electorate.subsample(75, rng=np.random.default_rng(3))
    assert subsample.has_groups is True
    assert set(subsample.group_names.values()) == {"Group A", "Group B"}
    assert subsample.group_ids.shape == (75,)


def test_compute_group_metrics_returns_expected_group_welfare_summary():
    electorate = Electorate(
        preferences=np.array(
            [
                [0.15, 0.50],
                [0.25, 0.50],
                [0.75, 0.50],
                [0.85, 0.50],
            ]
        ),
        group_ids=np.array([0, 0, 1, 1]),
        group_names={0: "Group A", 1: "Group B"},
    )
    candidates = fixed_candidates(
        [[0.20, 0.50], [0.80, 0.50]],
        ["Left", "Right"],
    )
    result = ElectionResult(
        outcome_position=np.array([0.20, 0.50]),
        centroid_position=np.array([0.20, 0.50]),
        median_legislator_position=np.array([0.20, 0.50]),
        winner_indices=[0],
        seat_shares={0: 1.0},
        elimination_order=[],
        system_name="Test System",
        is_pr=False,
    )

    summary = compute_group_metrics(result, electorate, candidates)
    by_name = {group.group_name: group for group in summary.groups}

    assert summary.system_name == "Test System"
    assert pytest.approx(summary.max_mean_distance_gap) == 0.55
    assert pytest.approx(summary.max_majority_satisfaction_gap) == 1.0
    assert pytest.approx(summary.min_group_welfare) == -0.60

    assert by_name["Group A"].n_voters == 2
    assert pytest.approx(by_name["Group A"].population_share) == 0.5
    assert pytest.approx(by_name["Group A"].mean_voter_distance) == 0.05
    assert pytest.approx(by_name["Group A"].majority_satisfaction) == 1.0
    assert pytest.approx(by_name["Group A"].welfare) == -0.05

    assert by_name["Group B"].n_voters == 2
    assert pytest.approx(by_name["Group B"].mean_voter_distance) == 0.60
    assert pytest.approx(by_name["Group B"].majority_satisfaction) == 0.0
    assert pytest.approx(by_name["Group B"].welfare) == -0.60


def test_compute_group_metrics_requires_group_labels():
    electorate = Electorate(
        preferences=np.array(
            [
                [0.20, 0.50],
                [0.80, 0.50],
            ]
        )
    )
    candidates = fixed_candidates([[0.20, 0.50], [0.80, 0.50]], ["A", "B"])
    result = ElectionResult(
        outcome_position=np.array([0.20, 0.50]),
        centroid_position=np.array([0.20, 0.50]),
        median_legislator_position=np.array([0.20, 0.50]),
        winner_indices=[0],
        seat_shares={0: 1.0},
        elimination_order=[],
        system_name="Test System",
        is_pr=False,
    )

    with pytest.raises(ValueError, match="group_ids"):
        compute_group_metrics(result, electorate, candidates)
