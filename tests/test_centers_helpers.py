import numpy as np

from electoral_sim.electorate import Electorate
from electoral_sim.types import ElectionResult
from electoral_sim.utils import compare_outcome_to_centers, compute_electorate_centers


def test_compute_electorate_centers_matches_electorate_methods():
    electorate = Electorate(
        np.array(
            [
                [0.1, 0.2],
                [0.3, 0.4],
                [0.7, 0.8],
            ],
            dtype=float,
        ),
        dim_names=["x", "y"],
    )

    centers = compute_electorate_centers(electorate)

    assert np.allclose(centers.mean, electorate.mean())
    assert np.allclose(centers.componentwise_median, electorate.componentwise_median())
    assert np.allclose(centers.geometric_median, electorate.geometric_median())


def test_compare_outcome_to_centers_reports_distances_for_selected_position():
    electorate = Electorate(
        np.array(
            [
                [0.2, 0.2],
                [0.3, 0.3],
                [0.8, 0.7],
            ],
            dtype=float,
        ),
        dim_names=["x", "y"],
    )
    result = ElectionResult(
        outcome_position=np.array([0.25, 0.25]),
        centroid_position=np.array([0.40, 0.40]),
        median_legislator_position=np.array([0.25, 0.25]),
        winner_indices=[0],
        seat_shares={0: 1.0},
        elimination_order=[],
        system_name="Test",
    )

    comparison = compare_outcome_to_centers(result, electorate)
    centroid_comparison = compare_outcome_to_centers(
        result,
        electorate,
        position_attr="centroid_position",
    )

    assert comparison.position_name == "outcome_position"
    assert np.allclose(comparison.position, [0.25, 0.25])
    assert comparison.distance_to_mean >= 0.0
    assert comparison.distance_to_componentwise_median >= 0.0
    assert comparison.distance_to_geometric_median >= 0.0
    assert centroid_comparison.position_name == "centroid_position"
    assert np.allclose(centroid_comparison.position, [0.40, 0.40])


def test_compare_outcome_to_centers_rejects_unknown_position_attr():
    electorate = Electorate(np.array([[0.5], [0.6]], dtype=float), dim_names=["x"])
    result = ElectionResult(
        outcome_position=np.array([0.5]),
        centroid_position=np.array([0.5]),
        median_legislator_position=np.array([0.5]),
        winner_indices=[0],
        seat_shares={0: 1.0},
        elimination_order=[],
        system_name="Test",
    )

    try:
        compare_outcome_to_centers(result, electorate, position_attr="missing_position")
    except ValueError as exc:
        assert "missing_position" in str(exc)
    else:
        raise AssertionError("Expected ValueError for an unknown ElectionResult position attribute.")
