"""Helpers for comparing election outcomes to different electorate centers."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from electoral_sim.electorate import Electorate
from electoral_sim.types import ElectionResult


@dataclass
class ElectorateCenters:
    """Common center summaries for an electorate."""

    mean: np.ndarray
    componentwise_median: np.ndarray
    geometric_median: np.ndarray


@dataclass
class OutcomeCenterComparison:
    """Distances from an election outcome to several notions of center."""

    position_name: str
    position: np.ndarray
    centers: ElectorateCenters
    distance_to_mean: float
    distance_to_componentwise_median: float
    distance_to_geometric_median: float


def compute_electorate_centers(electorate: Electorate) -> ElectorateCenters:
    """Return three useful notions of the electorate's center."""
    return ElectorateCenters(
        mean=electorate.mean(),
        componentwise_median=electorate.componentwise_median(),
        geometric_median=electorate.geometric_median(),
    )


def compare_outcome_to_centers(
    result: ElectionResult,
    electorate: Electorate,
    position_attr: str = "outcome_position",
) -> OutcomeCenterComparison:
    """
    Compare one position from an election result to common electorate centers.

    Parameters
    ----------
    result
        Election result whose position should be evaluated.
    electorate
        Electorate used to compute the centers.
    position_attr
        One of the position attributes on ``ElectionResult``, typically
        ``"outcome_position"``, ``"centroid_position"``, or
        ``"median_legislator_position"``.
    """
    if not hasattr(result, position_attr):
        raise ValueError(f"ElectionResult has no position attribute '{position_attr}'.")

    position = np.asarray(getattr(result, position_attr), dtype=float)
    centers = compute_electorate_centers(electorate)
    return OutcomeCenterComparison(
        position_name=position_attr,
        position=position,
        centers=centers,
        distance_to_mean=float(np.linalg.norm(position - centers.mean)),
        distance_to_componentwise_median=float(
            np.linalg.norm(position - centers.componentwise_median)
        ),
        distance_to_geometric_median=float(
            np.linalg.norm(position - centers.geometric_median)
        ),
    )
