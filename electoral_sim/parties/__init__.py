"""Party abstractions for ideology-based party assignment."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.spatial.distance import cdist

from electoral_sim.candidates import CandidateSet
from electoral_sim.electorate import Electorate


@dataclass
class PartySet:
    """
    Political parties represented as positions in the same spatial map.

    This deliberately mirrors ``CandidateSet`` so notebook code can treat
    party anchors as first-class spatial objects without changing any of the
    existing election APIs.
    """

    positions: np.ndarray
    labels: list[str]

    def __post_init__(self):
        assert self.positions.ndim == 2
        assert len(self.labels) == self.n_parties
        assert np.all(self.positions >= 0) and np.all(self.positions <= 1)

    @property
    def n_parties(self) -> int:
        return self.positions.shape[0]

    @property
    def n_dims(self) -> int:
        return self.positions.shape[1]

    def subset(self, party_indices: Sequence[int]) -> PartySet:
        idx = np.asarray(party_indices, dtype=int)
        return PartySet(
            positions=self.positions[idx].copy(),
            labels=[self.labels[int(i)] for i in idx],
        )


def fixed_parties(positions, labels=None) -> PartySet:
    pos = np.array(positions, dtype=float)
    if labels is None:
        labels = [f"Party {i}" for i in range(len(pos))]
    return PartySet(pos, list(labels))


def nearest_party_distances(points: np.ndarray, parties: PartySet) -> np.ndarray:
    """
    Pairwise distances from arbitrary positions to party anchors.

    ``points`` can be electorate preferences or candidate positions.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2:
        raise ValueError("points must be a 2D array.")
    if points.shape[1] != parties.n_dims:
        raise ValueError(
            f"Point dimension ({points.shape[1]}) does not match party dimension ({parties.n_dims})."
        )
    return cdist(points, parties.positions, metric="euclidean")


def nearest_party_indices(points: np.ndarray, parties: PartySet) -> np.ndarray:
    """Index of the nearest party anchor for each point."""
    dists = nearest_party_distances(points, parties)
    return dists.argmin(axis=1).astype(int)


def assign_voters_to_parties(electorate: Electorate, parties: PartySet) -> np.ndarray:
    """Assign each voter to the nearest party anchor."""
    return nearest_party_indices(electorate.preferences, parties)


def assign_candidates_to_parties(candidates: CandidateSet, parties: PartySet) -> np.ndarray:
    """Assign each candidate to the nearest party anchor."""
    return nearest_party_indices(candidates.positions, parties)


def membership_masks_from_indices(indices: np.ndarray, labels: Sequence[str]) -> dict[str, np.ndarray]:
    """Convert nearest-party indices into boolean masks keyed by label."""
    indices = np.asarray(indices, dtype=int)
    return {
        str(label): indices == idx
        for idx, label in enumerate(labels)
    }
