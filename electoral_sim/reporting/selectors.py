"""Helpers for selecting voter subsets in spatial reporting models."""
from __future__ import annotations

from typing import Callable

import numpy as np

from electoral_sim.candidates import CandidateSet
from electoral_sim.electorate import Electorate


VoterSelector = Callable[[Electorate, CandidateSet], np.ndarray]


def _validate_mask(mask: np.ndarray, electorate: Electorate) -> np.ndarray:
    arr = np.asarray(mask, dtype=bool)
    if arr.shape != (electorate.n_voters,):
        raise ValueError(
            f"voter mask must have shape ({electorate.n_voters},), got {arr.shape}"
        )
    return arr


def all_voters() -> VoterSelector:
    """Return a selector that targets every voter."""

    def selector(electorate: Electorate, candidates: CandidateSet) -> np.ndarray:
        del candidates
        return np.ones(electorate.n_voters, dtype=bool)

    return selector


def custom_mask(mask: np.ndarray) -> VoterSelector:
    """Return a selector backed by a fixed boolean mask."""

    def selector(electorate: Electorate, candidates: CandidateSet) -> np.ndarray:
        del candidates
        return _validate_mask(mask, electorate)

    return selector


def axis_threshold(dim: int, threshold: float, side: str = "ge") -> VoterSelector:
    """Select voters based on one spatial axis."""
    if side not in {"ge", "gt", "le", "lt"}:
        raise ValueError("side must be one of {'ge', 'gt', 'le', 'lt'}.")

    def selector(electorate: Electorate, candidates: CandidateSet) -> np.ndarray:
        del candidates
        coords = electorate.preferences[:, dim]
        if side == "ge":
            return coords >= threshold
        if side == "gt":
            return coords > threshold
        if side == "le":
            return coords <= threshold
        return coords < threshold

    return selector


def within_radius(center: np.ndarray, radius: float) -> VoterSelector:
    """Select voters lying within a radius of the given point."""
    center = np.asarray(center, dtype=float)

    def selector(electorate: Electorate, candidates: CandidateSet) -> np.ndarray:
        del candidates
        dists = np.linalg.norm(electorate.preferences - center, axis=1)
        return dists <= radius

    return selector


def nearest_to_candidate(
    candidate_idx: int,
    top_k: int | None = None,
    radius: float | None = None,
) -> VoterSelector:
    """
    Select voters closest to a candidate.

    If ``top_k`` is provided, returns exactly the ``top_k`` nearest voters.
    If ``radius`` is provided, returns voters within that radius.
    If neither is provided, returns sincere nearest-candidate supporters.
    """

    def selector(electorate: Electorate, candidates: CandidateSet) -> np.ndarray:
        positions = electorate.preferences
        target = candidates.positions[candidate_idx]
        dists = np.linalg.norm(positions - target, axis=1)

        if top_k is not None:
            order = np.argsort(dists, kind="stable")
            mask = np.zeros(electorate.n_voters, dtype=bool)
            mask[order[:top_k]] = True
            return mask

        if radius is not None:
            return dists <= radius

        all_dists = np.linalg.norm(
            positions[:, None, :] - candidates.positions[None, :, :],
            axis=2,
        )
        nearest = all_dists.argmin(axis=1)
        return nearest == candidate_idx

    return selector


def candidate_supporters(candidate_idx: int) -> VoterSelector:
    """Alias for sincere nearest-candidate supporters."""
    return nearest_to_candidate(candidate_idx)
