"""Spatial reporting models for practical fractional-ballot analysis."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from electoral_sim.candidates import CandidateSet
from electoral_sim.electorate import Electorate

from electoral_sim.reporting.selectors import VoterSelector


@dataclass
class ReportingContext:
    """Optional shared context for reporting models."""

    metadata: dict = field(default_factory=dict)


class PositionReportingModel(ABC):
    """Base class for models that turn true positions into reported positions."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def report_positions(
        self,
        electorate: Electorate,
        candidates: CandidateSet,
        context: ReportingContext | None = None,
    ) -> np.ndarray:
        """Return reported positions with shape ``(n_voters, n_dims)``."""


def _clip_unit_cube(points: np.ndarray) -> np.ndarray:
    return np.clip(points, 0.0, 1.0)


def _resolve_mask(
    electorate: Electorate,
    candidates: CandidateSet,
    voter_mask: np.ndarray | None = None,
    selector: VoterSelector | None = None,
) -> np.ndarray:
    if voter_mask is not None and selector is not None:
        raise ValueError("Provide either voter_mask or selector, not both.")

    if selector is not None:
        mask = np.asarray(selector(electorate, candidates), dtype=bool)
    elif voter_mask is not None:
        mask = np.asarray(voter_mask, dtype=bool)
    else:
        mask = np.ones(electorate.n_voters, dtype=bool)

    if mask.shape != (electorate.n_voters,):
        raise ValueError(
            f"Mask must have shape ({electorate.n_voters},), got {mask.shape}."
        )
    return mask


@dataclass
class HonestReporting(PositionReportingModel):
    """Truthful benchmark: the system sees the voters' real positions."""

    @property
    def name(self) -> str:
        return "Honest reporting"

    def report_positions(
        self,
        electorate: Electorate,
        candidates: CandidateSet,
        context: ReportingContext | None = None,
    ) -> np.ndarray:
        del candidates, context
        return electorate.preferences.copy()


@dataclass
class GaussianNoiseReporting(PositionReportingModel):
    """Noisy measurement of voter positions."""

    noise_std: float = 0.05
    rng: np.random.Generator | None = None

    @property
    def name(self) -> str:
        return f"Gaussian noise reporting (std={self.noise_std:.3g})"

    def report_positions(
        self,
        electorate: Electorate,
        candidates: CandidateSet,
        context: ReportingContext | None = None,
    ) -> np.ndarray:
        del candidates, context
        rng = self.rng or np.random.default_rng()
        noise = rng.normal(
            loc=0.0,
            scale=self.noise_std,
            size=electorate.preferences.shape,
        )
        return _clip_unit_cube(electorate.preferences + noise)


@dataclass
class BiasedNoiseReporting(PositionReportingModel):
    """
    Systematic measurement bias, optionally combined with random noise.

    The bias can be applied to all voters or only to a selected subset.
    """

    bias: np.ndarray
    noise_std: float = 0.0
    voter_mask: np.ndarray | None = None
    selector: VoterSelector | None = None
    rng: np.random.Generator | None = None

    @property
    def name(self) -> str:
        return "Biased noise reporting"

    def report_positions(
        self,
        electorate: Electorate,
        candidates: CandidateSet,
        context: ReportingContext | None = None,
    ) -> np.ndarray:
        del context
        mask = _resolve_mask(electorate, candidates, self.voter_mask, self.selector)
        reported = electorate.preferences.copy()
        bias = np.asarray(self.bias, dtype=float)
        if bias.shape != (electorate.n_dims,):
            raise ValueError(
                f"bias must have shape ({electorate.n_dims},), got {bias.shape}"
            )
        reported[mask] += bias
        if self.noise_std > 0:
            rng = self.rng or np.random.default_rng()
            reported[mask] += rng.normal(
                loc=0.0,
                scale=self.noise_std,
                size=(int(mask.sum()), electorate.n_dims),
            )
        return _clip_unit_cube(reported)


@dataclass
class DirectionalExaggerationReporting(PositionReportingModel):
    """
    Strategic spatial exaggeration along a chosen direction.

    Selected voters can report farther away from a rival, toward an ally,
    or both. The resulting direction is normalized before applying
    ``strength``.
    """

    strength: float = 0.15
    away_from_candidate_idx: int | None = None
    toward_candidate_idx: int | None = None
    voter_mask: np.ndarray | None = None
    selector: VoterSelector | None = None

    @property
    def name(self) -> str:
        return "Directional exaggeration reporting"

    def report_positions(
        self,
        electorate: Electorate,
        candidates: CandidateSet,
        context: ReportingContext | None = None,
    ) -> np.ndarray:
        del context
        if self.away_from_candidate_idx is None and self.toward_candidate_idx is None:
            raise ValueError("Provide away_from_candidate_idx and/or toward_candidate_idx.")

        mask = _resolve_mask(electorate, candidates, self.voter_mask, self.selector)
        reported = electorate.preferences.copy()
        selected = reported[mask]
        direction = np.zeros_like(selected)

        if self.away_from_candidate_idx is not None:
            away_pos = candidates.positions[self.away_from_candidate_idx]
            direction += selected - away_pos
        if self.toward_candidate_idx is not None:
            toward_pos = candidates.positions[self.toward_candidate_idx]
            direction += toward_pos - selected

        norms = np.linalg.norm(direction, axis=1, keepdims=True)
        safe_direction = np.divide(
            direction,
            np.where(norms > 1e-12, norms, 1.0),
        )
        reported[mask] = selected + self.strength * safe_direction
        return _clip_unit_cube(reported)


@dataclass
class CoalitionMisreporting(PositionReportingModel):
    """
    Coordinated bloc misreporting toward a common target point.

    ``strength=1`` means the selected voters report exactly the target point.
    Lower strengths interpolate between truth and the coalition target.
    """

    target_position: np.ndarray
    strength: float = 1.0
    voter_mask: np.ndarray | None = None
    selector: VoterSelector | None = None

    @property
    def name(self) -> str:
        return "Coalition misreporting"

    def report_positions(
        self,
        electorate: Electorate,
        candidates: CandidateSet,
        context: ReportingContext | None = None,
    ) -> np.ndarray:
        del context
        mask = _resolve_mask(electorate, candidates, self.voter_mask, self.selector)
        target = np.asarray(self.target_position, dtype=float)
        if target.shape != (electorate.n_dims,):
            raise ValueError(
                f"target_position must have shape ({electorate.n_dims},), got {target.shape}"
            )

        reported = electorate.preferences.copy()
        reported[mask] = (
            (1.0 - self.strength) * reported[mask]
            + self.strength * target
        )
        return _clip_unit_cube(reported)
