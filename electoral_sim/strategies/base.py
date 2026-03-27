"""Strategy abstractions for converting true preferences into cast ballots."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import numpy as np

from electoral_sim.ballots import BallotProfile
from electoral_sim.candidates import CandidateSet
from electoral_sim.electorate import Electorate


@dataclass
class VotingContext:
    """
    Optional strategic context shared across voters.

    Parameters
    ----------
    poll_shares : np.ndarray, optional
        Estimated vote shares for each candidate. Used by strategic models
        that reason about viability or frontrunners.
    frontrunner_indices : list[int], optional
        Explicit list of candidates voters treat as viable contenders.
    round_number : int
        Useful for multi-round election modeling.
    metadata : dict
        Free-form container for strategy-specific context.
    """
    poll_shares: np.ndarray | None = None
    frontrunner_indices: list[int] | None = None
    round_number: int = 1
    metadata: dict = field(default_factory=dict)


class StrategyModel(ABC):
    """Base class for ballot-generation strategies."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def generate_ballots(
        self,
        electorate: Electorate,
        candidates: CandidateSet,
        approval_threshold: float | None = None,
        context: VotingContext | None = None,
    ) -> BallotProfile:
        """Return a BallotProfile derived under this strategy."""
