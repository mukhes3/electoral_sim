"""Shared helpers for strategic ballot-generation models."""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from electoral_sim.ballots import BallotProfile
from electoral_sim.strategies.base import VotingContext


@dataclass
class ViabilityMixin:
    """Helper for strategies that reason about frontrunners or viable sets."""
    viability_threshold: float = 0.15
    frontrunner_count: int = 2

    def viable_candidates(
        self,
        sincere_ballots: BallotProfile,
        context: VotingContext | None,
    ) -> np.ndarray:
        if context is not None and context.frontrunner_indices is not None:
            return np.array(sorted(set(context.frontrunner_indices)), dtype=int)

        if context is not None and context.poll_shares is not None:
            poll_shares = np.asarray(context.poll_shares, dtype=float)
        else:
            denom = max(sincere_ballots.n_active_voters, 1)
            poll_shares = sincere_ballots.plurality_counts() / denom

        viable = np.flatnonzero(poll_shares >= self.viability_threshold)
        if len(viable) >= self.frontrunner_count:
            return viable
        return np.argsort(poll_shares)[-self.frontrunner_count:]
