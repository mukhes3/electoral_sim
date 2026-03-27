"""Plurality-specific strategic voting models."""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from electoral_sim.ballots import BallotProfile
from electoral_sim.candidates import CandidateSet
from electoral_sim.electorate import Electorate
from electoral_sim.strategies.base import StrategyModel, VotingContext


@dataclass
class PluralityCompromiseStrategy(StrategyModel):
    """
    Strategic plurality model where voters may desert non-viable favorites.

    The strategy leaves rankings, approvals, and scores sincere for now and
    changes only the cast plurality vote. This keeps the existing ballot
    interface intact while giving the simulator a first non-sincere baseline.
    """
    compromise_rate: float = 1.0
    viability_threshold: float = 0.15
    frontrunner_count: int = 2
    rng: np.random.Generator = field(default_factory=np.random.default_rng)

    @property
    def name(self) -> str:
        return "Plurality Compromise"

    def _viable_candidates(
        self,
        sincere_ballots: BallotProfile,
        context: VotingContext | None,
    ) -> np.ndarray:
        if context is not None and context.frontrunner_indices is not None:
            return np.array(sorted(set(context.frontrunner_indices)), dtype=int)

        if context is not None and context.poll_shares is not None:
            poll_shares = np.asarray(context.poll_shares, dtype=float)
        else:
            poll_shares = (
                np.bincount(sincere_ballots.plurality, minlength=sincere_ballots.n_candidates)
                / sincere_ballots.n_voters
            )

        viable = np.flatnonzero(poll_shares >= self.viability_threshold)
        if len(viable) >= self.frontrunner_count:
            return viable
        return np.argsort(poll_shares)[-self.frontrunner_count:]

    def generate_ballots(
        self,
        electorate: Electorate,
        candidates: CandidateSet,
        approval_threshold: float | None = None,
        context: VotingContext | None = None,
    ) -> BallotProfile:
        sincere = BallotProfile.from_preferences(
            electorate,
            candidates,
            approval_threshold=approval_threshold,
        )
        viable = self._viable_candidates(sincere, context)
        if len(viable) == 0:
            return sincere

        plurality = sincere.plurality.copy()
        for voter_idx in range(sincere.n_voters):
            sincere_choice = plurality[voter_idx]
            if sincere_choice in viable:
                continue
            if self.rng.random() > self.compromise_rate:
                continue
            viable_dists = sincere.distances[voter_idx, viable]
            plurality[voter_idx] = int(viable[viable_dists.argmin()])

        return BallotProfile(
            plurality=plurality,
            rankings=sincere.rankings,
            scores=sincere.scores,
            approvals=sincere.approvals,
            distances=sincere.distances,
            approval_threshold=sincere.approval_threshold,
            n_voters=sincere.n_voters,
            n_candidates=sincere.n_candidates,
        )
