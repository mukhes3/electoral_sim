"""Ranked-ballot strategic models."""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from electoral_sim.ballots import BallotProfile
from electoral_sim.candidates import CandidateSet
from electoral_sim.electorate import Electorate
from electoral_sim.strategies.base import StrategyModel, VotingContext
from electoral_sim.strategies.common import ViabilityMixin


@dataclass
class RankedTruncationStrategy(StrategyModel):
    """Submit only the top portion of a sincere ranking."""
    max_ranked: int = 2

    @property
    def name(self) -> str:
        return "Ranked Truncation"

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
        rankings = np.full_like(sincere.rankings, fill_value=-1)
        keep = max(1, min(self.max_ranked, sincere.n_candidates))
        rankings[:, :keep] = sincere.rankings[:, :keep]

        return BallotProfile(
            plurality=sincere.plurality,
            rankings=rankings,
            scores=sincere.scores,
            approvals=sincere.approvals,
            distances=sincere.distances,
            approval_threshold=sincere.approval_threshold,
            n_voters=sincere.n_voters,
            n_candidates=sincere.n_candidates,
            active_voter_mask=sincere.active_voter_mask,
        )


@dataclass
class RankedBuryingStrategy(ViabilityMixin, StrategyModel):
    """Demote a strong rival to the bottom of the ranking."""
    bury_rate: float = 1.0
    rng: np.random.Generator = field(default_factory=np.random.default_rng)

    @property
    def name(self) -> str:
        return "Ranked Burying"

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
        viable = self.viable_candidates(sincere, context)
        rankings = sincere.rankings.copy()

        for voter_idx in range(sincere.n_voters):
            if self.rng.random() > self.bury_rate:
                continue
            favorite = rankings[voter_idx, 0]
            viable_rivals = [cand for cand in viable if cand != favorite]
            if not viable_rivals:
                continue
            rival_dists = sincere.distances[voter_idx, viable_rivals]
            target = int(viable_rivals[int(rival_dists.argmin())])
            row = [c for c in rankings[voter_idx] if c != target]
            row.append(target)
            rankings[voter_idx] = np.array(row, dtype=rankings.dtype)

        return BallotProfile(
            plurality=sincere.plurality,
            rankings=rankings,
            scores=sincere.scores,
            approvals=sincere.approvals,
            distances=sincere.distances,
            approval_threshold=sincere.approval_threshold,
            n_voters=sincere.n_voters,
            n_candidates=sincere.n_candidates,
            active_voter_mask=sincere.active_voter_mask,
        )
