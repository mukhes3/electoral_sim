"""Turnout and abstention models."""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from electoral_sim.ballots import BallotProfile
from electoral_sim.candidates import CandidateSet
from electoral_sim.electorate import Electorate
from electoral_sim.strategies.base import StrategyModel, VotingContext
from electoral_sim.strategies.common import ViabilityMixin


@dataclass
class TurnoutStrategy(ViabilityMixin, StrategyModel):
    """
    Model abstention while leaving ballot expression sincere for participants.

    A voter turns out with `turnout_probability` by default. If
    `abstain_if_favorite_nonviable` is enabled, voters whose sincere favourite
    is not viable abstain with `abstain_probability_when_nonviable`.
    """
    turnout_probability: float = 1.0
    abstain_if_favorite_nonviable: bool = False
    abstain_probability_when_nonviable: float = 1.0
    rng: np.random.Generator = field(default_factory=np.random.default_rng)

    @property
    def name(self) -> str:
        return "Turnout"

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
        active = self.rng.random(sincere.n_voters) < self.turnout_probability

        if self.abstain_if_favorite_nonviable:
            viable = self.viable_candidates(sincere, context)
            for voter_idx, favorite in enumerate(sincere.plurality):
                if favorite in viable:
                    continue
                if self.rng.random() < self.abstain_probability_when_nonviable:
                    active[voter_idx] = False

        return BallotProfile(
            plurality=sincere.plurality,
            rankings=sincere.rankings,
            scores=sincere.scores,
            approvals=sincere.approvals,
            distances=sincere.distances,
            approval_threshold=sincere.approval_threshold,
            n_voters=sincere.n_voters,
            n_candidates=sincere.n_candidates,
            active_voter_mask=active,
        )
