"""Sincere voting baseline."""
from __future__ import annotations

from electoral_sim.ballots import BallotProfile
from electoral_sim.candidates import CandidateSet
from electoral_sim.electorate import Electorate
from electoral_sim.strategies.base import StrategyModel, VotingContext


class SincereStrategy(StrategyModel):
    """Current baseline: derive ballots directly from preference distances."""

    @property
    def name(self) -> str:
        return "Sincere"

    def generate_ballots(
        self,
        electorate: Electorate,
        candidates: CandidateSet,
        approval_threshold: float | None = None,
        context: VotingContext | None = None,
    ) -> BallotProfile:
        return BallotProfile.from_preferences(
            electorate,
            candidates,
            approval_threshold=approval_threshold,
        )
