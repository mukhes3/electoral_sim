"""Score-voting strategic models."""
from __future__ import annotations

from dataclasses import dataclass

from electoral_sim.ballots import BallotProfile
from electoral_sim.candidates import CandidateSet
from electoral_sim.electorate import Electorate
from electoral_sim.strategies.base import StrategyModel, VotingContext


@dataclass
class ScoreMaxMinStrategy(StrategyModel):
    """
    Exaggerate sincere scores into a max-min scale.

    Candidates above the threshold receive max_score; all others receive
    min_score.
    """
    utility_threshold: float = 0.5
    min_score: float = 0.0
    max_score: float = 1.0

    @property
    def name(self) -> str:
        return "Score Max-Min"

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
        scores = (
            (sincere.scores >= self.utility_threshold).astype(float)
            * (self.max_score - self.min_score)
            + self.min_score
        )
        return BallotProfile(
            plurality=sincere.plurality,
            rankings=sincere.rankings,
            scores=scores,
            approvals=sincere.approvals,
            distances=sincere.distances,
            approval_threshold=sincere.approval_threshold,
            n_voters=sincere.n_voters,
            n_candidates=sincere.n_candidates,
            active_voter_mask=sincere.active_voter_mask,
        )
