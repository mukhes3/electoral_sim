"""Approval-voting strategic models."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from electoral_sim.ballots import BallotProfile
from electoral_sim.candidates import CandidateSet
from electoral_sim.electorate import Electorate
from electoral_sim.strategies.base import StrategyModel, VotingContext


@dataclass
class ApprovalThresholdStrategy(StrategyModel):
    """
    Approve candidates that clear a utility threshold.

    Utility is measured using the simulator's normalized sincere score in [0, 1].
    """
    utility_threshold: float = 0.5

    @property
    def name(self) -> str:
        return "Approval Threshold"

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
        approvals = (sincere.scores >= self.utility_threshold).astype(int)
        nearest = sincere.distances.argmin(axis=1)
        for i in range(sincere.n_voters):
            if approvals[i].sum() == 0:
                approvals[i, nearest[i]] = 1

        return BallotProfile(
            plurality=sincere.plurality,
            rankings=sincere.rankings,
            scores=sincere.scores,
            approvals=approvals,
            distances=sincere.distances,
            approval_threshold=sincere.approval_threshold,
            n_voters=sincere.n_voters,
            n_candidates=sincere.n_candidates,
            active_voter_mask=sincere.active_voter_mask,
        )
