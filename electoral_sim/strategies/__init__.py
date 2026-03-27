"""Voting strategy models."""

from electoral_sim.strategies.approval import ApprovalThresholdStrategy
from electoral_sim.strategies.base import StrategyModel, VotingContext
from electoral_sim.strategies.plurality import PluralityCompromiseStrategy
from electoral_sim.strategies.ranked import RankedBuryingStrategy, RankedTruncationStrategy
from electoral_sim.strategies.score import ScoreMaxMinStrategy
from electoral_sim.strategies.sincere import SincereStrategy
from electoral_sim.strategies.turnout import TurnoutStrategy

__all__ = [
    "ApprovalThresholdStrategy",
    "PluralityCompromiseStrategy",
    "RankedBuryingStrategy",
    "RankedTruncationStrategy",
    "ScoreMaxMinStrategy",
    "SincereStrategy",
    "StrategyModel",
    "TurnoutStrategy",
    "VotingContext",
]
