"""Helpers for running fractional systems under reported-position assumptions."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from electoral_sim.ballots import BallotProfile
from electoral_sim.candidates import CandidateSet
from electoral_sim.electorate import Electorate
from electoral_sim.metrics import ElectionMetrics, compute_metrics
from electoral_sim.metrics.robustness import (
    FractionalRobustnessMetrics,
    compute_fractional_robustness_metrics,
)
from electoral_sim.reporting import PositionReportingModel, ReportingContext
from electoral_sim.types import ElectionResult


@dataclass
class FractionalReportingRun:
    """Bundle truthful and reporting-distorted fractional outcomes."""

    truthful_result: ElectionResult
    reported_result: ElectionResult
    truthful_metrics: ElectionMetrics
    reported_metrics: ElectionMetrics
    robustness: FractionalRobustnessMetrics
    reported_preferences: np.ndarray
    reporting_model_name: str


def run_fractional_reporting_simulation(
    electorate: Electorate,
    candidates: CandidateSet,
    system,
    reporting_model: PositionReportingModel,
    approval_threshold: float | None = None,
    context: ReportingContext | None = None,
    target_mask: np.ndarray | None = None,
) -> FractionalReportingRun:
    """
    Run a fractional system under truthful and reported-position assumptions.

    The system is evaluated against the true electorate in both cases so that
    one can compare how much distortion the reporting model introduces.
    """
    ballots = BallotProfile.from_preferences(
        electorate,
        candidates,
        approval_threshold=approval_threshold,
    )
    reported_preferences = reporting_model.report_positions(
        electorate,
        candidates,
        context=context,
    )

    truthful_result = system.run(ballots, candidates)
    reported_result = system.run(
        ballots,
        candidates,
        reported_preferences=reported_preferences,
    )

    truthful_metrics = compute_metrics(truthful_result, electorate, candidates)
    reported_metrics = compute_metrics(reported_result, electorate, candidates)
    robustness = compute_fractional_robustness_metrics(
        truthful_result,
        reported_result,
        electorate,
        candidates,
        target_mask=target_mask,
    )

    return FractionalReportingRun(
        truthful_result=truthful_result,
        reported_result=reported_result,
        truthful_metrics=truthful_metrics,
        reported_metrics=reported_metrics,
        robustness=robustness,
        reported_preferences=np.asarray(reported_preferences, dtype=float).copy(),
        reporting_model_name=reporting_model.name,
    )
