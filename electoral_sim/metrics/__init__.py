"""
Evaluation metrics for election outcomes.

Primary metric: Euclidean distance from election outcome to the
geometric median of voter preferences.

For PR systems, distances are reported for both the median legislator
position (outcome_position, the primary metric) and the seat-share
centroid (centroid_position, reported for reference).

Secondary metrics: distance to mean, majority satisfaction,
worst-case (max voter) distance, mean voter distance, Gini of distances.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

from electoral_sim.electorate import Electorate
from electoral_sim.candidates import CandidateSet
from electoral_sim.types import ElectionResult


@dataclass
class ElectionMetrics:
    """All metrics for a single election outcome."""
    system_name: str
    is_pr: bool

    # ── Primary metric: uses outcome_position (median legislator for PR) ──────
    distance_to_median: float        # distance from outcome to geometric median
    distance_to_mean: float          # distance from outcome to mean preference
    majority_satisfaction: float     # fraction of voters closer to outcome than any candidate
    worst_case_distance: float       # max voter distance to outcome
    mean_voter_distance: float       # mean voter distance to outcome
    gini_distance: float             # Gini coefficient of voter distances

    # ── PR-only: centroid distances (for comparison / paper discussion) ───────
    centroid_distance_to_median: float = field(default=float("nan"))
    centroid_distance_to_mean: float   = field(default=float("nan"))

    def __repr__(self) -> str:
        base = (
            f"ElectionMetrics({self.system_name}: "
            f"d_median={self.distance_to_median:.4f}, "
            f"d_mean={self.distance_to_mean:.4f}, "
            f"majority_sat={self.majority_satisfaction:.3f}"
        )
        if self.is_pr:
            base += (
                f", centroid_d_median={self.centroid_distance_to_median:.4f}"
                f", centroid_d_mean={self.centroid_distance_to_mean:.4f}"
            )
        return base + ")"


def compute_metrics(
    result: ElectionResult,
    electorate: Electorate,
    candidates: CandidateSet,
) -> ElectionMetrics:
    """
    Compute all evaluation metrics for an election result.

    For PR systems:
      - outcome_position = median legislator → used for all primary metrics
      - centroid_position → reported separately as centroid_distance_to_median
        and centroid_distance_to_mean for reference

    Parameters
    ----------
    result : ElectionResult
    electorate : Electorate
    candidates : CandidateSet
    """
    outcome = result.outcome_position
    geo_median = electorate.geometric_median()
    mean_pref = electorate.mean()

    # Primary distances (median legislator for PR, winner position for WTA)
    d_median = float(np.linalg.norm(outcome - geo_median))
    d_mean   = float(np.linalg.norm(outcome - mean_pref))

    # Per-voter distances to the outcome point
    voter_dists = np.linalg.norm(electorate.preferences - outcome, axis=1)
    mean_voter_dist = float(voter_dists.mean())
    worst_case      = float(voter_dists.max())

    # Majority satisfaction: fraction of voters closer to outcome than any candidate
    if candidates.n_candidates > 1:
        cand_dists = np.linalg.norm(
            electorate.preferences[:, None, :] - candidates.positions[None, :, :],
            axis=2,
        )
        nearest_candidate_dist = cand_dists.min(axis=1)
        majority_sat = float((voter_dists <= nearest_candidate_dist + 1e-9).mean())
    else:
        majority_sat = 1.0

    gini = _gini(voter_dists)

    # PR-only: centroid distances
    centroid_d_median = float("nan")
    centroid_d_mean   = float("nan")
    if result.is_pr:
        centroid_d_median = float(np.linalg.norm(result.centroid_position - geo_median))
        centroid_d_mean   = float(np.linalg.norm(result.centroid_position - mean_pref))

    return ElectionMetrics(
        system_name=result.system_name,
        is_pr=result.is_pr,
        distance_to_median=d_median,
        distance_to_mean=d_mean,
        majority_satisfaction=majority_sat,
        worst_case_distance=worst_case,
        mean_voter_distance=mean_voter_dist,
        gini_distance=gini,
        centroid_distance_to_median=centroid_d_median,
        centroid_distance_to_mean=centroid_d_mean,
    )


def _gini(x: np.ndarray) -> float:
    """Gini coefficient for an array of non-negative values."""
    x = np.sort(x)
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    cumx = np.cumsum(x)
    return float((2 * cumx.sum() - (n + 1) * x.sum()) / (n * x.sum()))


def run_simulation(
    electorate: Electorate,
    candidates: CandidateSet,
    systems: list,
    approval_threshold: float | None = None,
    strategy=None,
    context=None,
) -> list[ElectionMetrics]:
    """
    Run all electoral systems on a single (electorate, candidates) pair
    and return metrics for each.

    Parameters
    ----------
    strategy : optional StrategyModel
        If None, uses the existing sincere-voting baseline.
    context : optional VotingContext
        Extra strategic context such as polling information or frontrunners.
    """
    from electoral_sim.ballots import BallotProfile

    if strategy is None:
        ballots = BallotProfile.from_preferences(electorate, candidates, approval_threshold)
    else:
        ballots = BallotProfile.from_strategy(
            electorate,
            candidates,
            strategy=strategy,
            approval_threshold=approval_threshold,
            context=context,
        )
    results = []
    for system in systems:
        result = system.run(ballots, candidates)
        metrics = compute_metrics(result, electorate, candidates)
        results.append(metrics)
    return results


def run_monte_carlo(
    scenario_factory,
    systems: list,
    n_trials: int = 500,
    rng: np.random.Generator | None = None,
    approval_threshold: float | None = None,
    strategy=None,
    context_factory=None,
) -> dict[str, list[ElectionMetrics]]:
    """
    Run multiple simulation trials, aggregating metrics per system.

    Parameters
    ----------
    scenario_factory : callable
        Returns (Electorate, CandidateSet) — called fresh each trial.
        Should accept an rng argument.
    systems : list[ElectoralSystem]
    n_trials : int
    rng : np.random.Generator

    strategy : optional StrategyModel
        Strategy used to generate ballots in each trial. Defaults to sincere.
    context_factory : callable, optional
        If provided, called as context_factory(electorate, candidates, rng=rng)
        once per trial to produce a VotingContext.

    Returns
    -------
    dict mapping system_name -> list of ElectionMetrics across trials
    """
    rng = rng or np.random.default_rng()
    results_by_system: dict[str, list[ElectionMetrics]] = {
        s.name: [] for s in systems
    }

    for _ in range(n_trials):
        electorate, candidates = scenario_factory(rng=rng)
        context = None
        if context_factory is not None:
            context = context_factory(electorate, candidates, rng=rng)
        trial_metrics = run_simulation(
            electorate,
            candidates,
            systems,
            approval_threshold,
            strategy=strategy,
            context=context,
        )
        for m in trial_metrics:
            results_by_system[m.system_name].append(m)

    return results_by_system


def summarize_monte_carlo(
    results_by_system: dict[str, list[ElectionMetrics]],
) -> dict[str, dict[str, float]]:
    """
    Summarize Monte Carlo results: mean and std of each metric per system.
    Centroid fields are included for PR systems; NaN for WTA systems.
    """
    summary = {}
    fields = [
        "distance_to_median", "distance_to_mean", "majority_satisfaction",
        "worst_case_distance", "mean_voter_distance", "gini_distance",
        "centroid_distance_to_median", "centroid_distance_to_mean",
    ]
    for system_name, metrics_list in results_by_system.items():
        summary[system_name] = {}
        for f in fields:
            vals = np.array([getattr(m, f) for m in metrics_list], dtype=float)
            valid = vals[~np.isnan(vals)]
            summary[system_name][f"{f}_mean"] = float(valid.mean()) if len(valid) else float("nan")
            summary[system_name][f"{f}_std"]  = float(valid.std())  if len(valid) else float("nan")
    return summary


from electoral_sim.metrics.robustness import (  # noqa: E402
    FractionalRobustnessMetrics,
    compute_fractional_robustness_metrics,
)
from electoral_sim.metrics.groups import (  # noqa: E402
    GroupMetricsSummary,
    GroupOutcomeMetrics,
    compute_group_metrics,
)
