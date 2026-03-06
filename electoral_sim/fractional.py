"""
fractional.py
-------------
Fractional Ballot electoral systems.

Each voter submits a weight vector over candidates derived from their proximity
to each candidate via a Boltzmann (softmax) kernel:

    w_{ik} = exp(-d_{ik} / σ) / Σ_j exp(-d_{ij} / σ)

where d_{ik} = ||v_i - x_k|| is the Euclidean distance from voter i to
candidate k, and σ > 0 is a temperature parameter.

Special cases
-------------
  σ → 0  :  w_{ik} → 1[k = nearest candidate]  →  Plurality outcome
  σ → ∞  :  w_{ik} → 1/K for all k             →  uniform weight over all candidates

Two variants
------------
FractionalBallotDiscrete
    The population-mean weight vector is computed, and the candidate nearest to
    the resulting centroid position is declared the single winner. Behaves like
    a soft version of Plurality. Useful for primary elections or any context
    requiring a single nominee.

    outcome_position = mean_w @ X  (the centroid — same as the old FractionalBallot)
    winner_indices   = [nearest candidate to centroid]  (who "wins" the seat)
    seat_shares      = {winner_idx: 1.0}

FractionalBallotContinuous
    The population-mean weight vector IS the election outcome. No single winner
    is declared. Each candidate receives a fractional share of legislative
    voting power proportional to their mean weight across all voters.
    Suitable for policy voting in a legislature where candidates (parties)
    cast weighted votes on every bill.

    outcome_position = mean_w @ X  (convex combination of candidate positions)
    seat_shares      = {k: mean_w[k] for all k}  (sums to 1.0)

Both classes share the same _boltzmann_weights() computation and implement
the standard ElectoralSystem interface.
"""
from __future__ import annotations

import numpy as np

from electoral_sim.ballots import BallotProfile
from electoral_sim.candidates import CandidateSet
from electoral_sim.systems import ElectoralSystem
from electoral_sim.types import ElectionResult


# ── Shared internals ──────────────────────────────────────────────────────────

def _boltzmann_weights(distances: np.ndarray, sigma: float) -> np.ndarray:
    """
    Compute Boltzmann (softmax) weight matrix from voter-candidate distances.

    Parameters
    ----------
    distances : np.ndarray, shape (n_voters, n_candidates)
        Euclidean distances from each voter to each candidate.
    sigma : float
        Temperature parameter. Lower → sharper concentration on nearest candidate.

    Returns
    -------
    weights : np.ndarray, shape (n_voters, n_candidates)
        Each row sums to 1. weights[i, k] is voter i's fractional support for
        candidate k.
    """
    logits = -distances / sigma
    logits -= logits.max(axis=1, keepdims=True)   # numerical stability
    weights = np.exp(logits)
    weights /= weights.sum(axis=1, keepdims=True)
    return weights


def _mean_weights(distances: np.ndarray, sigma: float) -> np.ndarray:
    """
    Return population-mean weight vector: shape (n_candidates,), sums to 1.
    This is the core electoral outcome of the Fractional Ballot.
    """
    return _boltzmann_weights(distances, sigma).mean(axis=0)


# ── Discrete variant ──────────────────────────────────────────────────────────

class FractionalBallotDiscrete(ElectoralSystem):
    """
    Fractional Ballot — Discrete (single-winner) variant.

    Computes the population-mean weight vector, finds the convex-combination
    centroid of candidate positions, and declares the candidate nearest to
    that centroid as the sole winner.

    Use this variant when a single nominee is required (e.g. primary elections).

    Parameters
    ----------
    sigma : float
        Temperature parameter. Lower σ → approaches Plurality.
        Higher σ → approaches uniform-centroid of all candidates.
    """

    def __init__(self, sigma: float = 0.3):
        assert sigma > 0, "sigma must be positive"
        self.sigma = sigma

    @property
    def name(self) -> str:
        return f"Fractional Ballot Discrete (σ={self.sigma:.2g})"

    @property
    def parameters(self) -> dict:
        return {"sigma": self.sigma}

    def run(self, ballots: BallotProfile, candidates: CandidateSet) -> ElectionResult:
        X      = candidates.positions                        # (n_candidates, n_dims)
        mean_w = _mean_weights(ballots.distances, self.sigma) # (n_candidates,)

        # Centroid position in preference space
        centroid = mean_w @ X                               # (n_dims,)

        # Winner = candidate nearest the centroid
        dist_to_centroid = np.linalg.norm(X - centroid, axis=1)
        winner_idx       = int(dist_to_centroid.argmin())

        # Use centroid as outcome_position (not the winner's position).
        # The centroid is what actually tracks the geometric median well;
        # snapping to the winner's grid position would discard that property.
        return ElectionResult(
            outcome_position=centroid,
            centroid_position=centroid,
            median_legislator_position=candidates.positions[winner_idx].copy(),
            winner_indices=[winner_idx],
            seat_shares={winner_idx: 1.0},
            elimination_order=[],
            system_name=self.name,
            is_pr=False,
            metadata={
                "sigma":                   self.sigma,
                "mean_weights":            mean_w,
                "nearest_candidate":       candidates.labels[winner_idx],
                "dist_centroid_to_winner": float(dist_to_centroid.min()),
            },
        )

    def weight_matrix(self, ballots: BallotProfile, candidates: CandidateSet) -> np.ndarray:
        """Return full (n_voters, n_candidates) weight matrix for analysis."""
        return _boltzmann_weights(ballots.distances, self.sigma)


# ── Continuous variant ────────────────────────────────────────────────────────

class FractionalBallotContinuous(ElectoralSystem):
    """
    Fractional Ballot — Continuous (weighted legislature) variant.

    The election outcome is the population-mean weight vector itself.
    Each candidate receives a fractional share of legislative voting power
    equal to their mean weight across all voters. No single winner is declared.

    This is the intended system for policy voting: on each bill, candidate k
    casts a vote with weight mean_w[k], so the effective policy outcome is a
    convex combination of all candidates' positions.

    seat_shares  →  {candidate_index: mean_w[k]}  (sums to 1.0)
    outcome_position  →  mean_w @ X  (convex combination in preference space)

    Parameters
    ----------
    sigma : float
        Temperature parameter. Lower σ → weight concentrates on nearest
        candidate (approaching Plurality seat allocation). Higher σ → weight
        spreads evenly (approaching equal shares for all candidates).
    """

    def __init__(self, sigma: float = 0.3):
        assert sigma > 0, "sigma must be positive"
        self.sigma = sigma

    @property
    def name(self) -> str:
        return f"Fractional Ballot Continuous (σ={self.sigma:.2g})"

    @property
    def parameters(self) -> dict:
        return {"sigma": self.sigma}

    def run(self, ballots: BallotProfile, candidates: CandidateSet) -> ElectionResult:
        X      = candidates.positions                         # (n_candidates, n_dims)
        mean_w = _mean_weights(ballots.distances, self.sigma)  # (n_candidates,)

        # Outcome position: convex combination of candidate positions
        outcome = mean_w @ X                                  # (n_dims,)

        # All candidates with non-negligible weight are "winners"
        seat_shares  = {k: float(mean_w[k]) for k in range(len(mean_w)) if mean_w[k] > 1e-6}
        winner_indices = sorted(seat_shares, key=seat_shares.get, reverse=True)

        return ElectionResult(
            outcome_position=outcome,
            centroid_position=outcome,           # centroid IS the outcome here
            median_legislator_position=outcome,  # not meaningful; set to outcome
            winner_indices=winner_indices,
            seat_shares=seat_shares,
            elimination_order=[],
            system_name=self.name,
            is_pr=True,                          # treated as PR: no single winner
            metadata={
                "sigma":        self.sigma,
                "mean_weights": mean_w,
                "top_candidate": candidates.labels[winner_indices[0]],
                "top_weight":    float(mean_w[winner_indices[0]]),
                "entropy":       float(-np.sum(mean_w * np.log(mean_w + 1e-12))),
            },
        )

    def weight_matrix(self, ballots: BallotProfile, candidates: CandidateSet) -> np.ndarray:
        """Return full (n_voters, n_candidates) weight matrix for analysis."""
        return _boltzmann_weights(ballots.distances, self.sigma)

    def policy_outcome(self, ballots: BallotProfile, candidates: CandidateSet) -> np.ndarray:
        """
        Convenience method: return the effective policy position directly.
        Equivalent to result.outcome_position but without constructing ElectionResult.
        """
        mean_w = _mean_weights(ballots.distances, self.sigma)
        return mean_w @ candidates.positions


# ── Convenience constructors ──────────────────────────────────────────────────

def fractional_ballot_systems(
    sigmas: list[float] | None = None,
    variant: str = "both",
) -> list[ElectoralSystem]:
    """
    Return a list of Fractional Ballot instances across standard sigma values.

    Parameters
    ----------
    sigmas : list[float], optional
        Defaults to [0.1, 0.3, 1.0].
    variant : str
        "discrete"   → FractionalBallotDiscrete only
        "continuous" → FractionalBallotContinuous only
        "both"       → both variants for each sigma (default)
    """
    sigmas = sigmas or [0.1, 0.3, 1.0]
    systems = []
    for s in sigmas:
        if variant in ("discrete", "both"):
            systems.append(FractionalBallotDiscrete(sigma=s))
        if variant in ("continuous", "both"):
            systems.append(FractionalBallotContinuous(sigma=s))
    return systems


# ── Sigma sweep utilities ─────────────────────────────────────────────────────

def sigma_sweep(
    sigmas: list[float],
    ballots: BallotProfile,
    candidates: CandidateSet,
    variant: str = "continuous",
) -> list[tuple[float, np.ndarray]]:
    """
    Run fractional ballot across a range of σ values.

    Parameters
    ----------
    variant : str
        "discrete"   → outcome is the nearest-candidate position
        "continuous" → outcome is the mean_w @ X centroid (default)

    Returns
    -------
    list of (sigma, outcome_position) tuples.
    """
    cls = FractionalBallotContinuous if variant == "continuous" else FractionalBallotDiscrete
    return [
        (s, cls(sigma=s).run(ballots, candidates).outcome_position.copy())
        for s in sigmas
    ]


def weight_entropy_sweep(
    sigmas: list[float],
    ballots: BallotProfile,
    candidates: CandidateSet,
) -> list[tuple[float, float]]:
    """
    Compute the entropy of the mean weight vector across a range of σ.
    High entropy → power spread evenly. Low entropy → power concentrated.

    Returns list of (sigma, entropy) tuples.
    Useful for choosing σ based on desired concentration of power.
    """
    results = []
    for s in sigmas:
        mean_w  = _mean_weights(ballots.distances, s)
        entropy = float(-np.sum(mean_w * np.log(mean_w + 1e-12)))
        results.append((s, entropy))
    return results
