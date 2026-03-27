"""
Electoral system implementations.

All systems implement the ElectoralSystem interface:
    run(ballot_profile, candidates) -> ElectionResult

Systems implemented:
    - Plurality (FPTP)
    - Two-Round Runoff
    - Instant Runoff Voting (IRV / RCV)
    - Borda Count
    - Approval Voting
    - Score / Range Voting
    - Condorcet (Schulze method)
    - Party-List PR (D'Hondt)
    - Mixed Member Proportional (MMP)
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import numpy as np

from electoral_sim.ballots import BallotProfile
from electoral_sim.candidates import CandidateSet
from electoral_sim.types import ElectionResult


class ElectoralSystem(ABC):
    """Base class for all electoral systems."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def run(self, ballots: BallotProfile, candidates: CandidateSet) -> ElectionResult:
        ...

    # ------------------------------------------------------------------
    # Parametric interface (for RL parametric optimization)
    # ------------------------------------------------------------------
    @property
    def parameters(self) -> dict:
        """Return current tunable parameters as a dict."""
        return {}

    def set_parameters(self, params: dict) -> None:
        """Set tunable parameters from a dict."""
        for k, v in params.items():
            if hasattr(self, k):
                setattr(self, k, v)

    def _make_result(
        self,
        winner_idx: int,
        candidates: CandidateSet,
        seat_shares: dict[int, float] | None = None,
        elimination_order: list[int] | None = None,
        metadata: dict | None = None,
    ) -> ElectionResult:
        """Helper: build a winner-take-all ElectionResult.
        All three position fields are identical for single-winner systems."""
        if seat_shares is None:
            seat_shares = {winner_idx: 1.0}
        pos = candidates.positions[winner_idx].copy()
        return ElectionResult(
            outcome_position=pos,
            centroid_position=pos,
            median_legislator_position=pos,
            winner_indices=[winner_idx],
            seat_shares=seat_shares,
            elimination_order=elimination_order or [],
            system_name=self.name,
            is_pr=False,
            metadata=metadata or {},
        )

    def _make_pr_result(
        self,
        seat_shares: dict[int, float],
        candidates: CandidateSet,
        metadata: dict | None = None,
        outcome_rule: str = "axis_median",
    ) -> ElectionResult:
        """
        Build a PR ElectionResult with two distinct outcome representations:

        centroid_position
            Seat-share-weighted mean of elected party positions. Approximates
            the voter distribution center by construction; included for
            reference but biased as a performance metric.

        median_legislator_position
            Position of the legislator at the 50th percentile of the
            cumulative seat distribution, sorted along the first dimension
            (economic axis). Consistent with the pivot/median voter theorem
            applied to the legislature. Preserved for backward compatibility.
        """
        n_dims = candidates.n_dims
        metadata = metadata or {}

        # ── Centroid: weighted mean ───────────────────────────────────────────
        centroid = np.zeros(n_dims)
        for idx, share in seat_shares.items():
            centroid += share * candidates.positions[idx]

        # ── Median legislator: pivot along dimension 0 ────────────────────────
        # Sort elected candidates by their position on the first (economic) axis
        elected = sorted(seat_shares.items(), key=lambda kv: candidates.positions[kv[0]][0])
        cumulative = 0.0
        median_leg_pos = centroid.copy()  # fallback
        for idx, share in elected:
            cumulative += share
            if cumulative >= 0.5:
                median_leg_pos = candidates.positions[idx].copy()
                break

        outcome_pos = self._resolve_pr_outcome_position(
            seat_shares,
            candidates,
            centroid,
            median_leg_pos,
            outcome_rule,
        )

        winner_indices = sorted(seat_shares, key=seat_shares.get, reverse=True)
        return ElectionResult(
            outcome_position=outcome_pos,
            centroid_position=centroid,
            median_legislator_position=median_leg_pos,
            winner_indices=winner_indices,
            seat_shares=seat_shares,
            elimination_order=[],
            system_name=self.name,
            is_pr=True,
            metadata={**metadata, "outcome_rule": outcome_rule},
        )

    def _resolve_pr_outcome_position(
        self,
        seat_shares: dict[int, float],
        candidates: CandidateSet,
        centroid: np.ndarray,
        axis_median: np.ndarray,
        outcome_rule: str,
    ) -> np.ndarray:
        """Select a PR outcome position from the elected seat distribution."""
        if outcome_rule == "axis_median":
            return axis_median.copy()
        if outcome_rule == "centroid":
            return centroid.copy()

        elected_indices = list(seat_shares)
        elected_positions = np.array([candidates.positions[idx] for idx in elected_indices], dtype=float)
        weights = np.array([seat_shares[idx] for idx in elected_indices], dtype=float)

        if outcome_rule == "legislative_geometric_median":
            return self._weighted_geometric_median(elected_positions, weights)
        if outcome_rule == "legislative_medoid":
            return self._weighted_medoid(elected_positions, weights)

        valid_rules = [
            "axis_median",
            "centroid",
            "legislative_geometric_median",
            "legislative_medoid",
        ]
        raise ValueError(
            f"Unknown PR outcome_rule: {outcome_rule}. "
            f"Expected one of {valid_rules}."
        )

    @staticmethod
    def _weighted_geometric_median(
        points: np.ndarray,
        weights: np.ndarray,
        tol: float = 1e-6,
        max_iter: int = 300,
    ) -> np.ndarray:
        """Weighted geometric median via Weiszfeld's algorithm."""
        if len(points) == 1:
            return points[0].copy()

        weights = np.asarray(weights, dtype=float)
        weights = weights / weights.sum()
        x = (weights[:, None] * points).sum(axis=0)

        for _ in range(max_iter):
            dists = np.linalg.norm(points - x, axis=1)
            zero_mask = dists <= 1e-10
            if zero_mask.any():
                return points[zero_mask.argmax()].copy()

            inv = weights / dists
            x_new = (inv[:, None] * points).sum(axis=0) / inv.sum()
            if np.linalg.norm(x_new - x) < tol:
                return x_new
            x = x_new

        return x

    @staticmethod
    def _weighted_medoid(points: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Weighted medoid among elected positions."""
        if len(points) == 1:
            return points[0].copy()

        pairwise = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
        costs = pairwise @ weights
        return points[int(costs.argmin())].copy()


# ---------------------------------------------------------------------------
# Winner-take-all systems
# ---------------------------------------------------------------------------

class Plurality(ElectoralSystem):
    """First-Past-The-Post: candidate with most first-preference votes wins."""

    @property
    def name(self) -> str:
        return "Plurality (FPTP)"

    def run(self, ballots: BallotProfile, candidates: CandidateSet) -> ElectionResult:
        vote_counts = ballots.plurality_counts()
        n_voters = max(ballots.n_active_voters, 1)
        winner = int(vote_counts.argmax())
        return self._make_result(
            winner, candidates,
            metadata={"vote_counts": vote_counts, "vote_shares": vote_counts / n_voters}
        )


class TwoRoundRunoff(ElectoralSystem):
    """
    Two-round runoff: top-2 first-round candidates face off in round 2.
    If any candidate exceeds the threshold in round 1, they win outright.
    """

    def __init__(self, first_round_threshold: float = 0.5):
        self.first_round_threshold = first_round_threshold

    @property
    def name(self) -> str:
        return "Two-Round Runoff"

    @property
    def parameters(self) -> dict:
        return {"first_round_threshold": self.first_round_threshold}

    def run(self, ballots: BallotProfile, candidates: CandidateSet) -> ElectionResult:
        n_voters = max(ballots.n_active_voters, 1)
        vote_counts = ballots.plurality_counts()
        vote_shares = vote_counts / n_voters

        # Check for first-round majority
        if vote_shares.max() >= self.first_round_threshold:
            winner = int(vote_shares.argmax())
            return self._make_result(winner, candidates,
                                     metadata={"round": 1, "vote_shares_r1": vote_shares})

        # Round 2: top 2 advance
        top2 = vote_shares.argsort()[-2:]
        # Re-vote: each voter votes for whichever of top2 they ranked higher
        top2_set = set(top2)
        r2_votes = np.zeros(candidates.n_candidates)
        for voter_ranking in ballots.active_rankings():
            for c in voter_ranking:
                if c < 0:
                    continue
                if c in top2_set:
                    r2_votes[c] += 1
                    break

        winner = int(r2_votes.argmax())
        return self._make_result(
            winner, candidates,
            metadata={"round": 2, "vote_shares_r1": vote_shares,
                      "r2_votes": r2_votes / n_voters, "finalists": list(top2)}
        )


class InstantRunoff(ElectoralSystem):
    """
    Instant Runoff Voting (IRV / Ranked Choice Voting).
    Eliminate lowest vote-getter iteratively; transfer votes to next preference.
    Winner is first candidate to exceed majority_threshold.
    """

    def __init__(self, majority_threshold: float = 0.5):
        self.majority_threshold = majority_threshold

    @property
    def name(self) -> str:
        return "Instant Runoff (IRV)"

    @property
    def parameters(self) -> dict:
        return {"majority_threshold": self.majority_threshold}

    def run(self, ballots: BallotProfile, candidates: CandidateSet) -> ElectionResult:
        n_c = candidates.n_candidates
        active = set(range(n_c))
        elimination_order = []

        # Working copy of rankings, filtered to active candidates each round
        rankings = ballots.active_rankings().copy()

        for _ in range(n_c - 1):
            # Count first active-candidate votes for each voter
            vote_counts = np.zeros(n_c)
            for voter_ranking in rankings:
                for c in voter_ranking:
                    if c < 0:
                        continue
                    if c in active:
                        vote_counts[c] += 1
                        break

            continuing_ballots = max(vote_counts.sum(), 1.0)
            vote_shares = vote_counts / continuing_ballots

            # Check for majority winner
            for c in active:
                if vote_shares[c] >= self.majority_threshold:
                    return self._make_result(
                        c, candidates,
                        elimination_order=elimination_order,
                        metadata={"vote_shares_final": vote_shares}
                    )

            # Eliminate lowest vote-getter among active candidates
            active_votes = {c: vote_counts[c] for c in active}
            eliminated = min(active_votes, key=active_votes.get)
            active.remove(eliminated)
            elimination_order.append(eliminated)

        # Last remaining candidate wins
        winner = next(iter(active))
        return self._make_result(
            winner, candidates,
            elimination_order=elimination_order,
        )


class BordaCount(ElectoralSystem):
    """
    Borda Count: assign points by rank position; candidate with most points wins.
    Standard: n-1 points for rank 1, down to 0 for last.
    """

    @property
    def name(self) -> str:
        return "Borda Count"

    def run(self, ballots: BallotProfile, candidates: CandidateSet) -> ElectionResult:
        scores = ballots.borda_scores()
        winner = int(scores.argmax())
        total_points = scores.sum()
        borda_shares = scores / total_points if total_points > 0 else np.zeros_like(scores)
        return self._make_result(
            winner, candidates,
            metadata={"borda_scores": scores, "borda_shares": borda_shares}
        )


class ApprovalVoting(ElectoralSystem):
    """
    Approval Voting: each voter approves any number of candidates;
    candidate with most approvals wins.
    """

    @property
    def name(self) -> str:
        return "Approval Voting"

    def run(self, ballots: BallotProfile, candidates: CandidateSet) -> ElectionResult:
        n_voters = max(ballots.n_active_voters, 1)
        approval_counts = ballots.active_approvals().sum(axis=0)
        winner = int(approval_counts.argmax())
        return self._make_result(
            winner, candidates,
            metadata={
                "approval_counts": approval_counts,
                "approval_rates": approval_counts / n_voters,
                "threshold_used": ballots.approval_threshold,
            }
        )


class ScoreVoting(ElectoralSystem):
    """
    Score / Range Voting: voters give each candidate a score in [0,1];
    candidate with highest average score wins.
    Scores derived from distance (nearest = 1.0).
    """

    @property
    def name(self) -> str:
        return "Score Voting"

    def run(self, ballots: BallotProfile, candidates: CandidateSet) -> ElectionResult:
        if ballots.n_active_voters == 0:
            mean_scores = np.zeros(candidates.n_candidates)
        else:
            mean_scores = ballots.active_scores().mean(axis=0)
        winner = int(mean_scores.argmax())
        return self._make_result(
            winner, candidates,
            metadata={"mean_scores": mean_scores}
        )


class CondorcetSchulze(ElectoralSystem):
    """
    Schulze method (Condorcet family).
    Finds the candidate who beats all others in pairwise majority comparisons
    via the strongest path algorithm.
    Falls back to score voting if no Condorcet winner exists.
    """

    @property
    def name(self) -> str:
        return "Condorcet (Schulze)"

    def run(self, ballots: BallotProfile, candidates: CandidateSet) -> ElectionResult:
        n = candidates.n_candidates
        pairwise = ballots.pairwise_matrix()

        # Build strength matrix via Floyd-Warshall (Schulze)
        # strength[i][j] = strength of strongest path from i to j
        strength = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    strength[i, j] = pairwise[i, j] if pairwise[i, j] > pairwise[j, i] else 0

        for k in range(n):
            for i in range(n):
                if i != k:
                    for j in range(n):
                        if j != i and j != k:
                            strength[i, j] = max(
                                strength[i, j],
                                min(strength[i, k], strength[k, j])
                            )

        # Winner: candidate who beats all others on strongest-path
        wins = np.array([
            sum(strength[i, j] > strength[j, i] for j in range(n) if j != i)
            for i in range(n)
        ])
        winner = int(wins.argmax())

        # Check if true Condorcet winner exists (beats everyone directly)
        condorcet_winner = None
        for i in range(n):
            if all(pairwise[i, j] > 0.5 for j in range(n) if j != i):
                condorcet_winner = i
                break

        return self._make_result(
            winner, candidates,
            metadata={
                "pairwise_matrix": pairwise,
                "strength_matrix": strength,
                "wins_vector": wins,
                "condorcet_winner": condorcet_winner,
            }
        )


# ---------------------------------------------------------------------------
# Proportional systems
# ---------------------------------------------------------------------------

class PartyListPR(ElectoralSystem):
    """
    Party-list proportional representation using D'Hondt method.
    Allocates n_seats seats proportionally to vote shares.
    Includes an electoral threshold below which parties get no seats.
    By default, the outcome uses the legacy axis-median rule; alternative
    multidimensional outcome rules can be selected via outcome_rule.
    """

    def __init__(
        self,
        n_seats: int = 100,
        threshold: float = 0.05,
        outcome_rule: str = "axis_median",
    ):
        self.n_seats = n_seats
        self.threshold = threshold
        self.outcome_rule = outcome_rule

    @property
    def name(self) -> str:
        return f"Party-List PR (D'Hondt, {self.n_seats} seats)"

    @property
    def parameters(self) -> dict:
        return {
            "n_seats": self.n_seats,
            "threshold": self.threshold,
            "outcome_rule": self.outcome_rule,
        }

    def run(self, ballots: BallotProfile, candidates: CandidateSet) -> ElectionResult:
        n_c = candidates.n_candidates
        n_voters = max(ballots.n_active_voters, 1)
        vote_counts = ballots.plurality_counts().astype(float)
        vote_shares = vote_counts / n_voters

        # Apply threshold: parties below threshold get zero votes
        above_threshold = vote_shares >= self.threshold
        adjusted_votes = vote_counts * above_threshold

        if adjusted_votes.sum() == 0:
            # Fallback: no party clears threshold; give all seats to plurality winner
            winner = int(vote_counts.argmax())
            return self._make_pr_result(
                {winner: 1.0},
                candidates,
                metadata={"threshold_applied": True, "all_failed": True},
                outcome_rule=self.outcome_rule,
            )

        # D'Hondt seat allocation
        seats = np.zeros(n_c, dtype=int)
        quotients = adjusted_votes.copy()

        for _ in range(self.n_seats):
            winner_seat = int(quotients.argmax())
            seats[winner_seat] += 1
            quotients[winner_seat] = adjusted_votes[winner_seat] / (seats[winner_seat] + 1)

        seat_shares = {i: seats[i] / self.n_seats for i in range(n_c) if seats[i] > 0}
        return self._make_pr_result(
            seat_shares, candidates,
            metadata={"seats": seats, "vote_shares": vote_shares, "threshold": self.threshold},
            outcome_rule=self.outcome_rule,
        )


class MixedMemberProportional(ElectoralSystem):
    """
    Mixed Member Proportional (MMP) approximation.
    - Constituency seats: each voter votes for nearest candidate (plurality in districts).
      Approximated by assigning plurality winners in each of n_districts random subsamples.
    - List seats: top-up seats allocated to make overall result proportional (D'Hondt).
    - district_seat_fraction: fraction of total seats allocated as constituency seats.
    """

    def __init__(
        self,
        n_total_seats: int = 100,
        district_seat_fraction: float = 0.5,
        threshold: float = 0.05,
        outcome_rule: str = "axis_median",
        rng: np.random.Generator | None = None,
    ):
        self.n_total_seats = n_total_seats
        self.district_seat_fraction = district_seat_fraction
        self.threshold = threshold
        self.outcome_rule = outcome_rule
        self._rng = rng or np.random.default_rng()

    @property
    def name(self) -> str:
        return "Mixed Member Proportional (MMP)"

    @property
    def parameters(self) -> dict:
        return {
            "n_total_seats": self.n_total_seats,
            "district_seat_fraction": self.district_seat_fraction,
            "threshold": self.threshold,
            "outcome_rule": self.outcome_rule,
        }

    def run(self, ballots: BallotProfile, candidates: CandidateSet) -> ElectionResult:
        n_c = candidates.n_candidates
        n_district_seats = int(self.n_total_seats * self.district_seat_fraction)
        n_list_seats = self.n_total_seats - n_district_seats

        # Overall vote shares (list vote)
        active_plurality = ballots.active_plurality()
        n_voters = max(ballots.n_active_voters, 1)
        vote_counts = ballots.plurality_counts().astype(float)
        vote_shares = vote_counts / n_voters

        # Constituency seats via random district subsampling
        district_seats = np.zeros(n_c, dtype=int)
        voter_indices = np.arange(len(active_plurality))
        district_size = max(1, len(active_plurality) // max(n_district_seats, 1))

        self._rng.shuffle(voter_indices)
        for d in range(n_district_seats):
            start = d * district_size
            end = min(start + district_size, len(active_plurality))
            district_voters = active_plurality[voter_indices[start:end]]
            if len(district_voters) > 0:
                winner = int(np.bincount(district_voters, minlength=n_c).argmax())
                district_seats[winner] += 1

        # List seats: D'Hondt on remaining seats, adjusted for constituency seats already won
        above_threshold = vote_shares >= self.threshold
        adjusted_votes = vote_counts * above_threshold

        list_seats = np.zeros(n_c, dtype=int)
        if adjusted_votes.sum() > 0 and n_list_seats > 0:
            quotients = adjusted_votes / (district_seats + 1).astype(float)
            for _ in range(n_list_seats):
                winner_seat = int(quotients.argmax())
                list_seats[winner_seat] += 1
                total = district_seats[winner_seat] + list_seats[winner_seat]
                quotients[winner_seat] = adjusted_votes[winner_seat] / (total + 1)

        total_seats = district_seats + list_seats
        if total_seats.sum() == 0:
            total_seats[int(vote_counts.argmax())] = self.n_total_seats

        seat_shares = {
            i: total_seats[i] / total_seats.sum()
            for i in range(n_c) if total_seats[i] > 0
        }
        return self._make_pr_result(
            seat_shares, candidates,
            metadata={
                "district_seats": district_seats,
                "list_seats": list_seats,
                "total_seats": total_seats,
                "vote_shares": vote_shares,
            },
            outcome_rule=self.outcome_rule,
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def get_all_systems(rng: np.random.Generator | None = None) -> list[ElectoralSystem]:
    """Return one instance of every implemented electoral system."""
    return [
        Plurality(),
        TwoRoundRunoff(),
        InstantRunoff(),
        BordaCount(),
        ApprovalVoting(),
        ScoreVoting(),
        CondorcetSchulze(),
        PartyListPR(),
        MixedMemberProportional(rng=rng),
    ]


SYSTEM_REGISTRY: dict[str, type[ElectoralSystem]] = {
    "plurality": Plurality,
    "two_round": TwoRoundRunoff,
    "irv": InstantRunoff,
    "borda": BordaCount,
    "approval": ApprovalVoting,
    "score": ScoreVoting,
    "condorcet_schulze": CondorcetSchulze,
    "party_list_pr": PartyListPR,
    "mmp": MixedMemberProportional,
}
