"""
BallotProfile: all ballot formats derived from the same voter preference vectors.

All ballot types are derived deterministically from (electorate, candidates)
under the assumption of sincere voting. This ensures fair comparison across
electoral systems.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.spatial.distance import cdist

from electoral_sim.electorate import Electorate
from electoral_sim.candidates import CandidateSet


@dataclass
class BallotProfile:
    """
    All ballot formats for a given (electorate, candidates) pair.

    Attributes
    ----------
    plurality : np.ndarray, shape (n_voters,)
        Index of each voter's nearest (most-preferred) candidate.

    rankings : np.ndarray, shape (n_voters, n_candidates)
        Full preference ranking. rankings[i] gives candidate indices
        sorted from most-preferred (index 0) to least-preferred.

    scores : np.ndarray, shape (n_voters, n_candidates)
        Continuous score in [0, 1]. score = 1 - normalized_distance.
        Higher is better.

    approvals : np.ndarray, shape (n_voters, n_candidates)
        Binary approval. 1 if candidate is within approval_threshold of voter.

    distances : np.ndarray, shape (n_voters, n_candidates)
        Raw Euclidean distances between each voter and each candidate.
        Kept for diagnostics and alternative metric computation.

    approval_threshold : float
        The distance threshold used to derive approvals.
    """
    plurality: np.ndarray
    rankings: np.ndarray
    scores: np.ndarray
    approvals: np.ndarray
    distances: np.ndarray
    approval_threshold: float
    n_voters: int
    n_candidates: int

    @classmethod
    def from_preferences(
        cls,
        electorate: Electorate,
        candidates: CandidateSet,
        approval_threshold: float | None = None,
    ) -> BallotProfile:
        """
        Derive all ballot types from voter preference vectors (sincere voting).

        Parameters
        ----------
        electorate : Electorate
        candidates : CandidateSet
        approval_threshold : float, optional
            Distance threshold for approval ballots.
            If None, defaults to the median voter-to-nearest-candidate distance,
            which is a principled data-driven default.
        """
        assert electorate.n_dims == candidates.n_dims, (
            f"Electorate dims ({electorate.n_dims}) != "
            f"candidate dims ({candidates.n_dims})"
        )

        # Euclidean distances: shape (n_voters, n_candidates)
        distances = cdist(electorate.preferences, candidates.positions, metric="euclidean")

        # Plurality: nearest candidate per voter
        plurality = distances.argmin(axis=1)

        # Rankings: argsort distances ascending (nearest = rank 0)
        rankings = distances.argsort(axis=1)

        # Scores: 1 - normalized distance (so nearest candidate scores 1.0)
        max_possible = np.sqrt(electorate.n_dims)  # max Euclidean dist in [0,1]^N
        scores = 1.0 - (distances / max_possible)

        # Approval threshold default: median nearest-candidate distance
        if approval_threshold is None:
            nearest_dist = distances.min(axis=1)
            approval_threshold = float(np.median(nearest_dist)) * 1.5

        approvals = (distances <= approval_threshold).astype(int)

        # Guard: ensure every voter approves at least their nearest candidate
        # (prevents degenerate cases with very tight thresholds)
        for i in range(electorate.n_voters):
            if approvals[i].sum() == 0:
                approvals[i, plurality[i]] = 1

        return cls(
            plurality=plurality,
            rankings=rankings,
            scores=scores,
            approvals=approvals,
            distances=distances,
            approval_threshold=approval_threshold,
            n_voters=electorate.n_voters,
            n_candidates=candidates.n_candidates,
        )

    def pairwise_matrix(self) -> np.ndarray:
        """
        Compute pairwise majority matrix M where M[i,j] = fraction of voters
        who prefer candidate i over candidate j.
        Shape: (n_candidates, n_candidates).
        Used by Condorcet methods.
        """
        n = self.n_candidates
        M = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    # voter prefers i over j if i appears before j in their ranking
                    rank_i = np.where(self.rankings == i, np.arange(self.n_candidates), 999)
                    rank_j = np.where(self.rankings == j, np.arange(self.n_candidates), 999)
                    # Vectorized: find position of i and j in each voter's ranking
                    pos_i = (self.rankings == i).argmax(axis=1)
                    pos_j = (self.rankings == j).argmax(axis=1)
                    M[i, j] = (pos_i < pos_j).mean()
        return M

    def borda_scores(self, base: int | None = None) -> np.ndarray:
        """
        Compute Borda scores for each candidate.
        Standard Borda: n-1 points for rank 1, n-2 for rank 2, ..., 0 for last.
        Returns shape (n_candidates,).
        """
        n = self.n_candidates
        if base is None:
            base = n - 1
        # For each voter, position in rankings -> points awarded
        # rankings[i] = [c2, c0, c1] means c2 gets n-1 points, c0 gets n-2, etc.
        points = np.zeros(n)
        for rank_pos in range(n):
            candidates_at_rank = self.rankings[:, rank_pos]
            pts = base - rank_pos
            if pts < 0:
                pts = 0
            np.add.at(points, candidates_at_rank, pts)
        return points

    def summary_for_rl(self) -> np.ndarray:
        """
        Fixed-size feature vector for use as RL observation.
        Includes: score distribution moments, approval rates,
        pairwise win fractions, plurality vote shares.
        """
        # Plurality vote shares: (n_candidates,)
        vote_shares = np.bincount(self.plurality, minlength=self.n_candidates) / self.n_voters

        # Approval rates per candidate: (n_candidates,)
        approval_rates = self.approvals.mean(axis=0)

        # Mean and std of scores per candidate: (n_candidates,) x 2
        score_mean = self.scores.mean(axis=0)
        score_std = self.scores.std(axis=0)

        # Pairwise win matrix flattened: (n_candidates^2,)
        pw = self.pairwise_matrix().flatten()

        return np.concatenate([vote_shares, approval_rates, score_mean, score_std, pw])
