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

    active_voter_mask : np.ndarray, shape (n_voters,)
        Boolean mask indicating which voters cast a ballot. Defaults to all True.
    """
    plurality: np.ndarray
    rankings: np.ndarray
    scores: np.ndarray
    approvals: np.ndarray
    distances: np.ndarray
    approval_threshold: float
    n_voters: int
    n_candidates: int
    active_voter_mask: np.ndarray | None = None

    def __post_init__(self):
        if self.active_voter_mask is None:
            self.active_voter_mask = np.ones(self.n_voters, dtype=bool)
        else:
            self.active_voter_mask = np.asarray(self.active_voter_mask, dtype=bool)
        assert self.active_voter_mask.shape == (self.n_voters,)

    @property
    def n_active_voters(self) -> int:
        return int(self.active_voter_mask.sum())

    def active_plurality(self) -> np.ndarray:
        return self.plurality[self.active_voter_mask]

    def active_rankings(self) -> np.ndarray:
        return self.rankings[self.active_voter_mask]

    def active_scores(self) -> np.ndarray:
        return self.scores[self.active_voter_mask]

    def active_approvals(self) -> np.ndarray:
        return self.approvals[self.active_voter_mask]

    def plurality_counts(self) -> np.ndarray:
        votes = self.active_plurality()
        valid = votes[votes >= 0]
        return np.bincount(valid, minlength=self.n_candidates)

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
            active_voter_mask=np.ones(electorate.n_voters, dtype=bool),
        )

    @classmethod
    def from_strategy(
        cls,
        electorate: Electorate,
        candidates: CandidateSet,
        strategy,
        approval_threshold: float | None = None,
        context=None,
    ) -> BallotProfile:
        """
        Derive ballots through a strategy model.

        This is backward compatible with the existing sincere workflow:
        use SincereStrategy (or strategy=None at higher-level APIs) to recover
        the same ballots produced by from_preferences().
        """
        return strategy.generate_ballots(
            electorate,
            candidates,
            approval_threshold=approval_threshold,
            context=context,
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
        rankings = self.active_rankings()
        if len(rankings) == 0:
            return M
        for i in range(n):
            for j in range(n):
                if i != j:
                    ranked_i = (rankings == i).any(axis=1)
                    ranked_j = (rankings == j).any(axis=1)
                    pos_i = np.where(ranked_i, (rankings == i).argmax(axis=1), self.n_candidates + 1)
                    pos_j = np.where(ranked_j, (rankings == j).argmax(axis=1), self.n_candidates + 1)
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
        rankings = self.active_rankings()
        for rank_pos in range(n):
            candidates_at_rank = rankings[:, rank_pos]
            pts = base - rank_pos
            if pts < 0:
                pts = 0
            ranked_mask = candidates_at_rank >= 0
            np.add.at(points, candidates_at_rank[ranked_mask], pts)
        return points

    def summary_for_rl(self) -> np.ndarray:
        """
        Fixed-size feature vector for use as RL observation.
        Includes: score distribution moments, approval rates,
        pairwise win fractions, plurality vote shares.
        """
        if self.n_active_voters == 0:
            zeros = np.zeros(self.n_candidates)
            pw = self.pairwise_matrix().flatten()
            return np.concatenate([zeros, zeros, zeros, zeros, pw])

        # Plurality vote shares: (n_candidates,)
        vote_shares = self.plurality_counts() / max(self.n_active_voters, 1)

        # Approval rates per candidate: (n_candidates,)
        approval_rates = self.active_approvals().mean(axis=0)

        # Mean and std of scores per candidate: (n_candidates,) x 2
        active_scores = self.active_scores()
        score_mean = active_scores.mean(axis=0)
        score_std = active_scores.std(axis=0)

        # Pairwise win matrix flattened: (n_candidates^2,)
        pw = self.pairwise_matrix().flatten()

        return np.concatenate([vote_shares, approval_rates, score_mean, score_std, pw])
