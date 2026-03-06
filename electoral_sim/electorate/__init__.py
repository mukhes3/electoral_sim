"""
Electorate: voter preference distributions in [0,1]^N.

Voters are represented as points in an N-dimensional unit hypercube.
Each dimension represents a policy/value axis (e.g. economic, social).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import numpy as np
from scipy.spatial.distance import cdist


@dataclass
class Electorate:
    """
    A collection of voters with preference vectors in [0,1]^N.

    Parameters
    ----------
    preferences : np.ndarray
        Shape (n_voters, n_dims). Each row is a voter's preference vector.
    dim_names : list[str], optional
        Human-readable names for each dimension.
    """
    preferences: np.ndarray
    dim_names: list[str] | None = None

    def __post_init__(self):
        assert self.preferences.ndim == 2, "preferences must be 2D (n_voters, n_dims)"
        assert np.all(self.preferences >= 0) and np.all(self.preferences <= 1), \
            "All preferences must be in [0, 1]"
        if self.dim_names is None:
            self.dim_names = [f"dim_{i}" for i in range(self.n_dims)]

    @property
    def n_voters(self) -> int:
        return self.preferences.shape[0]

    @property
    def n_dims(self) -> int:
        return self.preferences.shape[1]

    def mean(self) -> np.ndarray:
        """Arithmetic mean of voter preferences. Shape: (n_dims,)."""
        return self.preferences.mean(axis=0)

    def geometric_median(self, tol: float = 1e-6, max_iter: int = 300) -> np.ndarray:
        """
        Geometric median (L1 median) via Weiszfeld's algorithm.
        Minimizes sum of Euclidean distances to all voters.
        Unique and well-defined in R^N; more robust to outliers than mean.
        """
        # Initialize at the mean
        x = self.preferences.mean(axis=0)
        for _ in range(max_iter):
            dists = np.linalg.norm(self.preferences - x, axis=1)
            # Avoid division by zero for points coinciding with x
            nonzero = dists > 1e-10
            if not nonzero.any():
                break
            weights = np.where(nonzero, 1.0 / dists, 0.0)
            x_new = (weights[:, None] * self.preferences).sum(axis=0) / weights.sum()
            if np.linalg.norm(x_new - x) < tol:
                break
            x = x_new
        return x

    def componentwise_median(self) -> np.ndarray:
        """Componentwise median. Not the geometric median; here for comparison."""
        return np.median(self.preferences, axis=0)

    def subsample(self, n: int, rng: np.random.Generator | None = None) -> Electorate:
        """Return a random subsample of n voters."""
        rng = rng or np.random.default_rng()
        idx = rng.choice(self.n_voters, size=n, replace=False)
        return Electorate(self.preferences[idx], dim_names=self.dim_names)

    def summary_statistics(self) -> dict:
        """
        Fixed-size summary of the electorate for use as RL observations.
        Returns moments and pairwise structure that don't depend on n_voters.
        """
        prefs = self.preferences
        mean = self.mean()
        std = prefs.std(axis=0)
        skew = (((prefs - mean) / (std + 1e-8)) ** 3).mean(axis=0)
        kurt = (((prefs - mean) / (std + 1e-8)) ** 4).mean(axis=0)
        cov = np.cov(prefs.T) if self.n_dims > 1 else np.array([[prefs.var()]])
        return {
            "mean": mean,
            "std": std,
            "skewness": skew,
            "kurtosis": kurt,
            "covariance": cov,
            "geometric_median": self.geometric_median(),
            "n_voters": self.n_voters,
            "n_dims": self.n_dims,
        }


# ---------------------------------------------------------------------------
# Factory functions for common distributions
# ---------------------------------------------------------------------------

def gaussian_electorate(
    n_voters: int,
    mean: Sequence[float],
    cov: np.ndarray | Sequence[Sequence[float]],
    rng: np.random.Generator | None = None,
    dim_names: list[str] | None = None,
) -> Electorate:
    """Single Gaussian cluster, clipped to [0,1]^N."""
    rng = rng or np.random.default_rng()
    mean = np.array(mean)
    cov = np.array(cov)
    samples = rng.multivariate_normal(mean, cov, size=n_voters)
    samples = np.clip(samples, 0.0, 1.0)
    return Electorate(samples, dim_names=dim_names)


def gaussian_mixture_electorate(
    n_voters: int,
    components: list[dict],
    rng: np.random.Generator | None = None,
    dim_names: list[str] | None = None,
) -> Electorate:
    """
    Gaussian mixture model electorate.

    Each component dict has keys:
        weight : float       (will be normalized)
        mean   : list[float]
        cov    : list[list[float]]

    Example (polarized bimodal):
        components = [
            {"weight": 0.48, "mean": [0.25, 0.45], "cov": [[0.03, 0.01],[0.01, 0.03]]},
            {"weight": 0.48, "mean": [0.75, 0.55], "cov": [[0.03,-0.01],[-0.01,0.03]]},
            {"weight": 0.04, "mean": [0.50, 0.30], "cov": [[0.01, 0.00],[0.00, 0.01]]},
        ]
    """
    rng = rng or np.random.default_rng()
    weights = np.array([c["weight"] for c in components], dtype=float)
    weights /= weights.sum()
    counts = rng.multinomial(n_voters, weights)

    samples_list = []
    for c, n in zip(components, counts):
        if n == 0:
            continue
        mean = np.array(c["mean"])
        cov = np.array(c["cov"])
        s = rng.multivariate_normal(mean, cov, size=n)
        samples_list.append(s)

    samples = np.clip(np.vstack(samples_list), 0.0, 1.0)
    rng.shuffle(samples)  # randomize voter order
    return Electorate(samples, dim_names=dim_names)


def uniform_electorate(
    n_voters: int,
    n_dims: int,
    rng: np.random.Generator | None = None,
    dim_names: list[str] | None = None,
) -> Electorate:
    """Uniformly distributed voters over [0,1]^N."""
    rng = rng or np.random.default_rng()
    samples = rng.uniform(0.0, 1.0, size=(n_voters, n_dims))
    return Electorate(samples, dim_names=dim_names)


def from_config(config: dict, rng: np.random.Generator | None = None) -> Electorate:
    """
    Build an Electorate from a scenario config dict.
    Supports types: 'gaussian', 'gaussian_mixture', 'uniform'.
    """
    rng = rng or np.random.default_rng()
    n_voters = config["n_voters"]
    dim_names = config.get("dim_names", None)
    etype = config["electorate"]["type"]

    if etype == "gaussian":
        ec = config["electorate"]
        return gaussian_electorate(n_voters, ec["mean"], ec["cov"], rng, dim_names)
    elif etype == "gaussian_mixture":
        ec = config["electorate"]
        return gaussian_mixture_electorate(n_voters, ec["components"], rng, dim_names)
    elif etype == "uniform":
        n_dims = config.get("n_dims", 2)
        return uniform_electorate(n_voters, n_dims, rng, dim_names)
    else:
        raise ValueError(f"Unknown electorate type: {etype}")
