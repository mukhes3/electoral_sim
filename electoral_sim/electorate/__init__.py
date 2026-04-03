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
    group_ids : np.ndarray, optional
        Integer group identifier for each voter. Useful for demographic or
        coalition-level analysis. If omitted, the electorate is treated as
        unlabeled and all existing behavior is unchanged.
    group_names : dict[int, str], optional
        Mapping from integer group ids to display names.
    """
    preferences: np.ndarray
    dim_names: list[str] | None = None
    group_ids: np.ndarray | None = None
    group_names: dict[int, str] | None = None

    def __post_init__(self):
        assert self.preferences.ndim == 2, "preferences must be 2D (n_voters, n_dims)"
        assert np.all(self.preferences >= 0) and np.all(self.preferences <= 1), \
            "All preferences must be in [0, 1]"
        if self.dim_names is None:
            self.dim_names = [f"dim_{i}" for i in range(self.n_dims)]
        if self.group_ids is not None:
            self.group_ids = np.asarray(self.group_ids, dtype=int)
            assert self.group_ids.shape == (self.n_voters,), (
                f"group_ids must have shape ({self.n_voters},), got {self.group_ids.shape}"
            )
            unique_ids = np.unique(self.group_ids)
            if self.group_names is None:
                self.group_names = {int(group_id): f"group_{int(group_id)}" for group_id in unique_ids}
            else:
                self.group_names = {
                    int(group_id): str(name) for group_id, name in self.group_names.items()
                }
                missing = [int(group_id) for group_id in unique_ids if int(group_id) not in self.group_names]
                if missing:
                    raise ValueError(
                        "group_names is missing labels for group ids: "
                        + ", ".join(str(group_id) for group_id in missing)
                    )
        elif self.group_names is not None:
            raise ValueError("group_names cannot be provided without group_ids")

    @property
    def n_voters(self) -> int:
        return self.preferences.shape[0]

    @property
    def n_dims(self) -> int:
        return self.preferences.shape[1]

    @property
    def has_groups(self) -> bool:
        """Whether voters carry optional group labels."""
        return self.group_ids is not None

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
        group_ids = None if self.group_ids is None else self.group_ids[idx]
        group_names = None if self.group_names is None else dict(self.group_names)
        return Electorate(
            self.preferences[idx],
            dim_names=self.dim_names,
            group_ids=group_ids,
            group_names=group_names,
        )

    def group_indices(self) -> dict[int, np.ndarray]:
        """Boolean masks for each labeled group in the electorate."""
        if self.group_ids is None:
            return {}
        return {
            int(group_id): self.group_ids == group_id
            for group_id in np.unique(self.group_ids)
        }

    def group_labels(self) -> dict[int, str]:
        """Return a copy of the id-to-name mapping for labeled groups."""
        return {} if self.group_names is None else dict(self.group_names)

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
    group: str | None = None,
) -> Electorate:
    """Single Gaussian cluster, clipped to [0,1]^N."""
    rng = rng or np.random.default_rng()
    mean = np.array(mean)
    cov = np.array(cov)
    samples = rng.multivariate_normal(mean, cov, size=n_voters)
    samples = np.clip(samples, 0.0, 1.0)
    if group is None:
        return Electorate(samples, dim_names=dim_names)
    group_ids = np.zeros(n_voters, dtype=int)
    return Electorate(
        samples,
        dim_names=dim_names,
        group_ids=group_ids,
        group_names={0: group},
    )


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

    group_labels = [component.get("group") for component in components]
    use_groups = any(label is not None for label in group_labels)
    group_name_to_id: dict[str, int] = {}
    group_ids_list = []
    samples_list = []

    for idx, (c, n) in enumerate(zip(components, counts)):
        if n == 0:
            continue
        mean = np.array(c["mean"])
        cov = np.array(c["cov"])
        s = rng.multivariate_normal(mean, cov, size=n)
        samples_list.append(s)
        if use_groups:
            group_name = c.get("group")
            if group_name is None:
                group_name = f"component_{idx}"
            group_name = str(group_name)
            group_id = group_name_to_id.setdefault(group_name, len(group_name_to_id))
            group_ids_list.append(np.full(n, group_id, dtype=int))

    samples = np.clip(np.vstack(samples_list), 0.0, 1.0)
    if use_groups:
        group_ids = np.concatenate(group_ids_list)
        order = rng.permutation(len(samples))
        samples = samples[order]
        group_ids = group_ids[order]
        group_names = {group_id: name for name, group_id in group_name_to_id.items()}
        return Electorate(
            samples,
            dim_names=dim_names,
            group_ids=group_ids,
            group_names=group_names,
        )

    rng.shuffle(samples)  # randomize voter order
    return Electorate(samples, dim_names=dim_names)


def uniform_electorate(
    n_voters: int,
    n_dims: int,
    rng: np.random.Generator | None = None,
    dim_names: list[str] | None = None,
    group: str | None = None,
) -> Electorate:
    """Uniformly distributed voters over [0,1]^N."""
    rng = rng or np.random.default_rng()
    samples = rng.uniform(0.0, 1.0, size=(n_voters, n_dims))
    if group is None:
        return Electorate(samples, dim_names=dim_names)
    group_ids = np.zeros(n_voters, dtype=int)
    return Electorate(
        samples,
        dim_names=dim_names,
        group_ids=group_ids,
        group_names={0: group},
    )


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
        return gaussian_electorate(
            n_voters,
            ec["mean"],
            ec["cov"],
            rng,
            dim_names,
            group=ec.get("group"),
        )
    elif etype == "gaussian_mixture":
        ec = config["electorate"]
        return gaussian_mixture_electorate(n_voters, ec["components"], rng, dim_names)
    elif etype == "uniform":
        n_dims = config.get("n_dims", 2)
        ec = config["electorate"]
        return uniform_electorate(
            n_voters,
            n_dims,
            rng,
            dim_names,
            group=ec.get("group"),
        )
    else:
        raise ValueError(f"Unknown electorate type: {etype}")
