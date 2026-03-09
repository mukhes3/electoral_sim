"""CandidateSet: candidate positions in the same [0,1]^N preference space as voters."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import numpy as np
from electoral_sim.electorate import Electorate


@dataclass
class CandidateSet:
    positions: np.ndarray
    labels: list[str]

    def __post_init__(self):
        assert self.positions.ndim == 2
        assert len(self.labels) == self.n_candidates
        assert np.all(self.positions >= 0) and np.all(self.positions <= 1)

    @property
    def n_candidates(self) -> int:
        return self.positions.shape[0]

    @property
    def n_dims(self) -> int:
        return self.positions.shape[1]

    def __repr__(self) -> str:
        lines = [f"CandidateSet ({self.n_candidates} candidates, {self.n_dims}D):"]
        for label, pos in zip(self.labels, self.positions):
            lines.append(f"  {label}: {np.round(pos, 3)}")
        return "\n".join(lines)


def fixed_candidates(positions, labels=None):
    pos = np.array(positions, dtype=float)
    if labels is None:
        labels = [f"C{i}" for i in range(len(pos))]
    return CandidateSet(pos, labels)


def sampled_candidates(electorate, n_candidates, labels=None, rng=None):
    rng = rng or np.random.default_rng()
    idx = rng.choice(electorate.n_voters, size=n_candidates, replace=False)
    positions = electorate.preferences[idx]
    if labels is None:
        labels = [f"C{i}" for i in range(n_candidates)]
    return CandidateSet(positions, labels)


def evenly_spaced_candidates(n_candidates, n_dims, labels=None):
    if n_dims == 1:
        positions = np.linspace(0.1, 0.9, n_candidates).reshape(-1, 1)
    else:
        first = np.linspace(0.1, 0.9, n_candidates)
        rest = np.full((n_candidates, n_dims - 1), 0.5)
        positions = np.hstack([first.reshape(-1, 1), rest])
    if labels is None:
        labels = [f"C{i}" for i in range(n_candidates)]
    return CandidateSet(positions, labels)


def from_config(config):
    cconf = config["candidates"]
    ctype = cconf["type"]
    if ctype == "fixed":
        positions = [c["position"] for c in cconf["positions"]]
        labels = [c["name"] for c in cconf["positions"]]
        return fixed_candidates(positions, labels)
    elif ctype == "evenly_spaced":
        n_dims = config.get("n_dims", 2)
        return evenly_spaced_candidates(cconf["n_candidates"], n_dims)
    else:
        raise ValueError(f"Unknown candidate type: {ctype}")
