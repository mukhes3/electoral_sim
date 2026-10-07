"""Transitive, whole-ballot delegation and liquid approval/score elections.

Voter indices always refer to the original BallotProfile rows. Delegation changes
whose ballot carries each participating voter's unit of power, not preferences
used to evaluate the result. Delegate selection is explicit, not inferred.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from electoral_sim.ballots import BallotProfile
from electoral_sim.candidates import CandidateSet
from electoral_sim.systems import ElectoralSystem
from electoral_sim.types import ElectionResult

__all__ = [
    "DelegationProfile", "DelegationResolution",
    "LiquidApprovalVoting", "LiquidScoreVoting",
]


@dataclass(frozen=True)
class DelegationResolution:
    """Resolved destinations (-1 for unrepresented voters) and ballot weights.

    Inactive voters have destination -1 and supply no power. Active voters whose
    chains terminate at an inactive voter also have destination -1, but count
    toward ``unrepresented_power``. Cycle members vote directly; incoming chains
    use the ballot of the first cycle member they encounter.
    """

    destinations: np.ndarray
    effective_weights: np.ndarray
    cycles: tuple[tuple[int, ...], ...]
    participating_power: int
    represented_power: int
    unrepresented_power: int


class DelegationProfile:
    """One outgoing delegation per voter; -1 or a self-reference means direct.

    ``delegates[i] = j`` transfers voter i's power to voter j, transitively.
    Input is copied and made read-only. Delegation applies to the whole ballot.
    Abstaining voters cannot act as intermediaries or final ballot casters.
    """

    def __init__(self, delegates):
        values = np.asarray(delegates)
        if values.ndim != 1 or values.size == 0:
            raise ValueError("delegates must be a non-empty 1D integer sequence")
        if values.dtype.kind not in "iu":
            raise ValueError("delegates must contain integer voter indices")
        if np.any(values < -1) or np.any(values >= len(values)):
            raise ValueError("delegate indices must be -1 or valid voter indices")
        self._delegates = values.astype(np.int64, copy=True)
        self._delegates.setflags(write=False)

    @property
    def delegates(self) -> np.ndarray:
        """A copy of the requested delegation targets."""
        return self._delegates.copy()

    @property
    def n_voters(self) -> int:
        return len(self._delegates)

    def resolve(self, active_voter_mask=None) -> DelegationResolution:
        """Resolve chains in O(n) time without recursive graph traversal.

        Each active voter supplies one unit. Represented plus unrepresented
        power equals the number of active voters. An all-inactive profile is
        resolvable, although an election with no represented power is rejected.
        """
        n = self.n_voters
        if active_voter_mask is None:
            active = np.ones(n, dtype=bool)
        else:
            active = np.asarray(active_voter_mask)
            if active.shape != (n,) or active.dtype.kind != "b":
                raise ValueError("active_voter_mask must be a boolean vector of length n_voters")

        destinations = np.full(n, -2, dtype=np.int64)  # -2: unresolved
        destinations[~active] = -1
        cycles = []
        for start in range(n):
            if destinations[start] != -2:
                continue
            path = []
            offsets = {}
            node = start
            while destinations[node] == -2:
                if node in offsets:
                    cycle = tuple(path[offsets[node]:])
                    cycles.append(cycle)
                    for member in cycle:
                        destinations[member] = member
                    break
                offsets[node] = len(path)
                path.append(node)
                target = int(self._delegates[node])
                if target == -1 or target == node:
                    destinations[node] = node
                    break
                node = target
            for member in reversed(path):
                if destinations[member] == -2:
                    destinations[member] = destinations[self._delegates[member]]

        represented = destinations >= 0
        weights = np.bincount(destinations[represented], minlength=n)
        participating_power = int(active.sum())
        represented_power = int(weights.sum())
        destinations.setflags(write=False)
        weights.setflags(write=False)
        return DelegationResolution(
            destinations=destinations,
            effective_weights=weights,
            cycles=tuple(cycles),
            participating_power=participating_power,
            represented_power=represented_power,
            unrepresented_power=participating_power - represented_power,
        )


class _LiquidVoting(ElectoralSystem):
    def __init__(self, delegation: DelegationProfile):
        if not isinstance(delegation, DelegationProfile):
            raise TypeError("delegation must be a DelegationProfile")
        self.delegation = delegation

    def _resolve(
        self, ballots: BallotProfile, candidates: CandidateSet,
    ) -> DelegationResolution:
        if ballots.n_voters != self.delegation.n_voters:
            raise ValueError("delegation and ballots must have the same voter count")
        if ballots.n_candidates != candidates.n_candidates or candidates.n_candidates == 0:
            raise ValueError("ballots and candidates must have matching nonzero candidate counts")
        resolution = self.delegation.resolve(ballots.active_voter_mask)
        if resolution.represented_power == 0:
            raise ValueError("liquid election has no represented voting power")
        return resolution

    def _metadata(self, resolution: DelegationResolution) -> dict:
        return {
            "delegates": self.delegation.delegates,
            "delegation_destinations": resolution.destinations.copy(),
            "effective_weights": resolution.effective_weights.copy(),
            "delegation_cycles": resolution.cycles,
            "cycle_policy": "direct_fallback",
            "participating_power": resolution.participating_power,
            "represented_power": resolution.represented_power,
            "unrepresented_power": resolution.unrepresented_power,
        }


class LiquidApprovalVoting(_LiquidVoting):
    """Approval totals weighted by resolved delegation power.

    Ties select the lowest candidate index, as in ordinary ApprovalVoting.
    Rates are normalized by represented power, excluding failed delegations.
    """

    @property
    def name(self) -> str:
        return "Liquid Approval Voting"

    def run(self, ballots: BallotProfile, candidates: CandidateSet) -> ElectionResult:
        resolution = self._resolve(ballots, candidates)
        counts = resolution.effective_weights @ ballots.approvals
        return self._make_result(
            int(counts.argmax()), candidates,
            metadata={
                **self._metadata(resolution),
                "approval_counts": counts,
                "approval_rates": counts / resolution.represented_power,
                "threshold_used": ballots.approval_threshold,
            },
        )


class LiquidScoreVoting(_LiquidVoting):
    """Score averages weighted by resolved delegation power.

    Ties select the lowest candidate index, as in ordinary ScoreVoting.
    """

    @property
    def name(self) -> str:
        return "Liquid Score Voting"

    def run(self, ballots: BallotProfile, candidates: CandidateSet) -> ElectionResult:
        resolution = self._resolve(ballots, candidates)
        means = (resolution.effective_weights @ ballots.scores) / resolution.represented_power
        return self._make_result(
            int(means.argmax()), candidates,
            metadata={**self._metadata(resolution), "mean_scores": means},
        )
