"""
primaries.py
------------
Infrastructure for modelling primary elections within a two-party framework.

A primary is a nested election: one electoral system runs over a subset of
voters (party members) and a subset of candidates (party contenders) to
select a single nominee. The general election then runs over the full
electorate with only the nominees on the ballot.

Key design decisions
--------------------
- Party membership is derived from voter preferences, not assigned externally.
  Voters are assigned to a party if their nearest candidate belongs to that
  party's slate. This is consistent with the spatial voting model.
- Primary electorate composition:
    CLOSED  — only voters assigned to that party participate
    OPEN    — all voters participate in a single primary (top-K advance)
    SEMI    — party members + independents (voters equidistant between parties)

Classes
-------
PartySpec           — defines a party's name, candidate indices, and primary system
PrimaryResult       — outcome of a single party's primary
GeneralElection     — runs general election given nominees from each party
TwoPartyPrimary     — orchestrates the full primary → general pipeline
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import numpy as np

from electoral_sim.electorate import Electorate
from electoral_sim.candidates import CandidateSet, fixed_candidates
from electoral_sim.ballots import BallotProfile
from electoral_sim.systems import ElectoralSystem, Plurality
from electoral_sim.metrics import ElectionMetrics, compute_metrics
from electoral_sim.types import ElectionResult


# ── Enums ─────────────────────────────────────────────────────────────────────

class PrimaryType(Enum):
    CLOSED = "closed"   # only registered party members vote
    OPEN   = "open"     # all voters participate (top-K advance)
    SEMI   = "semi"     # party members + unaffiliated voters


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class PartySpec:
    """
    Defines one party in a two-party primary system.

    Parameters
    ----------
    name : str
        Party label (e.g. "Left Party", "Right Party").
    candidate_indices : list[int]
        Indices into the full CandidateSet that belong to this party.
    primary_system : ElectoralSystem
        The system used to select this party's nominee.
        Defaults to Plurality if not specified.
    """
    name: str
    candidate_indices: list[int]
    primary_system: ElectoralSystem = field(default_factory=Plurality)


@dataclass
class PrimaryResult:
    """Outcome of a single party primary."""
    party_name: str
    nominee_index: int                  # index in the *full* CandidateSet
    nominee_position: np.ndarray        # position vector of the nominee
    primary_system_name: str
    primary_type: PrimaryType
    n_primary_voters: int
    primary_vote_share: float           # nominee's share of primary votes
    eliminated_indices: list[int]       # candidates eliminated before nominee won
    primary_metrics: ElectionMetrics    # metrics computed over the primary electorate


@dataclass
class TwoPartyGeneralResult:
    """
    Full result of a two-party primary + general election.

    Attributes
    ----------
    primary_results : list[PrimaryResult]
        One per party.
    general_result : ElectionResult
        The general election outcome, run over the full electorate
        with only the two nominees on the ballot.
    general_metrics : ElectionMetrics
        Metrics computed over the full electorate.
    baseline_metrics : ElectionMetrics
        Metrics from running the same general system directly on the
        *full* candidate set (no primary), for comparison.
    primary_divergence : float
        Euclidean distance between the two nominees. Larger = more
        polarised outcome post-primary.
    """
    primary_results: list[PrimaryResult]
    general_result: ElectionResult
    general_metrics: ElectionMetrics
    baseline_metrics: ElectionMetrics
    primary_divergence: float


# ── Core functions ─────────────────────────────────────────────────────────────

def assign_party_membership(
    electorate: Electorate,
    candidates: CandidateSet,
    parties: list[PartySpec],
    primary_type: PrimaryType = PrimaryType.CLOSED,
) -> dict[str, np.ndarray]:
    """
    Assign each voter to a party based on which party's candidates are nearest.

    Returns
    -------
    dict mapping party_name -> boolean mask of shape (n_voters,)
    """
    from scipy.spatial.distance import cdist

    dists = cdist(electorate.preferences, candidates.positions)  # (n_voters, n_cands)

    # For each voter, find their nearest candidate overall
    nearest_cand = dists.argmin(axis=1)  # (n_voters,)

    # Build a map: candidate_index -> party_name
    cand_to_party = {}
    for party in parties:
        for idx in party.candidate_indices:
            cand_to_party[idx] = party.name

    memberships = {}

    if primary_type == PrimaryType.CLOSED:
        # Voter belongs to party whose candidate is nearest
        for party in parties:
            mask = np.array([
                cand_to_party.get(nearest_cand[i]) == party.name
                for i in range(electorate.n_voters)
            ])
            memberships[party.name] = mask

    elif primary_type == PrimaryType.OPEN:
        # All voters can vote in any primary (handled at call site)
        for party in parties:
            memberships[party.name] = np.ones(electorate.n_voters, dtype=bool)

    elif primary_type == PrimaryType.SEMI:
        # Party members + voters who are "unaffiliated" (equidistant between parties)
        party_nearest_dists = {}
        for party in parties:
            party_cand_dists = dists[:, party.candidate_indices]
            party_nearest_dists[party.name] = party_cand_dists.min(axis=1)

        party_names = [p.name for p in parties]
        for party in parties:
            own_dist   = party_nearest_dists[party.name]
            other_dist = np.min(
                [party_nearest_dists[n] for n in party_names if n != party.name],
                axis=0,
            )
            # Member if own party is closer, or within 10% of the other party's distance
            mask = own_dist <= other_dist * 1.10
            memberships[party.name] = mask

    return memberships


def run_party_primary(
    electorate: Electorate,
    candidates: CandidateSet,
    party: PartySpec,
    voter_mask: np.ndarray,
    primary_type: PrimaryType,
) -> PrimaryResult:
    """
    Run a single party's primary election.

    Parameters
    ----------
    electorate : Electorate
        Full electorate.
    candidates : CandidateSet
        Full candidate set.
    party : PartySpec
        The party whose primary to run.
    voter_mask : np.ndarray
        Boolean mask selecting which voters participate.
    primary_type : PrimaryType
    """
    # Subset voters
    primary_prefs = electorate.preferences[voter_mask]
    assert len(primary_prefs) > 0, f"Party {party.name} has no primary voters"
    primary_electorate = Electorate(primary_prefs, dim_names=electorate.dim_names)

    # Subset candidates to this party's slate
    party_positions = candidates.positions[party.candidate_indices]
    party_labels    = [candidates.labels[i] for i in party.candidate_indices]
    party_candidates = CandidateSet(party_positions, party_labels)

    # Run primary
    ballots = BallotProfile.from_preferences(primary_electorate, party_candidates)
    result  = party.primary_system.run(ballots, party_candidates)
    metrics = compute_metrics(result, primary_electorate, party_candidates)

    # Map local winner index back to global candidate index
    local_winner  = result.winner_indices[0]
    global_winner = party.candidate_indices[local_winner]

    # Primary vote share of the nominee
    vote_counts = np.bincount(ballots.plurality, minlength=party_candidates.n_candidates)
    primary_vote_share = float(vote_counts[local_winner] / ballots.n_voters)

    # Elimination order in global indices
    elim_global = [party.candidate_indices[i] for i in result.elimination_order]

    return PrimaryResult(
        party_name=party.name,
        nominee_index=global_winner,
        nominee_position=candidates.positions[global_winner].copy(),
        primary_system_name=party.primary_system.name,
        primary_type=primary_type,
        n_primary_voters=int(voter_mask.sum()),
        primary_vote_share=primary_vote_share,
        eliminated_indices=elim_global,
        primary_metrics=metrics,
    )


def run_two_party_primary(
    electorate: Electorate,
    candidates: CandidateSet,
    parties: list[PartySpec],
    general_system: ElectoralSystem,
    primary_type: PrimaryType = PrimaryType.CLOSED,
) -> TwoPartyGeneralResult:
    """
    Run the full pipeline: primaries → general election.

    1. Assign voters to parties
    2. Run each party's primary to select a nominee
    3. Run the general election with just the two nominees
    4. Compute baseline (same general system, full candidate set, no primary)

    Parameters
    ----------
    electorate : Electorate
    candidates : CandidateSet
    parties : list[PartySpec]
        Exactly two parties expected (extendable, but notebook assumes two).
    general_system : ElectoralSystem
        System used for the general election (typically Plurality or IRV).
    primary_type : PrimaryType
    """
    # ── Step 1: assign party membership ──────────────────────────────────────
    memberships = assign_party_membership(
        electorate, candidates, parties, primary_type
    )

    # ── Step 2: run primaries ─────────────────────────────────────────────────
    primary_results = []
    for party in parties:
        voter_mask = memberships[party.name]
        pr = run_party_primary(electorate, candidates, party, voter_mask, primary_type)
        primary_results.append(pr)

    # ── Step 3: general election with nominees only ───────────────────────────
    nominee_indices   = [pr.nominee_index for pr in primary_results]
    nominee_positions = np.array([pr.nominee_position for pr in primary_results])
    nominee_labels    = [f"{pr.party_name} nominee ({candidates.labels[pr.nominee_index]})"
                         for pr in primary_results]
    nominee_candidates = CandidateSet(nominee_positions, nominee_labels)

    general_ballots = BallotProfile.from_preferences(electorate, nominee_candidates)
    general_result  = general_system.run(general_ballots, nominee_candidates)
    general_metrics = compute_metrics(general_result, electorate, nominee_candidates)

    # ── Step 4: baseline (no primary — full candidate set) ────────────────────
    full_ballots      = BallotProfile.from_preferences(electorate, candidates)
    baseline_result   = general_system.run(full_ballots, candidates)
    baseline_metrics  = compute_metrics(baseline_result, electorate, candidates)

    # ── Primary divergence: distance between nominees ─────────────────────────
    if len(nominee_positions) == 2:
        divergence = float(np.linalg.norm(nominee_positions[0] - nominee_positions[1]))
    else:
        divergence = float(np.mean([
            np.linalg.norm(nominee_positions[i] - nominee_positions[j])
            for i in range(len(nominee_positions))
            for j in range(i + 1, len(nominee_positions))
        ]))

    return TwoPartyGeneralResult(
        primary_results=primary_results,
        general_result=general_result,
        general_metrics=general_metrics,
        baseline_metrics=baseline_metrics,
        primary_divergence=divergence,
    )


# ── Scenario factory helpers ──────────────────────────────────────────────────

def make_two_party_scenario(
    n_voters: int,
    left_mean: list[float],
    right_mean: list[float],
    left_cov: list[list[float]],
    right_cov: list[list[float]],
    left_weight: float,
    right_weight: float,
    left_candidates: list[dict],
    right_candidates: list[dict],
    dim_names: list[str] | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[Electorate, CandidateSet, list[PartySpec]]:
    """
    Build a two-party (electorate, candidates, parties) triple from parameters.

    Each candidate dict has keys 'name' and 'position'.
    Returns the full electorate, a combined CandidateSet, and PartySpec list.
    """
    from electoral_sim.electorate import gaussian_mixture_electorate

    rng = rng or np.random.default_rng()
    dim_names = dim_names or ["economic (left-right)", "social (libertarian-authoritarian)"]

    electorate = gaussian_mixture_electorate(
        n_voters,
        components=[
            {"weight": left_weight,  "mean": left_mean,  "cov": left_cov},
            {"weight": right_weight, "mean": right_mean, "cov": right_cov},
        ],
        rng=rng,
        dim_names=dim_names,
    )

    # Combined candidate set: left candidates first, then right
    all_positions = [c["position"] for c in left_candidates + right_candidates]
    all_labels    = [c["name"]     for c in left_candidates + right_candidates]
    candidates    = CandidateSet(np.array(all_positions), all_labels)

    n_left = len(left_candidates)
    parties = [
        PartySpec("Left Party",  list(range(n_left)),                          Plurality()),
        PartySpec("Right Party", list(range(n_left, n_left + len(right_candidates))), Plurality()),
    ]

    return electorate, candidates, parties
