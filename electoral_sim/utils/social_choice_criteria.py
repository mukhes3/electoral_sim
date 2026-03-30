"""Notebook-friendly social choice criterion checks."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import inspect
from typing import Callable

import numpy as np

from electoral_sim.ballots import BallotProfile
from electoral_sim.candidates import CandidateSet
from electoral_sim.social_choice import SocialRanking, social_prefers


RankingBuilder = Callable[..., SocialRanking]


@dataclass
class PairwiseCriterionCheck:
    """Summary of a pairwise social-choice criterion on one profile."""

    criterion_name: str
    satisfied: bool
    details: str
    violating_pairs: list[tuple[int, int]]
    violating_labels: list[tuple[str, str]]

    @property
    def n_violations(self) -> int:
        return len(self.violating_pairs)


@dataclass
class ProfileDictatorshipDiagnostic:
    """
    Profile-level non-dictatorship diagnostic.

    This is intentionally weaker than Arrow's theorem statement, which is about
    an entire rule across all possible preference profiles. Here we only ask
    whether the social ranking on one profile exactly matches one or more
    individual voters on all pairwise comparisons.
    """

    satisfied: bool
    matching_voter_indices: list[int]
    details: str


@dataclass
class IIAComparison:
    """Comparison of a candidate pair before and after removing alternatives."""

    candidate_pair: tuple[int, int]
    candidate_labels: tuple[str, str]
    removed_candidates: list[int]
    removed_labels: list[str]
    full_preference: bool
    restricted_preference: bool
    satisfied: bool
    full_ranking: SocialRanking
    restricted_ranking: SocialRanking


def _candidate_labels(candidates: CandidateSet | None, n_candidates: int) -> list[str]:
    if candidates is None:
        return [f"C{i}" for i in range(n_candidates)]
    return list(candidates.labels)


def _build_ranking(
    ranking_builder: RankingBuilder,
    ballots: BallotProfile,
    candidates: CandidateSet | None = None,
) -> SocialRanking:
    signature = inspect.signature(ranking_builder)
    n_positional = sum(
        1
        for param in signature.parameters.values()
        if param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    )
    if candidates is not None and n_positional >= 2:
        return ranking_builder(ballots, candidates)
    return ranking_builder(ballots)


def _voter_prefers(voter_ranking: np.ndarray, candidate_a: int, candidate_b: int) -> bool:
    pos_a = np.where(voter_ranking == candidate_a)[0]
    pos_b = np.where(voter_ranking == candidate_b)[0]
    rank_a = int(pos_a[0]) if len(pos_a) else len(voter_ranking) + 1
    rank_b = int(pos_b[0]) if len(pos_b) else len(voter_ranking) + 1
    return rank_a < rank_b


def unanimous_preference_pairs(ballots: BallotProfile) -> list[tuple[int, int]]:
    """
    Return all ordered candidate pairs unanimously supported by active voters.

    Each returned tuple ``(a, b)`` means every active voter ranks ``a`` above
    ``b`` on the given profile.
    """
    if ballots.n_active_voters == 0:
        return []

    counts = ballots.pairwise_preference_counts()
    total = ballots.n_active_voters
    unanimous_pairs: list[tuple[int, int]] = []
    for candidate_a in range(ballots.n_candidates):
        for candidate_b in range(ballots.n_candidates):
            if candidate_a == candidate_b:
                continue
            if np.isclose(counts[candidate_a, candidate_b], total):
                unanimous_pairs.append((candidate_a, candidate_b))
    return unanimous_pairs


def check_unanimity(
    ballots: BallotProfile,
    ranking: SocialRanking,
    candidates: CandidateSet | None = None,
) -> PairwiseCriterionCheck:
    """Check whether the social ranking respects every unanimous pairwise preference."""
    labels = _candidate_labels(candidates, ballots.n_candidates)
    unanimous_pairs = unanimous_preference_pairs(ballots)
    violations = [(a, b) for a, b in unanimous_pairs if not social_prefers(ranking, a, b)]
    violation_labels = [(labels[a], labels[b]) for a, b in violations]

    if ballots.n_active_voters == 0:
        details = "No active voters, so unanimity is vacuously satisfied on this profile."
    elif violations:
        pair_text = ", ".join(f"{left} over {right}" for left, right in violation_labels)
        details = f"The social ranking violates unanimous voter preferences for: {pair_text}."
    elif unanimous_pairs:
        details = "The social ranking respects every unanimous pairwise voter preference."
    else:
        details = "There are no unanimous pairwise voter preferences to test on this profile."

    return PairwiseCriterionCheck(
        criterion_name="Unanimity",
        satisfied=len(violations) == 0,
        details=details,
        violating_pairs=violations,
        violating_labels=violation_labels,
    )


def find_dictatorial_voters(ballots: BallotProfile, ranking: SocialRanking) -> list[int]:
    """
    Return active voter indices whose pairwise preferences exactly match the social ranking.

    This is a profile-level diagnostic, useful for notebook explanations. It is
    not itself a proof that a rule is dictatorial in Arrow's theorem sense.
    """
    rankings = ballots.active_rankings()
    active_voter_indices = np.flatnonzero(ballots.active_voter_mask)
    if len(rankings) == 0:
        return []

    matching_voters: list[int] = []
    candidate_pairs = list(combinations(range(ballots.n_candidates), 2))
    for local_idx, voter_ranking in enumerate(rankings):
        matches_social = True
        for candidate_a, candidate_b in candidate_pairs:
            voter_prefers_ab = _voter_prefers(voter_ranking, candidate_a, candidate_b)
            social_prefers_ab = social_prefers(ranking, candidate_a, candidate_b)
            if voter_prefers_ab != social_prefers_ab:
                matches_social = False
                break
        if matches_social:
            matching_voters.append(int(active_voter_indices[local_idx]))
    return matching_voters


def check_non_dictatorship(
    ballots: BallotProfile,
    ranking: SocialRanking,
) -> ProfileDictatorshipDiagnostic:
    """Run a profile-level non-dictatorship diagnostic against one social ranking."""
    matching_voters = find_dictatorial_voters(ballots, ranking)
    if ballots.n_active_voters == 0:
        details = "No active voters, so there is no dictator on this profile."
    elif matching_voters:
        voter_text = ", ".join(str(idx) for idx in matching_voters)
        details = (
            "The social ranking exactly matches voter(s) "
            f"{voter_text} on all pairwise comparisons in this profile."
        )
    else:
        details = "No active voter's full pairwise ordering matches the social ranking on this profile."

    return ProfileDictatorshipDiagnostic(
        satisfied=len(matching_voters) == 0,
        matching_voter_indices=matching_voters,
        details=details,
    )


def compare_iia(
    ballots: BallotProfile,
    ranking_builder: RankingBuilder,
    candidate_a: int,
    candidate_b: int,
    removed_candidates: list[int] | tuple[int, ...] | np.ndarray,
    candidates: CandidateSet | None = None,
) -> IIAComparison:
    """
    Compare a pairwise social preference before and after removing alternatives.

    The ranking builder should return a ``SocialRanking`` for the supplied
    ballots, such as ``borda_social_ranking`` or ``plurality_social_ranking``.
    """
    if candidate_a == candidate_b:
        raise ValueError("candidate_a and candidate_b must be different.")

    removed = [int(idx) for idx in removed_candidates]
    if candidate_a in removed or candidate_b in removed:
        raise ValueError("Removed candidates must not include the focal pair.")
    if len(set(removed)) != len(removed):
        raise ValueError("removed_candidates must not contain duplicates.")

    keep = [idx for idx in range(ballots.n_candidates) if idx not in removed]
    if candidate_a not in keep or candidate_b not in keep:
        raise ValueError("The focal candidate pair must remain after restriction.")

    labels = _candidate_labels(candidates, ballots.n_candidates)
    restricted_ballots = ballots.restrict_to_candidates(keep)
    restricted_candidates = candidates.subset(keep) if candidates is not None else None
    remap = {old_idx: new_idx for new_idx, old_idx in enumerate(keep)}

    full_ranking = _build_ranking(ranking_builder, ballots, candidates)
    restricted_ranking = _build_ranking(ranking_builder, restricted_ballots, restricted_candidates)

    full_preference = social_prefers(full_ranking, candidate_a, candidate_b)
    restricted_preference = social_prefers(
        restricted_ranking,
        remap[candidate_a],
        remap[candidate_b],
    )

    return IIAComparison(
        candidate_pair=(candidate_a, candidate_b),
        candidate_labels=(labels[candidate_a], labels[candidate_b]),
        removed_candidates=removed,
        removed_labels=[labels[idx] for idx in removed],
        full_preference=full_preference,
        restricted_preference=restricted_preference,
        satisfied=full_preference == restricted_preference,
        full_ranking=full_ranking,
        restricted_ranking=restricted_ranking,
    )


def find_iia_violations(
    ballots: BallotProfile,
    ranking_builder: RankingBuilder,
    candidates: CandidateSet | None = None,
    max_removed: int = 1,
) -> list[IIAComparison]:
    """
    Search for IIA violations by removing up to ``max_removed`` alternatives.

    The default only checks single-candidate removals, which is usually enough
    for clean notebook examples and avoids combinatorial blow-up.
    """
    if max_removed < 1:
        raise ValueError("max_removed must be at least 1.")

    violations: list[IIAComparison] = []
    all_candidates = list(range(ballots.n_candidates))
    for candidate_a, candidate_b in combinations(all_candidates, 2):
        others = [idx for idx in all_candidates if idx not in (candidate_a, candidate_b)]
        if not others:
            continue
        for n_removed in range(1, min(max_removed, len(others)) + 1):
            for removed in combinations(others, n_removed):
                comparison = compare_iia(
                    ballots=ballots,
                    ranking_builder=ranking_builder,
                    candidate_a=candidate_a,
                    candidate_b=candidate_b,
                    removed_candidates=list(removed),
                    candidates=candidates,
                )
                if not comparison.satisfied:
                    violations.append(comparison)
    return violations


def check_iia(
    ballots: BallotProfile,
    ranking_builder: RankingBuilder,
    candidates: CandidateSet | None = None,
    max_removed: int = 1,
) -> PairwiseCriterionCheck:
    """Check whether a ranking rule preserves pairwise social order under restriction."""
    labels = _candidate_labels(candidates, ballots.n_candidates)
    violations = find_iia_violations(
        ballots=ballots,
        ranking_builder=ranking_builder,
        candidates=candidates,
        max_removed=max_removed,
    )
    violating_pairs = [comparison.candidate_pair for comparison in violations]
    violating_labels = [(labels[a], labels[b]) for a, b in violating_pairs]

    if ballots.n_candidates < 3:
        details = "IIA needs at least three candidates to be meaningfully tested."
    elif violations:
        example = violations[0]
        removed_text = ", ".join(example.removed_labels)
        details = (
            "Removing other candidates changes at least one social pairwise ordering. "
            f"For example, {example.candidate_labels[0]} vs {example.candidate_labels[1]} flips "
            f"when removing {removed_text}."
        )
    else:
        details = "No IIA violations were found in the searched candidate restrictions."

    return PairwiseCriterionCheck(
        criterion_name="Independence of irrelevant alternatives",
        satisfied=len(violations) == 0,
        details=details,
        violating_pairs=violating_pairs,
        violating_labels=violating_labels,
    )
