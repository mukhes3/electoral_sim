"""Paired synthetic experiments for the liquid-democracy tutorial.

All losses use the original electorate, never a delegate-only subsample.
Network draws are nested within independently seeded electorates; uncertainty
summaries average within seed before bootstrapping seed blocks.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from electoral_sim.ballots import BallotProfile
from electoral_sim.candidates import fixed_candidates
from electoral_sim.electorate import Electorate
from electoral_sim.liquid import DelegationProfile, LiquidApprovalVoting, LiquidScoreVoting
from electoral_sim.systems import (
    ApprovalVoting, BordaCount, CondorcetSchulze, InstantRunoff,
    Plurality, ScoreVoting, TwoRoundRunoff,
)

FAMILIES = ("Consensus", "Polarized", "Fragmented", "Asymmetric")
SLATES = ("Broad coverage", "No central candidate")
MECHANISMS = ("Nearby", "Random", "Popular")
PARTICIPATION_PATTERNS = ("Random absence", "Bloc-concentrated absence")
STANDARD_RULES = {
    "Plurality": Plurality, "Runoff": TwoRoundRunoff, "IRV": InstantRunoff,
    "Borda": BordaCount, "Approval": ApprovalVoting, "Score": ScoreVoting,
    "Schulze": CondorcetSchulze,
}
LIQUID_RULES = {"Approval": LiquidApprovalVoting, "Score": LiquidScoreVoting}


@dataclass(frozen=True)
class ExperimentConfig:
    seeds: tuple[int, ...] = tuple(range(12))
    network_draws: int = 3
    n_voters: int = 400
    families: tuple[str, ...] = FAMILIES
    slates: tuple[str, ...] = SLATES
    delegation_fractions: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    absence_fractions: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75)
    approval_threshold: float = 0.25

    def __post_init__(self):
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be nonempty and unique")
        if any(not isinstance(s, (int, np.integer)) or s < 0 for s in self.seeds):
            raise ValueError("seeds must be nonnegative integers")
        if self.network_draws < 1 or self.n_voters < 12:
            raise ValueError("use at least one network draw and twelve voters")
        for values, allowed in [(self.families, FAMILIES), (self.slates, SLATES)]:
            if not values or len(set(values)) != len(values) or not set(values) <= set(allowed):
                raise ValueError("scenario names must be nonempty, unique, and supported")
        for values, upper_inclusive in [(self.delegation_fractions, True),
                                         (self.absence_fractions, False)]:
            if not values or len(set(values)) != len(values):
                raise ValueError("fractions must be nonempty and unique")
            if any(not np.isfinite(f) or f < 0 or f > 1 or
                   (not upper_inclusive and f == 1) for f in values):
                raise ValueError("delegation fractions must be in [0,1]; absence in [0,1)")
        if not np.isfinite(self.approval_threshold) or self.approval_threshold < 0:
            raise ValueError("approval_threshold must be finite and nonnegative")


def make_case(family: str, slate: str, seed: int, n_voters: int):
    """One electorate per family/seed, shared exactly across candidate slates."""
    specifications = {
        "Consensus": ([1.0], [[0.5, 0.5]], 0.12),
        "Polarized": ([0.5, 0.5], [[0.22, 0.42], [0.78, 0.58]], 0.09),
        "Fragmented": ([0.34, 0.33, 0.33], [[0.20, 0.25], [0.45, 0.78], [0.82, 0.35]], 0.08),
        "Asymmetric": ([0.7, 0.3], [[0.30, 0.40], [0.80, 0.65]], 0.09),
    }
    proportions, means, spread = specifications[family]
    rng = np.random.default_rng(np.random.SeedSequence([seed, FAMILIES.index(family), 10]))
    groups = rng.choice(len(means), size=n_voters, p=proportions)
    preferences = np.clip(np.asarray(means)[groups] + rng.normal(0, spread, (n_voters, 2)), 0, 1)
    electorate = Electorate(preferences, dim_names=["economic", "social"],
                            group_ids=groups,
                            group_names={int(g): f"Bloc {g}" for g in np.unique(groups)})
    candidates = fixed_candidates(
        [[0.18, 0.35], [0.35, 0.72], [0.50, 0.50], [0.70, 0.28], [0.84, 0.70]],
        ["Left", "Upper-left", "Center", "Lower-right", "Right"],
    )
    if slate == "No central candidate":
        candidates = candidates.subset([0, 1, 3, 4])
    elif slate != "Broad coverage":
        raise ValueError(slate)
    return electorate, candidates


def representative_pool(preferences):
    """Six distinct voters nearest fixed policy anchors; no expertise assumption."""
    anchors = [[0.15, 0.2], [0.2, 0.8], [0.5, 0.3], [0.5, 0.7], [0.85, 0.2], [0.85, 0.8]]
    chosen = []
    for anchor in anchors:
        distances = np.linalg.norm(preferences - anchor, axis=1)
        distances[chosen] = np.inf
        chosen.append(int(distances.argmin()))
    return np.asarray(chosen)


def assign_delegates(preferences, selected, available, mechanism, rng):
    """Selected voters delegate directly to available voters; all others are direct."""
    selected, available = np.asarray(selected, dtype=int), np.asarray(available, dtype=int)
    if len(available) == 0 or np.intersect1d(selected, available).size:
        raise ValueError("delegate pool must be nonempty and disjoint from delegators")
    targets = np.full(len(preferences), -1, dtype=int)
    if mechanism == "Nearby":
        distances = np.linalg.norm(preferences[selected, None] - preferences[available], axis=2)
        targets[selected] = available[distances.argmin(axis=1)]
    elif mechanism == "Random":
        # Draw for every original voter to keep choices paired across fractions.
        targets[selected] = rng.choice(available, size=len(preferences))[selected]
    elif mechanism == "Popular":
        targets[selected] = available[preferences[available, 0].argmax()]
    else:
        raise ValueError(mechanism)
    return DelegationProfile(targets)


def _case_metrics(electorate, candidates):
    d = np.linalg.norm(electorate.preferences[:, None] - candidates.positions, axis=2)
    group_means = np.stack([d[electorate.group_ids == g].mean(axis=0)
                            for g in np.unique(electorate.group_ids)])
    return d, d.mean(axis=0), group_means.max(axis=0)


def _metrics(winner, baseline, distances, losses, worst):
    difference = distances[:, winner] - distances[:, baseline]
    return {
        "winner": int(winner), "mean_distance": losses[winner],
        "worst_group_distance": worst[winner],
        "distance_delta": losses[winner] - losses[baseline],
        "worst_group_delta": worst[winner] - worst[baseline],
        "prefer_liquid": float((difference < -1e-9).mean()),
        "prefer_direct": float((difference > 1e-9).mean()),
        "indifferent": float((np.abs(difference) <= 1e-9).mean()),
        "winner_changed": float(winner != baseline),
    }


def _power(resolution):
    shares = resolution.effective_weights / resolution.represented_power
    return {"represented_power": resolution.represented_power,
            "unrepresented_power": resolution.unrepresented_power,
            "largest_share": shares.max(), "effective_casters": 1 / np.square(shares).sum()}


def run_system_comparison(config=ExperimentConfig()):
    """Return direct (once per case) and liquid (paired nested draws) result frames."""
    direct_rows, liquid_rows = [], []
    for family in config.families:
        for seed in config.seeds:
            for slate in config.slates:
                e, c = make_case(family, slate, seed, config.n_voters)
                ballots = BallotProfile.from_preferences(e, c, config.approval_threshold)
                distances, losses, worst = _case_metrics(e, c)
                base = {"family": family, "slate": slate, "seed": seed}
                winners = {}
                for label, cls in STANDARD_RULES.items():
                    winner = cls().run(ballots, c).winner_indices[0]
                    winners[label] = winner
                    direct_rows.append({**base, "system": label,
                                        **_metrics(winner, winner, distances, losses, worst)})
                pool = representative_pool(e.preferences)
                eligible = np.setdiff1d(np.arange(e.n_voters), pool)
                for draw in range(config.network_draws):
                    key = [seed, FAMILIES.index(family), draw, 20]
                    order = np.random.default_rng(np.random.SeedSequence(key)).permutation(eligible)
                    for fraction in config.delegation_fractions:
                        selected = order[:round(fraction * len(eligible))]
                        for mechanism in MECHANISMS:
                            rng = np.random.default_rng(np.random.SeedSequence(key + [1]))
                            profile = assign_delegates(e.preferences, selected, pool, mechanism, rng)
                            power = _power(profile.resolve())
                            for label, cls in LIQUID_RULES.items():
                                winner = cls(profile).run(ballots, c).winner_indices[0]
                                liquid_rows.append({**base, "draw": draw, "fraction": fraction,
                                                    "actual_fraction": len(selected) / e.n_voters,
                                                    "mechanism": mechanism, "system": label,
                                                    **_metrics(winner, winners[label], distances, losses, worst),
                                                    **power})
    return pd.DataFrame(direct_rows), pd.DataFrame(liquid_rows)


def absence_order(electorate, pattern, rng):
    """Nested absence order with equal total absence across patterns.

    Bloc-concentrated absence takes the bloc with the highest mean first-axis
    position first, and then other voters. Each part is randomly ordered.
    """
    order = rng.permutation(electorate.n_voters)
    if pattern == "Random absence":
        return order
    if pattern != "Bloc-concentrated absence":
        raise ValueError(pattern)
    groups = np.unique(electorate.group_ids)
    target = max(groups, key=lambda g: electorate.preferences[electorate.group_ids == g, 0].mean())
    targeted = electorate.group_ids[order] == target
    return np.concatenate([order[targeted], order[~targeted]])


def run_participation_comparison(config=ExperimentConfig()):
    """Compare full participation, matched abstention, and costless delegation.

    Voters unwilling to cast their own ballot either abstain or delegate to a
    willing voter. All voters participate in the delegation condition. The same
    non-casting subset is used across liquid rules and delegate mechanisms.
    """
    rows = []
    for family in config.families:
        for seed in config.seeds:
            for slate in config.slates:
                e, c = make_case(family, slate, seed, config.n_voters)
                full = BallotProfile.from_preferences(e, c, config.approval_threshold)
                distances, losses, worst = _case_metrics(e, c)
                full_winners = {rule: STANDARD_RULES[rule]().run(full, c).winner_indices[0]
                                for rule in LIQUID_RULES}
                for draw in range(config.network_draws):
                    for pattern in PARTICIPATION_PATTERNS:
                        key = [seed, FAMILIES.index(family), draw, 30]
                        order = absence_order(e, pattern, np.random.default_rng(np.random.SeedSequence(key)))
                        for fraction in config.absence_fractions:
                            selected = order[:min(round(fraction * e.n_voters), e.n_voters - 1)]
                            active = np.ones(e.n_voters, dtype=bool)
                            active[selected] = False
                            available = np.flatnonzero(active)
                            abstention = replace(full, active_voter_mask=active)
                            profiles = {
                                mechanism: assign_delegates(
                                    e.preferences, selected, available, mechanism,
                                    np.random.default_rng(np.random.SeedSequence(key + [1])),
                                ) for mechanism in MECHANISMS
                            }
                            for rule, cls in LIQUID_RULES.items():
                                reference = full_winners[rule]
                                absent_winner = STANDARD_RULES[rule]().run(abstention, c).winner_indices[0]
                                conditions = [("Full participation", "Direct", reference, None),
                                              ("Abstention", "Direct", absent_winner, None)]
                                for mechanism, profile in profiles.items():
                                    result = cls(profile).run(full, c)
                                    conditions.append(("Delegation", mechanism, result.winner_indices[0],
                                                       profile.resolve()))
                                for condition, mechanism, winner, resolution in conditions:
                                    ref_distance = np.linalg.norm(c.positions[winner] - c.positions[reference])
                                    absent_ref_distance = np.linalg.norm(c.positions[absent_winner] - c.positions[reference])
                                    if resolution is None:
                                        count = len(available) if condition == "Abstention" else e.n_voters
                                        power = {"represented_power": count, "unrepresented_power": 0,
                                                 "largest_share": 1 / count, "effective_casters": count}
                                    else:
                                        power = _power(resolution)
                                    rows.append({
                                        "family": family, "slate": slate, "seed": seed, "draw": draw,
                                        "pattern": pattern, "fraction": fraction,
                                        "actual_fraction": len(selected) / e.n_voters,
                                        "system": rule, "condition": condition, "mechanism": mechanism,
                                        **_metrics(winner, absent_winner, distances, losses, worst), **power,
                                        "full_winner": reference, "abstention_winner": absent_winner,
                                        "reference_distance": ref_distance,
                                        "reference_gap_reduction": absent_ref_distance - ref_distance,
                                        "winner_matches_full": float(winner == reference),
                                        "distance_delta_full": losses[winner] - losses[reference],
                                        "worst_group_delta_full": worst[winner] - worst[reference],
                                        "absent_power": len(selected) if condition == "Abstention" else 0,
                                    })
    return pd.DataFrame(rows)


def summarize_by_seed(frame, groups, metrics, bootstrap_draws=1000, seed=904):
    """Means and percentile 95% bootstrap intervals over independent seed blocks.

    First average draws within each family/slate/seed, then average scenario
    cells equally within each seed. Shared seeds across slates and families are
    resampled together. A singleton seed has undefined (NaN) interval bounds.
    Pointwise intervals are descriptive and are not simultaneous tests.
    """
    case_keys = list(dict.fromkeys([*groups, "family", "slate", "seed"]))
    case_means = frame.groupby(case_keys, observed=True)[metrics].mean().reset_index()
    seed_means = case_means.groupby([*groups, "seed"], observed=True)[metrics].mean().reset_index()
    rng = np.random.default_rng(seed)
    rows = []
    for key, subset in seed_means.groupby(groups, observed=True, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        values = subset[metrics].to_numpy(dtype=float)
        count = len(values)
        if count > 1:
            draws = rng.integers(0, count, size=(bootstrap_draws, count))
            low, high = np.quantile(values[draws].mean(axis=1), [0.025, 0.975], axis=0)
        else:
            low = high = np.full(len(metrics), np.nan)
        for j, metric in enumerate(metrics):
            rows.append({**dict(zip(groups, key)), "metric": metric,
                         "mean": values[:, j].mean(), "low": low[j], "high": high[j],
                         "n_seeds": count})
    return pd.DataFrame(rows)
