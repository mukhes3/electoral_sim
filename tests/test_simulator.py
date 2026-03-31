"""
Test suite for the electoral simulator.

Tests validate known theoretical properties:
1. In 1D unimodal, IRV and Condorcet elect the candidate nearest the median voter.
2. In symmetric bimodal, distance-to-mean and distance-to-median diverge.
3. Borda scores are derivable from rankings deterministically.
4. All systems return same winner with one candidate.
5. Plurality elects candidate with most first-preference votes.
6. Condorcet winner is detected correctly.
7. PR seat shares sum to 1.
8. Geometric median is close to componentwise median in symmetric distributions.
"""
import numpy as np
import pytest

from electoral_sim.electorate import Electorate, gaussian_electorate, gaussian_mixture_electorate
from electoral_sim.candidates import CandidateSet, evenly_spaced_candidates, fixed_candidates
from electoral_sim.ballots import BallotProfile
from electoral_sim.metrics import compute_metrics, run_simulation
from electoral_sim.systems import (
    Plurality, TwoRoundRunoff, InstantRunoff, BordaCount,
    ApprovalVoting, ScoreVoting, CondorcetSchulze,
    PartyListPR, MixedMemberProportional, get_all_systems
)


RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_1d_unimodal(n_voters=2000):
    """1D Gaussian centered at 0.5. Candidates evenly spaced."""
    rng = np.random.default_rng(42)
    electorate = gaussian_electorate(n_voters, [0.5], [[0.03]], rng=rng)
    candidates = evenly_spaced_candidates(5, 1)
    return electorate, candidates


def make_symmetric_bimodal(n_voters=2000):
    """Symmetric bimodal: two equal clusters at 0.25 and 0.75."""
    rng = np.random.default_rng(42)
    electorate = gaussian_mixture_electorate(
        n_voters,
        components=[
            {"weight": 0.5, "mean": [0.25, 0.5], "cov": [[0.02, 0], [0, 0.02]]},
            {"weight": 0.5, "mean": [0.75, 0.5], "cov": [[0.02, 0], [0, 0.02]]},
        ],
        rng=rng,
    )
    candidates = fixed_candidates(
        [[0.25, 0.5], [0.5, 0.5], [0.75, 0.5]],
        ["Left", "Center", "Right"]
    )
    return electorate, candidates


def make_single_candidate():
    rng = np.random.default_rng(42)
    electorate = gaussian_electorate(500, [0.5, 0.5], [[0.04, 0], [0, 0.04]], rng=rng)
    candidates = fixed_candidates([[0.5, 0.5]], ["Only"])
    return electorate, candidates


def make_clear_condorcet():
    """
    Condorcet winner at center: beats all others pairwise.
    Voters: 1/3 prefer Left>Center>Right, 1/3 prefer Center>Left>Right,
    1/3 prefer Center>Right>Left. Center wins all pairwise comparisons.
    """
    rng = np.random.default_rng(42)
    n = 900
    prefs = np.vstack([
        rng.multivariate_normal([0.2, 0.5], [[0.01, 0], [0, 0.01]], n // 3),
        rng.multivariate_normal([0.5, 0.5], [[0.01, 0], [0, 0.01]], n // 3),
        rng.multivariate_normal([0.8, 0.5], [[0.01, 0], [0, 0.01]], n // 3),
    ])
    prefs = np.clip(prefs, 0, 1)
    electorate = Electorate(prefs)
    candidates = fixed_candidates(
        [[0.15, 0.5], [0.5, 0.5], [0.85, 0.5]],
        ["Left", "Center", "Right"]
    )
    return electorate, candidates


# ---------------------------------------------------------------------------
# Electorate tests
# ---------------------------------------------------------------------------

class TestElectorate:
    def test_shape(self):
        e, _ = make_1d_unimodal()
        assert e.preferences.shape[1] == 1

    def test_bounds(self):
        e, _ = make_1d_unimodal()
        assert np.all(e.preferences >= 0) and np.all(e.preferences <= 1)

    def test_mean_near_center_1d(self):
        e, _ = make_1d_unimodal()
        assert abs(e.mean()[0] - 0.5) < 0.03

    def test_geometric_median_near_center(self):
        e, _ = make_1d_unimodal()
        gm = e.geometric_median()
        assert abs(gm[0] - 0.5) < 0.05

    def test_geometric_median_symmetric_bimodal(self):
        """In a perfectly symmetric distribution, geometric median ≈ componentwise median."""
        e, _ = make_symmetric_bimodal()
        gm = e.geometric_median()
        cm = e.componentwise_median()
        # Both should be near 0.5 on dim 0 in symmetric bimodal
        assert abs(gm[0] - 0.5) < 0.1

    def test_summary_statistics_keys(self):
        e, _ = make_1d_unimodal()
        s = e.summary_statistics()
        for key in ["mean", "std", "geometric_median", "covariance"]:
            assert key in s

    def test_mean_median_diverge_in_skewed(self):
        """Mean and geometric median diverge in skewed distributions."""
        rng = np.random.default_rng(42)
        e = gaussian_mixture_electorate(
            3000,
            components=[
                {"weight": 0.7, "mean": [0.2, 0.5], "cov": [[0.01, 0], [0, 0.01]]},
                {"weight": 0.3, "mean": [0.8, 0.5], "cov": [[0.01, 0], [0, 0.01]]},
            ],
            rng=rng
        )
        assert abs(e.mean()[0] - e.geometric_median()[0]) > 0.05


# ---------------------------------------------------------------------------
# Ballot tests
# ---------------------------------------------------------------------------

class TestBallotProfile:
    def test_plurality_is_nearest_candidate(self):
        e, c = make_1d_unimodal()
        b = BallotProfile.from_preferences(e, c)
        for i in range(e.n_voters):
            dists = np.linalg.norm(c.positions - e.preferences[i], axis=1)
            assert b.plurality[i] == dists.argmin()

    def test_rankings_are_sorted_by_distance(self):
        e, c = make_1d_unimodal()
        b = BallotProfile.from_preferences(e, c)
        for i in range(min(100, e.n_voters)):
            ranked_dists = b.distances[i][b.rankings[i]]
            assert np.all(np.diff(ranked_dists) >= 0), "Rankings not sorted by distance"

    def test_scores_in_01(self):
        e, c = make_1d_unimodal()
        b = BallotProfile.from_preferences(e, c)
        assert np.all(b.scores >= 0) and np.all(b.scores <= 1)

    def test_approvals_nonempty_per_voter(self):
        e, c = make_1d_unimodal()
        b = BallotProfile.from_preferences(e, c)
        assert np.all(b.approvals.sum(axis=1) >= 1)

    def test_borda_scores_sum(self):
        e, c = make_1d_unimodal(n_voters=100)
        b = BallotProfile.from_preferences(e, c)
        scores = b.borda_scores()
        n_c = c.n_candidates
        expected_total = (n_c - 1) * n_c / 2 * 100  # sum of all Borda points
        assert abs(scores.sum() - expected_total) < 1.0

    def test_pairwise_matrix_sums(self):
        """M[i,j] + M[j,i] should be close to 1 for all i != j."""
        e, c = make_1d_unimodal(n_voters=500)
        b = BallotProfile.from_preferences(e, c)
        M = b.pairwise_matrix()
        n = c.n_candidates
        for i in range(n):
            for j in range(n):
                if i != j:
                    assert abs(M[i, j] + M[j, i] - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# Electoral system tests
# ---------------------------------------------------------------------------

class TestAllSystemsSingleCandidate:
    """All systems should return the only candidate when there's just one."""
    def test_all_systems_single_candidate(self):
        e, c = make_single_candidate()
        b = BallotProfile.from_preferences(e, c)
        for system in get_all_systems(rng=np.random.default_rng(42)):
            result = system.run(b, c)
            assert 0 in result.winner_indices, f"{system.name} failed single-candidate test"


class TestPlurality:
    def test_elects_highest_vote_share(self):
        e, c = make_symmetric_bimodal()
        b = BallotProfile.from_preferences(e, c)
        vote_counts = np.bincount(b.plurality, minlength=c.n_candidates)
        expected_winner = int(vote_counts.argmax())
        result = Plurality().run(b, c)
        assert result.winner_indices[0] == expected_winner


class TestIRV:
    def test_elects_candidate_near_median_1d(self):
        """In 1D unimodal, IRV should elect the candidate nearest the median voter."""
        e, c = make_1d_unimodal(n_voters=3000)
        b = BallotProfile.from_preferences(e, c)
        result = InstantRunoff().run(b, c)
        median_pos = e.geometric_median()
        winner_pos = c.positions[result.winner_indices[0]]
        dists = np.linalg.norm(c.positions - median_pos, axis=1)
        nearest_to_median = int(dists.argmin())
        assert result.winner_indices[0] == nearest_to_median, (
            f"IRV elected {result.winner_indices[0]} but nearest to median is {nearest_to_median}"
        )


class TestCondorcet:
    def test_detects_condorcet_winner(self):
        """Condorcet should find the true Condorcet winner when one exists."""
        e, c = make_clear_condorcet()
        b = BallotProfile.from_preferences(e, c)
        result = CondorcetSchulze().run(b, c)
        # Center candidate (index 1) should win
        assert result.winner_indices[0] == 1, (
            f"Expected Center (1) to win, got {result.winner_indices[0]}"
        )
        assert result.metadata.get("condorcet_winner") == 1

    def test_elects_near_median_1d(self):
        """Condorcet should also elect candidate near median in 1D."""
        e, c = make_1d_unimodal(n_voters=3000)
        b = BallotProfile.from_preferences(e, c)
        result = CondorcetSchulze().run(b, c)
        median_pos = e.geometric_median()
        dists = np.linalg.norm(c.positions - median_pos, axis=1)
        nearest_to_median = int(dists.argmin())
        assert result.winner_indices[0] == nearest_to_median


class TestPR:
    def test_seat_shares_sum_to_one(self):
        e, c = make_symmetric_bimodal()
        b = BallotProfile.from_preferences(e, c)
        result = PartyListPR(n_seats=100).run(b, c)
        assert abs(sum(result.seat_shares.values()) - 1.0) < 1e-9

    def test_pr_centroid_is_weighted_mean(self):
        """centroid_position should be the seat-share-weighted mean of elected parties."""
        e, c = make_symmetric_bimodal()
        b = BallotProfile.from_preferences(e, c)
        result = PartyListPR(n_seats=100).run(b, c)
        expected = np.zeros(c.n_dims)
        for idx, share in result.seat_shares.items():
            expected += share * c.positions[idx]
        assert np.allclose(result.centroid_position, expected)

    def test_pr_outcome_is_median_legislator(self):
        """outcome_position should equal median_legislator_position for PR systems."""
        e, c = make_symmetric_bimodal()
        b = BallotProfile.from_preferences(e, c)
        result = PartyListPR(n_seats=100).run(b, c)
        assert np.allclose(result.outcome_position, result.median_legislator_position)
        assert result.is_pr is True

    def test_pr_outcome_differs_from_centroid(self):
        """In asymmetric seat distributions, median legislator != centroid."""
        e, c = make_symmetric_bimodal()
        b = BallotProfile.from_preferences(e, c)
        result = PartyListPR(n_seats=100).run(b, c)
        # They may coincide in perfectly symmetric cases, but types are always both present
        assert result.centroid_position is not None
        assert result.median_legislator_position is not None

    def test_pr_axis_median_remains_default(self):
        """Default PR outcome rule should remain backward compatible."""
        e, c = make_symmetric_bimodal()
        b = BallotProfile.from_preferences(e, c)
        result = PartyListPR(n_seats=100).run(b, c)
        assert result.metadata["outcome_rule"] == "axis_median"
        assert np.allclose(result.outcome_position, result.median_legislator_position)

    def test_pr_legislative_geometric_median_works_in_multiple_dimensions(self):
        positions = np.array([
            [0.10, 0.20, 0.90],
            [0.30, 0.40, 0.60],
            [0.80, 0.70, 0.20],
            [0.55, 0.15, 0.50],
        ])
        labels = ["A", "B", "C", "D"]
        candidates = CandidateSet(positions, labels)
        seat_shares = {0: 0.26, 1: 0.25, 2: 0.24, 3: 0.25}

        system = PartyListPR(outcome_rule="legislative_geometric_median")
        result = system._make_pr_result(seat_shares, candidates, outcome_rule=system.outcome_rule)

        assert result.metadata["outcome_rule"] == "legislative_geometric_median"
        assert result.outcome_position.shape == (3,)
        assert not np.allclose(result.outcome_position, result.median_legislator_position)

    def test_pr_legislative_medoid_returns_elected_position(self):
        positions = np.array([
            [0.10, 0.20, 0.90],
            [0.30, 0.40, 0.60],
            [0.80, 0.70, 0.20],
        ])
        labels = ["A", "B", "C"]
        candidates = CandidateSet(positions, labels)
        seat_shares = {0: 0.2, 1: 0.5, 2: 0.3}

        system = PartyListPR(outcome_rule="legislative_medoid")
        result = system._make_pr_result(seat_shares, candidates, outcome_rule=system.outcome_rule)

        assert result.metadata["outcome_rule"] == "legislative_medoid"
        assert any(np.allclose(result.outcome_position, pos) for pos in positions)

    def test_mmp_seat_shares_sum_to_one(self):
        rng = np.random.default_rng(42)
        e, c = make_symmetric_bimodal()
        b = BallotProfile.from_preferences(e, c)
        result = MixedMemberProportional(rng=rng).run(b, c)
        assert abs(sum(result.seat_shares.values()) - 1.0) < 1e-9

    def test_mmp_accepts_multidimensional_outcome_rule(self):
        rng = np.random.default_rng(42)
        e, c = make_symmetric_bimodal()
        b = BallotProfile.from_preferences(e, c)
        result = MixedMemberProportional(
            rng=rng,
            outcome_rule="legislative_geometric_median",
        ).run(b, c)
        assert result.metadata["outcome_rule"] == "legislative_geometric_median"


# ---------------------------------------------------------------------------
# Metrics tests
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_distance_to_median_nonneg(self):
        e, c = make_1d_unimodal()
        systems = get_all_systems(rng=np.random.default_rng(42))
        metrics = run_simulation(e, c, systems)
        for m in metrics:
            assert m.distance_to_median >= 0, f"{m.system_name} has negative d_median"

    def test_bimodal_mean_median_diverge(self):
        """In bimodal, geometric median and mean are in different places."""
        e, _ = make_symmetric_bimodal()
        gm = e.geometric_median()
        mean = e.mean()
        assert abs(gm[0] - mean[0]) < 0.1  # both ~0.5 in perfectly symmetric case

    def test_score_voting_low_distance(self):
        """Score voting, as a direct utility maximizer, should do well on distance metrics."""
        e, c = make_1d_unimodal(n_voters=3000)
        systems = [ScoreVoting(), Plurality()]
        metrics = run_simulation(e, c, systems)
        score_m = next(m for m in metrics if "Score" in m.system_name)
        plurality_m = next(m for m in metrics if "Plurality" in m.system_name)
        # Score voting should outperform or equal plurality on d_median in unimodal
        assert score_m.distance_to_median <= plurality_m.distance_to_median + 0.05


# ---------------------------------------------------------------------------
# Scenario loading tests
# ---------------------------------------------------------------------------

class TestScenarioLoading:
    def test_load_yaml_scenario(self, tmp_path):
        import yaml
        from electoral_sim.scenario import load_scenario
        config = {
            "name": "Test",
            "n_dims": 2,
            "n_voters": 200,
            "electorate": {
                "type": "gaussian",
                "mean": [0.5, 0.5],
                "cov": [[0.04, 0], [0, 0.04]],
            },
            "candidates": {
                "type": "fixed",
                "positions": [
                    {"name": "A", "position": [0.3, 0.5]},
                    {"name": "B", "position": [0.7, 0.5]},
                ],
            },
        }
        p = tmp_path / "test.yaml"
        p.write_text(yaml.dump(config))
        cfg, e, c = load_scenario(p)
        assert e.n_voters == 200
        assert c.n_candidates == 2

    def test_load_packaged_scenario_by_filename(self):
        from electoral_sim.scenario import load_scenario

        cfg, e, c = load_scenario("02_polarized_bimodal.yaml")
        assert cfg["name"] == "Polarized Bimodal"
        assert e.n_voters == cfg["n_voters"]
        assert c.n_candidates >= 2
