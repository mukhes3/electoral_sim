import numpy as np

from electoral_sim.ballots import BallotProfile
from electoral_sim.candidates import fixed_candidates
from electoral_sim.electorate import Electorate
from electoral_sim.metrics import run_simulation
from electoral_sim.strategies import (
    ApprovalThresholdStrategy,
    PluralityCompromiseStrategy,
    RankedBuryingStrategy,
    RankedTruncationStrategy,
    ScoreMaxMinStrategy,
    SincereStrategy,
    TurnoutStrategy,
    VotingContext,
)
from electoral_sim.systems import InstantRunoff, Plurality, ScoreVoting


def make_compromise_case():
    left = np.tile([0.20], (40, 1))
    center = np.tile([0.49], (20, 1))
    right = np.tile([0.80], (40, 1))
    electorate = Electorate(np.vstack([left, center, right]), dim_names=["economic"])
    candidates = fixed_candidates([[0.20], [0.49], [0.80]], ["Left", "Center", "Right"])
    return electorate, candidates


def make_ranked_case():
    electorate = Electorate(
        np.array([[0.10], [0.15], [0.45], [0.55], [0.80], [0.85]]),
        dim_names=["economic"],
    )
    candidates = fixed_candidates([[0.10], [0.45], [0.55], [0.85]], ["A", "B", "C", "D"])
    return electorate, candidates


def test_sincere_strategy_matches_existing_ballot_generation():
    electorate, candidates = make_compromise_case()

    direct = BallotProfile.from_preferences(electorate, candidates)
    via_strategy = BallotProfile.from_strategy(
        electorate,
        candidates,
        strategy=SincereStrategy(),
    )

    assert np.array_equal(via_strategy.plurality, direct.plurality)
    assert np.array_equal(via_strategy.rankings, direct.rankings)
    assert np.allclose(via_strategy.scores, direct.scores)
    assert np.array_equal(via_strategy.approvals, direct.approvals)


def test_run_simulation_defaults_to_sincere_behavior():
    electorate, candidates = make_compromise_case()
    systems = [Plurality(), ScoreVoting()]

    default_metrics = run_simulation(electorate, candidates, systems)
    sincere_metrics = run_simulation(
        electorate,
        candidates,
        systems,
        strategy=SincereStrategy(),
    )

    assert [m.system_name for m in default_metrics] == [m.system_name for m in sincere_metrics]
    assert [m.distance_to_median for m in default_metrics] == [
        m.distance_to_median for m in sincere_metrics
    ]


def test_plurality_compromise_strategy_can_change_plurality_winner():
    electorate, candidates = make_compromise_case()
    sincere_ballots = BallotProfile.from_preferences(electorate, candidates)
    strategy = PluralityCompromiseStrategy(
        compromise_rate=1.0,
        viability_threshold=0.30,
        frontrunner_count=2,
        rng=np.random.default_rng(42),
    )
    strategic_ballots = strategy.generate_ballots(electorate, candidates)

    sincere_counts = np.bincount(sincere_ballots.plurality, minlength=candidates.n_candidates)
    strategic_counts = np.bincount(strategic_ballots.plurality, minlength=candidates.n_candidates)

    assert sincere_counts.tolist() == [40, 20, 40]
    assert strategic_counts.tolist() == [60, 0, 40]

    result = Plurality().run(strategic_ballots, candidates)
    assert result.winner_indices[0] == 0


def test_plurality_compromise_strategy_respects_explicit_poll_context():
    electorate, candidates = make_compromise_case()
    strategy = PluralityCompromiseStrategy(
        compromise_rate=1.0,
        viability_threshold=0.25,
        frontrunner_count=2,
        rng=np.random.default_rng(1),
    )
    context = VotingContext(poll_shares=np.array([0.25, 0.50, 0.25]))

    ballots = strategy.generate_ballots(electorate, candidates, context=context)
    counts = np.bincount(ballots.plurality, minlength=candidates.n_candidates)

    # Center is treated as viable under the polling context, so no compromise occurs.
    assert counts.tolist() == [40, 20, 40]


def test_approval_threshold_strategy_recomputes_approvals():
    electorate, candidates = make_compromise_case()
    strategy = ApprovalThresholdStrategy(utility_threshold=0.80)
    ballots = strategy.generate_ballots(electorate, candidates)

    # With a high threshold most voters approve only their nearest candidate.
    assert np.all(ballots.approvals.sum(axis=1) >= 1)
    assert ballots.approvals[:40, 0].all()
    assert ballots.approvals[:40, 1].sum() == 0


def test_score_max_min_strategy_exaggerates_scores():
    electorate, candidates = make_compromise_case()
    strategy = ScoreMaxMinStrategy(utility_threshold=0.75)
    ballots = strategy.generate_ballots(electorate, candidates)

    assert set(np.unique(ballots.scores)).issubset({0.0, 1.0})
    assert ballots.scores[0, 0] == 1.0
    assert ballots.scores[0, 2] == 0.0


def test_ranked_truncation_strategy_marks_unranked_candidates():
    electorate, candidates = make_ranked_case()
    strategy = RankedTruncationStrategy(max_ranked=2)
    ballots = strategy.generate_ballots(electorate, candidates)

    assert np.all(ballots.rankings[:, 2:] == -1)
    result = InstantRunoff().run(ballots, candidates)
    assert result.winner_indices[0] in range(candidates.n_candidates)


def test_ranked_burying_strategy_pushes_rival_to_bottom():
    electorate, candidates = make_ranked_case()
    strategy = RankedBuryingStrategy(
        bury_rate=1.0,
        viability_threshold=0.20,
        frontrunner_count=2,
        rng=np.random.default_rng(5),
    )
    context = VotingContext(frontrunner_indices=[1, 2])
    ballots = strategy.generate_ballots(electorate, candidates, context=context)

    # Voter nearest A should bury B or C; the chosen viable rival is sent to the bottom.
    assert ballots.rankings[0, -1] in {1, 2}
    assert set(ballots.rankings[0]) == {0, 1, 2, 3}


def test_turnout_strategy_marks_inactive_voters_and_systems_ignore_them():
    electorate, candidates = make_compromise_case()
    strategy = TurnoutStrategy(turnout_probability=0.0, rng=np.random.default_rng(9))
    ballots = strategy.generate_ballots(electorate, candidates)

    assert ballots.n_active_voters == 0
    result = Plurality().run(ballots, candidates)
    assert result.winner_indices[0] == 0


def test_turnout_strategy_can_abstain_when_favorite_is_nonviable():
    electorate, candidates = make_compromise_case()
    strategy = TurnoutStrategy(
        turnout_probability=1.0,
        abstain_if_favorite_nonviable=True,
        abstain_probability_when_nonviable=1.0,
        viability_threshold=0.30,
        frontrunner_count=2,
        rng=np.random.default_rng(11),
    )
    ballots = strategy.generate_ballots(electorate, candidates)

    # Center supporters abstain because Center is not viable under the inferred polls.
    assert ballots.n_active_voters == 80
