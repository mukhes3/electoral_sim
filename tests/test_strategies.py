import numpy as np

from electoral_sim.ballots import BallotProfile
from electoral_sim.candidates import fixed_candidates
from electoral_sim.electorate import Electorate
from electoral_sim.metrics import run_simulation
from electoral_sim.strategies import (
    PluralityCompromiseStrategy,
    SincereStrategy,
    VotingContext,
)
from electoral_sim.systems import Plurality, ScoreVoting


def make_compromise_case():
    left = np.tile([0.20], (40, 1))
    center = np.tile([0.49], (20, 1))
    right = np.tile([0.80], (40, 1))
    electorate = Electorate(np.vstack([left, center, right]), dim_names=["economic"])
    candidates = fixed_candidates([[0.20], [0.49], [0.80]], ["Left", "Center", "Right"])
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
