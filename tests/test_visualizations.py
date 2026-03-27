import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from electoral_sim.ballots import BallotProfile
from electoral_sim.candidates import CandidateSet, fixed_candidates
from electoral_sim.electorate import Electorate
from electoral_sim.primaries import PartySpec, PrimaryType, run_two_party_primary
from electoral_sim.systems import Plurality
from electoral_sim.utils.viz_electorate import (
    plot_all_systems_spatial,
    plot_election_result,
    plot_electorate,
)
from electoral_sim.utils.viz_primaries import plot_primary_spatial


def _make_2d_case():
    rng = np.random.default_rng(42)
    preferences = rng.uniform(0.0, 1.0, size=(200, 2))
    electorate = Electorate(
        preferences,
        dim_names=["economic (left-right)", "social (libertarian-authoritarian)"],
    )
    candidates = fixed_candidates(
        [[0.2, 0.3], [0.5, 0.5], [0.8, 0.7]],
        ["A", "B", "C"],
    )
    return electorate, candidates


def _make_3d_case():
    rng = np.random.default_rng(7)
    left = rng.multivariate_normal(
        [0.25, 0.35, 0.45],
        np.diag([0.015, 0.012, 0.010]),
        size=220,
    )
    right = rng.multivariate_normal(
        [0.75, 0.65, 0.55],
        np.diag([0.015, 0.012, 0.010]),
        size=220,
    )
    electorate = Electorate(
        np.clip(np.vstack([left, right]), 0.0, 1.0),
        dim_names=["econ", "social", "climate"],
    )
    candidates = CandidateSet(
        np.array(
            [
                [0.15, 0.30, 0.35],
                [0.35, 0.40, 0.55],
                [0.65, 0.60, 0.45],
                [0.85, 0.70, 0.65],
            ]
        ),
        ["L1", "L2", "R1", "R2"],
    )
    return electorate, candidates


def test_plot_electorate_keeps_native_2d_axes():
    electorate, candidates = _make_2d_case()

    fig = plot_electorate(electorate, candidates)
    ax = fig.axes[0]

    assert ax.get_xlabel() == electorate.dim_names[0]
    assert ax.get_ylabel() == electorate.dim_names[1]
    assert ax.get_xlim() == pytest.approx((0.0, 1.0))
    assert ax.get_ylim() == pytest.approx((0.0, 1.0))

    plt.close(fig)


def test_higher_dimensional_spatial_plots_use_pca_projection():
    electorate, candidates = _make_3d_case()
    ballots = BallotProfile.from_preferences(electorate, candidates)
    result = Plurality().run(ballots, candidates)

    fig = plot_electorate(electorate, candidates)
    ax = fig.axes[0]
    assert ax.get_xlabel().startswith("PC1")
    assert ax.get_ylabel().startswith("PC2")
    plt.close(fig)

    fig = plot_election_result(electorate, candidates, result)
    ax = fig.axes[0]
    assert ax.get_xlabel().startswith("PC1")
    plt.close(fig)

    fig = plot_all_systems_spatial(electorate, candidates, [result])
    assert fig.axes[0].get_xlabel().startswith("PC1")
    plt.close(fig)

    parties = [
        PartySpec("Left Party", [0, 1], Plurality()),
        PartySpec("Right Party", [2, 3], Plurality()),
    ]
    primary_result = run_two_party_primary(
        electorate,
        candidates,
        parties,
        general_system=Plurality(),
        primary_type=PrimaryType.CLOSED,
    )
    fig = plot_primary_spatial(electorate, candidates, parties, primary_result)
    assert fig.axes[0].get_xlabel().startswith("PC1")
    plt.close(fig)
