import numpy as np

from electoral_sim.candidates import fixed_candidates
from electoral_sim.electorate import Electorate
from electoral_sim.parties import (
    PartySet,
    assign_candidates_to_parties,
    assign_voters_to_parties,
    fixed_parties,
    membership_masks_from_indices,
)
from electoral_sim.primaries import (
    PrimaryType,
    assign_party_membership,
    build_party_specs_from_positions,
    run_two_party_primary,
)
from electoral_sim.systems import Plurality


def make_party_case():
    preferences = np.vstack(
        [
            np.tile([0.12, 0.60], (30, 1)),
            np.tile([0.30, 0.52], (20, 1)),
            np.tile([0.70, 0.48], (20, 1)),
            np.tile([0.88, 0.40], (30, 1)),
        ]
    )
    electorate = Electorate(preferences, dim_names=["economic", "social"])
    candidates = fixed_candidates(
        [
            [0.10, 0.62],
            [0.28, 0.54],
            [0.72, 0.46],
            [0.90, 0.38],
        ],
        ["L-Base", "L-Moderate", "R-Moderate", "R-Base"],
    )
    parties = fixed_parties(
        [
            [0.24, 0.55],
            [0.76, 0.45],
        ],
        ["Left Party", "Right Party"],
    )
    return electorate, candidates, parties


def test_party_set_basic_properties():
    parties = PartySet(np.array([[0.2], [0.8]]), ["Left", "Right"])
    assert parties.n_parties == 2
    assert parties.n_dims == 1


def test_nearest_party_assignment_for_voters_and_candidates():
    electorate, candidates, parties = make_party_case()

    voter_indices = assign_voters_to_parties(electorate, parties)
    candidate_indices = assign_candidates_to_parties(candidates, parties)

    assert voter_indices.shape == (electorate.n_voters,)
    assert candidate_indices.tolist() == [0, 0, 1, 1]


def test_membership_masks_from_indices_round_trips_labels():
    indices = np.array([0, 1, 1, 0, 1])
    masks = membership_masks_from_indices(indices, ["Left", "Right"])
    assert masks["Left"].tolist() == [True, False, False, True, False]
    assert masks["Right"].tolist() == [False, True, True, False, True]


def test_build_party_specs_from_positions_assigns_candidate_slates():
    _, candidates, parties = make_party_case()
    specs = build_party_specs_from_positions(candidates, parties)

    assert [spec.name for spec in specs] == ["Left Party", "Right Party"]
    assert [spec.candidate_indices for spec in specs] == [[0, 1], [2, 3]]


def test_assign_party_membership_accepts_party_positions():
    electorate, candidates, parties = make_party_case()
    specs = build_party_specs_from_positions(candidates, parties)

    memberships = assign_party_membership(
        electorate,
        candidates,
        specs,
        primary_type=PrimaryType.CLOSED,
        party_positions=parties,
    )

    assert int(memberships["Left Party"].sum()) == 50
    assert int(memberships["Right Party"].sum()) == 50


def test_run_two_party_primary_works_with_party_position_workflow():
    electorate, candidates, parties = make_party_case()
    specs = build_party_specs_from_positions(candidates, parties)

    memberships = assign_party_membership(
        electorate,
        candidates,
        specs,
        primary_type=PrimaryType.CLOSED,
        party_positions=parties,
    )

    result = run_two_party_primary(
        electorate,
        candidates,
        specs,
        general_system=Plurality(),
        primary_type=PrimaryType.CLOSED,
        memberships=memberships,
    )

    assert [pr.nominee_index for pr in result.primary_results] == [0, 3]
    assert result.general_result.winner_indices[0] in {0, 1}
