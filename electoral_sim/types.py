"""Core data types shared across the simulator."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class ElectionResult:
    """
    The outcome of running an electoral system.

    For winner-take-all systems:
        outcome_position  = the winning candidate's position vector
        centroid_position = same (identical to outcome_position)
        median_legislator_position = same (identical to outcome_position)

    For PR systems:
        centroid_position          = seat-share-weighted mean of elected parties.
                                     Approximates the voter mean by construction;
                                     included for reference but NOT used as the
                                     primary outcome metric unless explicitly selected.
        median_legislator_position = position of the legislator at the 50th
                                     percentile of the cumulative seat-share
                                     distribution (sorted by first dimension).
                                     Preserved as the default backward-compatible
                                     PR outcome summary.
        outcome_position           = the configured PR outcome position used for
                                     all distance calculations. By default this
                                     equals median_legislator_position, but
                                     multidimensional PR systems may choose a
                                     different outcome rule.
    """
    outcome_position: np.ndarray           # shape (n_dims,) — primary policy outcome
    centroid_position: np.ndarray          # shape (n_dims,) — seat-share centroid (PR only meaningful)
    median_legislator_position: np.ndarray # shape (n_dims,) — median legislator (PR only meaningful)
    winner_indices: list[int]              # candidate indices (1 for WTA, multiple for PR)
    seat_shares: dict[int, float]          # candidate_index -> fraction of seats/weight
    elimination_order: list[int]           # for ranked systems; empty otherwise
    system_name: str
    is_pr: bool = False                    # True for PR and MMP systems
    metadata: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        winners = {k: round(v, 3) for k, v in self.seat_shares.items()}
        return (
            f"ElectionResult(system={self.system_name}, "
            f"outcome={np.round(self.outcome_position, 3)}, "
            f"seat_shares={winners})"
        )
