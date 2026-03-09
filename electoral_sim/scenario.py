"""Scenario loading: parse YAML/dict configs into (Electorate, CandidateSet) pairs."""
from __future__ import annotations
from pathlib import Path
import yaml
import numpy as np

from electoral_sim.electorate import Electorate, from_config as electorate_from_config
from electoral_sim.candidates import CandidateSet, from_config as candidates_from_config


def load_scenario(path, rng=None):
    path = Path(path)
    with open(path) as f:
        config = yaml.safe_load(f)
    rng = rng or np.random.default_rng()
    electorate = electorate_from_config(config, rng=rng)
    candidates = candidates_from_config(config)
    return config, electorate, candidates


def load_all_scenarios(scenarios_dir, rng=None):
    scenarios_dir = Path(scenarios_dir)
    rng = rng or np.random.default_rng()
    scenarios = []
    for path in sorted(scenarios_dir.glob("*.yaml")):
        scenarios.append(load_scenario(path, rng=rng))
    return scenarios
