"""Scenario loading: parse YAML/dict configs into (Electorate, CandidateSet) pairs."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from electoral_sim.electorate import Electorate, from_config as electorate_from_config
from electoral_sim.candidates import CandidateSet, from_config as candidates_from_config

PACKAGE_ROOT = Path(__file__).resolve().parent
BUILTIN_SCENARIOS_DIR = PACKAGE_ROOT / "data" / "scenarios"
REPO_SCENARIOS_DIR = PACKAGE_ROOT.parent / "configs" / "scenarios"


def built_in_scenarios_dir() -> Path:
    """Return the packaged built-in scenarios directory."""
    return BUILTIN_SCENARIOS_DIR


def built_in_scenario_paths() -> list[Path]:
    """Return all packaged built-in scenario YAML files."""
    return sorted(BUILTIN_SCENARIOS_DIR.glob("*.yaml"))


def resolve_scenario_path(path: str | Path) -> Path:
    """
    Resolve a scenario path with backward-compatible fallbacks.

    Resolution order:
    1. exact filesystem path provided by the caller
    2. packaged built-in scenarios by filename
    3. repo-local ``configs/scenarios`` by filename
    """
    candidate = Path(path)
    if candidate.exists():
        return candidate

    possible_names = []
    raw = str(candidate)
    if candidate.suffix:
        possible_names.append(candidate.name)
        possible_names.append(raw)
    else:
        possible_names.append(f"{raw}.yaml")
        possible_names.append(candidate.name)

    seen = set()
    for name in possible_names:
        if name in seen:
            continue
        seen.add(name)

        packaged = BUILTIN_SCENARIOS_DIR / name
        if packaged.exists():
            return packaged

        repo_local = REPO_SCENARIOS_DIR / name
        if repo_local.exists():
            return repo_local

    raise FileNotFoundError(f"Could not resolve scenario path: {path}")


def load_scenario(path, rng=None):
    path = resolve_scenario_path(path)
    with open(path) as f:
        config = yaml.safe_load(f)
    rng = rng or np.random.default_rng()
    electorate = electorate_from_config(config, rng=rng)
    candidates = candidates_from_config(config)
    return config, electorate, candidates


def load_all_scenarios(scenarios_dir=None, rng=None):
    scenarios_dir = built_in_scenarios_dir() if scenarios_dir is None else Path(scenarios_dir)
    rng = rng or np.random.default_rng()
    scenarios = []
    for path in sorted(scenarios_dir.glob("*.yaml")):
        scenarios.append(load_scenario(path, rng=rng))
    return scenarios
