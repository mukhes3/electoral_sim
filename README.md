# electoral-sim

A spatial voting simulator for comparing electoral systems under real-world-grounded electorate patterns.

Voter and candidate preferences are represented as vectors in `[0,1]^N`. The primary metric the simulator uses to judge how close each electoral system's outcome is to the **geometric median** of voter preferences — the central tendency most robust to outliers, although there are other metrics also available to use. 

The simulation framework as well as some experimental results can be found in the [Arxiv Preprint](https://arxiv.org/abs/2603.08752).


---

## Implemented Electoral Systems

| System | Family |
|---|---|
| Plurality (FPTP) | Winner-take-all |
| Two-Round Runoff | Winner-take-all |
| Instant Runoff (IRV/RCV) | Ranked |
| Borda Count | Ranked/Scoring |
| Approval Voting | Approval |
| Score Voting | Scoring |
| Condorcet (Schulze) | Pairwise |
| Party-List PR (D'Hondt) | Proportional |
| Mixed Member Proportional (MMP) | Hybrid |
| Fractional Ballot (σ = 0.1, 0.3, 1.0) | Hypothetical benchmark |

The **Fractional Ballot** is a hypothetical system included as a theoretical benchmark. It is not currently implemented in any jurisdiction. Each voter's influence is distributed across candidates via a Boltzmann softmax over preference distances; see the paper for details.

## Scenarios

Thirteen real-world-grounded electorate archetypes, including a higher-dimensional primary case:

| # | Name | Real-world analogue |
|---|---|---|
| 1 | Unimodal Consensus | Nordic / consensus democracies |
| 2 | Polarized Bimodal | Contemporary USA, Brexit-era UK |
| 3 | Multimodal Fragmented | Netherlands, Israel, Italy |
| 4 | Dominant Party with Minorities | Japan (LDP), Hungary |
| 5 | Asymmetric Skewed | Latin American / post-Soviet patterns |
| 6 | Two-Party Symmetric Polarized | Stylised US two-party system |
| 7 | Two-Party Asymmetric Centrist Majority | Many European two-bloc systems |
| 8 | Two-Party Dominant Left | Dominant-party with formal primary process |
| 9 | Two-Party High Overlap | Low-polarization or catch-all party systems |
| 10 | Two-Party Diagonal Cross-Pressures | Coalitions split across economic and cultural axes |
| 11 | Factionalized Majority Party | Big-tent dominant parties with contested nominations |
| 12 | Three-Dimensional Primary Competition | Electorates split by economics, culture, and trust in institutions |
| 13 | Insurgent Outlier Candidate | Primary electorates with a highly energized insurgent wing |

## Installation

Install from PyPI:

```bash
pip install electoral-simulator
```

For local development:

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
import numpy as np
from electoral_sim.scenario import load_scenario
from electoral_sim.systems import get_all_systems
from electoral_sim.metrics import run_simulation

rng = np.random.default_rng(42)
config, electorate, candidates = load_scenario(
    "configs/scenarios/02_polarized_bimodal.yaml", rng=rng
)

systems = get_all_systems(rng=rng)
metrics = run_simulation(electorate, candidates, systems)

for m in sorted(metrics, key=lambda x: x.distance_to_median):
    print(f"{m.system_name:35s}  d_median={m.distance_to_median:.4f}"
          f"  majority_sat={m.majority_satisfaction:.3f}")
```

## Notebooks

The repository includes a growing notebook collection covering quickstarts,
social choice theory, and simulation-driven political questions. A guided index
is available in `notebooks/README.md`, including experimental notebooks on
topics like strategic voting, fractional-ballot robustness, primary dynamics,
and majority-versus-minority welfare.


## CLI Runner

The package also ships with a lightweight command-line interface for running
scenarios without opening a notebook.

List the built-in system keys:

```bash
electoral-sim list-systems
```

List available scenario files:

```bash
electoral-sim list-scenarios
```

Run a scenario with the default suite of implemented systems:

```bash
electoral-sim run configs/scenarios/02_polarized_bimodal.yaml --seed 42
```

Run a smaller comparison with selected systems only:

```bash
electoral-sim run configs/scenarios/02_polarized_bimodal.yaml \
  --system plurality,score,irv \
  --seed 42 \
  --sort-by distance_to_median
```

Include the Fractional Ballot benchmark systems:

```bash
electoral-sim run configs/scenarios/02_polarized_bimodal.yaml \
  --include-fractional \
  --fractional-variant both \
  --fractional-sigmas 0.1,0.3,1.0
```

Machine-readable output is available in both JSON and CSV:

```bash
electoral-sim run configs/scenarios/02_polarized_bimodal.yaml \
  --system plurality,score \
  --format json
```

## Adding a New System

Subclass `ElectoralSystem` and implement a single `run` method:

```python
from electoral_sim.systems import ElectoralSystem
from electoral_sim.types import ElectionResult

class MySystem(ElectoralSystem):
    name = "My System"

    def run(self, ballots, candidates) -> ElectionResult:
        # ballots.plurality_votes, ballots.rankings, ballots.scores,
        # ballots.approvals, ballots.distances are all available
        winner_idx = ...
        return self._make_result(winner_idx, candidates)
```

## Adding a New Scenario

Create a YAML file in `configs/scenarios/`:

```yaml
name: "My Scenario"
real_world_analog: "Description"
n_voters: 5000
electorate:
  type: gaussian_mixture
  components:
    - weight: 0.6
      mean: [0.65, 0.55]
      std:  [0.10, 0.08]
    - weight: 0.4
      mean: [0.30, 0.40]
      std:  [0.10, 0.08]
candidates:
  - {label: "Candidate A", position: [0.70, 0.60]}
  - {label: "Candidate B", position: [0.50, 0.48]}
  - {label: "Candidate C", position: [0.25, 0.35]}
```

If you want to track demographic or coalition-level welfare, electorate
components can also carry optional `group` labels. Multiple components may
share the same label, which is useful when one group has internal ideological
variation:

```yaml
electorate:
  type: gaussian_mixture
  components:
    - weight: 0.30
      mean: [0.20, 0.40]
      cov: [[0.01, 0.00], [0.00, 0.01]]
      group: Group A
    - weight: 0.20
      mean: [0.40, 0.60]
      cov: [[0.01, 0.00], [0.00, 0.01]]
      group: Group A
    - weight: 0.50
      mean: [0.75, 0.50]
      cov: [[0.01, 0.00], [0.00, 0.01]]
      group: Group B
```

When no `group` labels are provided, scenarios behave exactly as before.


## Voting Strategies

Ballots can be generated under different voting strategies by passing a
strategy model to `run_simulation(...)`.

The currently available strategy models are:

| Strategy | Description |
|---|---|
| `SincereStrategy` | Ballots are derived directly from preference distances |
| `PluralityCompromiseStrategy` | Voters may abandon non-viable favourites under plurality |
| `ApprovalThresholdStrategy` | Voters approve candidates above a utility threshold |
| `ScoreMaxMinStrategy` | Voters exaggerate scores to a max-min scale |
| `RankedTruncationStrategy` | Voters submit only the top portion of a ranking |
| `RankedBuryingStrategy` | Voters demote a strong rival to the bottom of the ranking |
| `TurnoutStrategy` | Some voters abstain while participants vote sincerely |

If no strategy is specified, `run_simulation(...)` uses `SincereStrategy`.

Use a strategy explicitly in the Python API:

```python
import numpy as np
from electoral_sim.scenario import load_scenario
from electoral_sim.metrics import run_simulation
from electoral_sim.systems import Plurality
from electoral_sim.strategies import PluralityCompromiseStrategy

rng = np.random.default_rng(42)
config, electorate, candidates = load_scenario(
    "configs/scenarios/02_polarized_bimodal.yaml", rng=rng
)

metrics = run_simulation(
    electorate,
    candidates,
    systems=[Plurality()],
    strategy=PluralityCompromiseStrategy(
        compromise_rate=1.0,
        viability_threshold=0.15,
        frontrunner_count=2,
        rng=rng,
    ),
)
```

Other strategies are selected in the same way:

```python
from electoral_sim.metrics import run_simulation
from electoral_sim.strategies import (
    ApprovalThresholdStrategy,
    RankedTruncationStrategy,
    ScoreMaxMinStrategy,
    TurnoutStrategy,
)
from electoral_sim.systems import ApprovalVoting, InstantRunoff, ScoreVoting

approval_metrics = run_simulation(
    electorate,
    candidates,
    systems=[ApprovalVoting()],
    strategy=ApprovalThresholdStrategy(utility_threshold=0.65),
)

score_metrics = run_simulation(
    electorate,
    candidates,
    systems=[ScoreVoting()],
    strategy=ScoreMaxMinStrategy(utility_threshold=0.7),
)

ranked_metrics = run_simulation(
    electorate,
    candidates,
    systems=[InstantRunoff()],
    strategy=RankedTruncationStrategy(max_ranked=3),
)

turnout_metrics = run_simulation(
    electorate,
    candidates,
    systems=[ApprovalVoting()],
    strategy=TurnoutStrategy(turnout_probability=0.8),
)
```

To provide explicit polling information or other strategic context, pass a
`VotingContext` alongside the strategy:

```python
import numpy as np
from electoral_sim.metrics import run_simulation
from electoral_sim.strategies import PluralityCompromiseStrategy, VotingContext
from electoral_sim.systems import Plurality

context = VotingContext(
    poll_shares=np.array([0.42, 0.18, 0.40]),
)

metrics = run_simulation(
    electorate,
    candidates,
    systems=[Plurality()],
    strategy=PluralityCompromiseStrategy(),
    context=context,
)
```

## Spatial Reporting Models

The repository also includes practical reporting models for experiments where a
system does not observe voters' true spatial positions directly.

This is mainly useful for the fractional ballot benchmark, where one may want
to compare the truthful result with outcomes produced from noisy or strategic
reported positions.

The currently available reporting models are:

| Reporting model | Description |
|---|---|
| `HonestReporting` | Uses true voter positions unchanged |
| `GaussianNoiseReporting` | Adds symmetric random noise to reported positions |
| `BiasedNoiseReporting` | Applies a systematic shift to selected voters, optionally with noise |
| `DirectionalExaggerationReporting` | Moves selected voters away from or toward chosen candidates |
| `CoalitionMisreporting` | Pulls a selected bloc toward a shared false target position |

Selectors can be used to target only part of the electorate, for example with
`axis_threshold(...)`, `within_radius(...)`, `nearest_to_candidate(...)`, or
`candidate_supporters(...)`.

Use the reporting helpers in the Python API like this:

```python
import numpy as np
from electoral_sim.fractional import FractionalBallotDiscrete
from electoral_sim.fractional_reporting import run_fractional_reporting_simulation
from electoral_sim.reporting import BiasedNoiseReporting, GaussianNoiseReporting, axis_threshold

run = run_fractional_reporting_simulation(
    electorate,
    candidates,
    system=FractionalBallotDiscrete(sigma=0.3),
    reporting_model=GaussianNoiseReporting(noise_std=0.05, rng=np.random.default_rng(42)),
)

targeted_run = run_fractional_reporting_simulation(
    electorate,
    candidates,
    system=FractionalBallotDiscrete(sigma=0.3),
    reporting_model=BiasedNoiseReporting(
        bias=np.array([0.06, 0.0]),
        selector=axis_threshold(0, 0.45, side="le"),
    ),
)

print(run.truthful_result.winner_indices, run.reported_result.winner_indices)
print(run.robustness.outcome_shift, run.robustness.winner_changed)
```

## Group Welfare Metrics

Electorates can optionally carry per-voter group labels, which makes it
possible to evaluate how different electoral systems affect different groups
through the same welfare lens used elsewhere in the package.

This support is backward compatible:

- if no group labels are present, the rest of the simulator works exactly as before
- existing overall metrics such as `distance_to_median` and `mean_voter_distance` are unchanged
- group-aware analysis is available only when you opt into labeled electorates

Use `compute_group_metrics(...)` on an election result to get per-group mean
distance to the outcome, worst-case distance, majority satisfaction, population
share, welfare, and simple disparity summaries:

```python
from electoral_sim.ballots import BallotProfile
from electoral_sim.metrics import compute_group_metrics
from electoral_sim.systems import Plurality

ballots = BallotProfile.from_preferences(electorate, candidates)
result = Plurality().run(ballots, candidates)
group_summary = compute_group_metrics(result, electorate, candidates)

for group in group_summary.groups:
    print(
        group.group_name,
        group.population_share,
        group.mean_voter_distance,
        group.majority_satisfaction,
    )

print(group_summary.max_mean_distance_gap)
print(group_summary.min_group_welfare)
```

## Running Tests

```bash
pytest tests/ -v
```

## Project Structure

```
electoral_sim/
├── cli.py          # Command-line runner
├── strategies/     # Sincere and strategic ballot-generation models
├── reporting/      # Honest, noisy, and adversarial spatial reporting models
├── electorate/     # Voter distribution generation (Gaussian, GMM, uniform)
├── candidates/     # Candidate positioning in [0,1]^N
├── ballots/        # BallotProfile: all ballot types from preference vectors
├── systems/        # Electoral system implementations, including PR variants
├── fractional.py   # Fractional Ballot (hypothetical benchmark)
├── fractional_reporting.py  # Helpers for reported-position fractional experiments
├── metrics/        # Distance metrics, Monte Carlo runner
├── primaries/      # Two-party primary pipeline
├── scenario.py     # YAML scenario loader
├── utils/          # Visualization helpers
└── types.py        # ElectionResult dataclass
configs/
└── scenarios/      # Built-in scenario definitions
notebooks/
tests/              # Unit tests for simulator, visualization, and CLI behaviour
```

## Companion repo

If you want a more user-friendly way to create scenarios, check out the companion repo [electoral_scenario_studio](https://github.com/mukhes3/electoral_scenario_studio) for natural-language authoring, iterative refinement, and visual scenario editing before running comparisons in `electoral_sim`.

## Citation

If you use this package in your research, please cite:
```bibtex
@misc{mukherjee2026electoralsystemssimulatoropen,
      title={Electoral Systems Simulator: An Open Framework for Comparing Electoral Mechanisms Across Voter Distribution Scenarios}, 
      author={Sumit Mukherjee},
      year={2026},
      eprint={2603.08752},
      archivePrefix={arXiv},
      primaryClass={cs.GT},
      url={https://arxiv.org/abs/2603.08752}, 
}
```
