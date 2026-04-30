# Notebook Guide

The notebooks in this repository are meant to serve a few different purposes. Some are practical entry points for using the package. Some explain classic ideas from social choice theory in a more visual, simulation-based way. Others are more exploratory and use the simulator to ask open-ended political questions.

## Getting started with the package

These notebooks are the best place to start for someone who wants to learn how to use `electoral_sim` and see the built-in scenarios and systems in action.

| Notebook | Purpose |
|---|---|
| `quickstart.ipynb` | Minimal introduction to loading scenarios, running systems, and reading the core metrics |
| `electoral_systems_comparison.ipynb` | Main comparison notebook for the built-in scenarios and paper-style figures |

## Classic ideas from social choice theory

These notebooks use the spatial simulator to explain well-known ideas from voting theory in a more concrete and visual way.

| Notebook | Purpose |
|---|---|
| `arrows_impossibility_theorem.ipynb` | Explains Arrow's impossibility theorem using spatial examples and simple criterion checks |
| `electoral_dynamics_theory_companion.ipynb` | Follows repeated-election geometric quantities such as winner centrality, supporter alignment, and coverage, with simulation evidence that matches the theory-facing setup |
| `median_voter_theorem_higher_dimensions.ipynb` | Introduces the median voter theorem in one dimension and shows how it weakens as competition moves into higher dimensions |
| `vote_splitting_and_spoiler_effect.ipynb` | Explores spoiler dynamics and vote splitting across electoral systems |

## Experimental political simulations

These notebooks are less about re-deriving textbook results and more about using the simulator to study practical or still-contested questions.

| Notebook | Purpose |
|---|---|
| `plurality_vs_irv_strategic_comparison.ipynb` | Compares closed-primary plurality against alternative primary and general-election pipelines under different strategy assumptions |
| `fractional_reporting_robustness.ipynb` | Examines how the fractional system behaves under honest, noisy, biased, and adversarial spatial reporting, with plurality as a reference point |
| `majority_vs_minority_welfare.ipynb` | Studies how parties, primaries, turnout, and strategy shape overall, majority, and minority welfare across a broad sampled design |
| `representation_vs_policy_consequences.ipynb` | Shows how similar-looking electoral outcomes can imply very different policy consequences once group-sensitive and threshold-sensitive utility models are made explicit |
| `electoral_systems_polarization_dynamics.ipynb` | Studies how different electoral systems interact with stylized voter and candidate polarization mechanisms over repeated elections, including oracle comparisons between winner-centering and depolarization |
| `primaries_moderation_backfire.ipynb` | Studies when primaries pull nominees toward the electorate's center and when they instead make outcomes less moderate |

## Notes

- Some notebooks define lightweight helper code in `notebooks/helpers` to keep the notebook narrative readable.
