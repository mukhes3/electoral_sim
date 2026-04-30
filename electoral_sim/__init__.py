"""Electoral Systems Simulator."""

__version__ = "0.2.0"

from electoral_sim.policy import (  # noqa: F401
    PolicyConsequenceSpec,
    PolicyDefinition,
    PolicyFeedbackSpec,
    PolicyOutcome,
    PolicyThresholdEffect,
    PolicyUtilityComponents,
    apply_policy_feedback,
    extract_policy_outcome,
    policy_utility_components,
)
