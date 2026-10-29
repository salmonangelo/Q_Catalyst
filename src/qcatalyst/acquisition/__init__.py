"""Candidate acquisition and fail-safe global gate policy."""

from qcatalyst.acquisition.gate import (
    GatePolicyDecision,
    annotate_candidate_gate,
    resolve_gate0_policy,
)

__all__ = [
    "GatePolicyDecision",
    "annotate_candidate_gate",
    "resolve_gate0_policy",
]
