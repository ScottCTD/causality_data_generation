from __future__ import annotations

from enum import Enum


class Tense(str, Enum):
    """Supported verbal tenses / modes for option phrasing."""

    BASE = "base"  # Descriptive, past-ish / factual
    FUTURE = "future"  # Predictive, "will ..."
    CONDITIONAL = "conditional"  # Counterfactual, "would ..."


def parse_tense(value: str) -> Tense:
    """Convert a string value to a ``Tense`` enum, defaulting to BASE."""

    value_lower = (value or "").lower()
    if value_lower in ("base", "past"):
        return Tense.BASE
    if value_lower in ("future", "will"):
        return Tense.FUTURE
    if value_lower in ("conditional", "would"):
        return Tense.CONDITIONAL
    return Tense.BASE

