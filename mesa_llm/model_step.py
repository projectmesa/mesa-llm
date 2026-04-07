"""Compatibility helpers for Mesa model step/time access."""

from __future__ import annotations

from typing import Any


def get_model_step(model: Any) -> int:
    """Return the current simulation step with Mesa 3.x/4.x compatibility.

    Mesa 4.x removed ``model.steps`` in favor of ``model.time``. This helper
    normalizes both APIs and returns an ``int`` so existing plan/memory records
    remain stable.

    Example:
        >>> class LegacyModel:
        ...     steps = 3
        >>> get_model_step(LegacyModel())
        3
        >>> class Mesa4Model:
        ...     time = 4.0
        >>> get_model_step(Mesa4Model())
        4
    """
    steps_value = getattr(model, "steps", None)
    if isinstance(steps_value, int | float):
        return int(steps_value)

    time_value = getattr(model, "time", None)
    if isinstance(time_value, int | float):
        return int(time_value)

    return 0
