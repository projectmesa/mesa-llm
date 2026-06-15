"""Explicit tool-set factories for Mesa-LLM capability configuration."""

from collections.abc import Callable

from mesa_llm.tools.inbuilt_tools import (
    move_one_step,
    speak_to,
    teleport_to_location,
)


def default_tools() -> tuple[Callable, ...]:
    """Return the recommended default read-only tools."""
    return ()


def math_tools() -> tuple[Callable, ...]:
    """Return math/calculation tools."""
    return ()


def spatial_tools() -> tuple[Callable, ...]:
    """Return read-only spatial query tools."""
    return ()


def environment_tools() -> tuple[Callable, ...]:
    """Return read-only environment/context tools."""
    return ()


def social_query_tools() -> tuple[Callable, ...]:
    """Return read-only social-context query tools."""
    return ()


def external_tools() -> tuple[Callable, ...]:
    """Return opt-in external tools."""
    return ()


def legacy_tools() -> tuple[Callable, Callable, Callable]:
    """Return the compatibility tools for old implicit built-in behavior."""
    return (move_one_step, teleport_to_location, speak_to)


__all__ = [
    "default_tools",
    "environment_tools",
    "external_tools",
    "legacy_tools",
    "math_tools",
    "social_query_tools",
    "spatial_tools",
]
