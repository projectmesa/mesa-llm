"""
Automatic parallel stepping for Mesa-LLM simulations.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from mesa.agent import Agent, AgentSet

if TYPE_CHECKING:
    from .llm_agent import LLMAgent

logger = logging.getLogger(__name__)

# Global variable to control parallel stepping mode
_PARALLEL_STEPPING_MODE = "asyncio"  # or "threading"
_PARALLEL_ON_ERROR = "continue"


@dataclass
class AgentStepResult:
    """Result of stepping one agent in a parallel execution batch."""

    agent: Agent | LLMAgent
    success: bool
    exception: Exception | None = None


def _validate_on_error(on_error: str) -> None:
    """Validate error handling mode for parallel stepping APIs."""
    if on_error not in ("continue", "raise"):
        raise ValueError("on_error must be either 'continue' or 'raise'")


async def step_agents_parallel(
    agents: list[Agent | LLMAgent],
    *,
    on_error: str | None = None,
) -> list[AgentStepResult]:
    """Step all agents in parallel using async/await.

    Args:
        agents: Agents to step.
        on_error: ``continue`` to isolate failures, ``raise`` to fail fast.

    Returns:
        Per-agent step outcomes.
    """
    effective_on_error = on_error or _PARALLEL_ON_ERROR
    _validate_on_error(effective_on_error)

    tasks = []
    for agent in agents:
        if hasattr(agent, "astep"):
            tasks.append(agent.astep())
        elif hasattr(agent, "step"):
            tasks.append(_sync_step(agent))

    gathered = await asyncio.gather(*tasks, return_exceptions=True)
    results: list[AgentStepResult] = []
    for agent, result in zip(agents, gathered, strict=False):
        if isinstance(result, Exception):
            logger.exception(
                "Parallel step failed for agent %s", agent, exc_info=result
            )
            results.append(
                AgentStepResult(agent=agent, success=False, exception=result)
            )
            if effective_on_error == "raise":
                raise result
        else:
            results.append(AgentStepResult(agent=agent, success=True, exception=None))
    return results


async def _sync_step(agent: Agent) -> None:
    """Run synchronous step in async context."""
    agent.step()


def step_agents_multithreaded(
    agents: list[Agent | LLMAgent],
    *,
    on_error: str | None = None,
) -> list[AgentStepResult]:
    """Step all agents in parallel using threads with per-agent isolation."""
    effective_on_error = on_error or _PARALLEL_ON_ERROR
    _validate_on_error(effective_on_error)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures: list[tuple[Agent | LLMAgent, concurrent.futures.Future]] = []
        for agent in agents:
            if hasattr(agent, "astep"):
                # run async steps in the event loop in a thread
                futures.append(
                    (
                        agent,
                        executor.submit(lambda agent=agent: asyncio.run(agent.astep())),
                    )
                )
            elif hasattr(agent, "step"):
                futures.append((agent, executor.submit(agent.step)))

        results: list[AgentStepResult] = []
        for agent, future in futures:
            try:
                future.result()
                results.append(
                    AgentStepResult(agent=agent, success=True, exception=None)
                )
            except Exception as error:
                logger.exception(
                    "Multithreaded parallel step failed for agent %s",
                    agent,
                    exc_info=error,
                )
                results.append(
                    AgentStepResult(agent=agent, success=False, exception=error)
                )
                if effective_on_error == "raise":
                    raise
        return results


def step_agents_parallel_sync(
    agents: list[Agent | LLMAgent],
    *,
    on_error: str | None = None,
) -> list[AgentStepResult]:
    """Synchronous wrapper for parallel stepping using the global mode."""
    effective_on_error = on_error or _PARALLEL_ON_ERROR
    _validate_on_error(effective_on_error)

    if _PARALLEL_STEPPING_MODE == "asyncio":
        try:
            asyncio.get_running_loop()
            # If in event loop, use thread
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    lambda: asyncio.run(
                        step_agents_parallel(agents, on_error=effective_on_error)
                    )
                )
                return future.result()
        except RuntimeError:
            # No event loop - create one
            return asyncio.run(
                step_agents_parallel(agents, on_error=effective_on_error)
            )
    elif _PARALLEL_STEPPING_MODE == "threading":
        return step_agents_multithreaded(agents, on_error=effective_on_error)
    else:
        raise ValueError(f"Unknown parallel stepping mode: {_PARALLEL_STEPPING_MODE}")


# Patch Mesa's shuffle_do for automatic parallel detection
_original_shuffle_do = AgentSet.shuffle_do


def _enhanced_shuffle_do(self, method: str, *args, **kwargs):
    """Enhanced shuffle_do with automatic parallel stepping."""
    if method == "step" and self:
        agent = next(iter(self))
        if hasattr(agent, "model") and getattr(agent.model, "parallel_stepping", False):
            step_agents_parallel_sync(list(self))
            return
    _original_shuffle_do(self, method, *args, **kwargs)


def enable_automatic_parallel_stepping(
    mode: str = "asyncio", on_error: str = "continue"
):
    """Enable automatic parallel stepping with selectable mode ('asyncio' or 'threading')."""
    global _PARALLEL_STEPPING_MODE, _PARALLEL_ON_ERROR  # noqa: PLW0603
    if mode not in ("asyncio", "threading"):
        raise ValueError("mode must be either 'asyncio' or 'threading'")
    _validate_on_error(on_error)
    _PARALLEL_STEPPING_MODE = mode
    _PARALLEL_ON_ERROR = on_error
    AgentSet.shuffle_do = _enhanced_shuffle_do


def disable_automatic_parallel_stepping():
    """Restore original shuffle_do behavior."""
    AgentSet.shuffle_do = _original_shuffle_do


# --- Monkey-patch AgentSet with do_async for async parallel method calls ---


def _agentset_do_async(self, method: str, *args, **kwargs):
    """
    Call the given async method on all agents in the set in parallel.
    Usage: await agents.do_async("async_function")
    """
    logger.info("Running async method '%s' on %d agents", method, len(self))

    async def _run():
        tasks = []
        agents = []
        for agent in self:
            fn = getattr(agent, method, None)
            if fn is not None and asyncio.iscoroutinefunction(fn):
                agents.append(agent)
                tasks.append(fn(*args, **kwargs))
            else:
                raise AttributeError(
                    f"Agent {agent} does not have async method '{method}'"
                )

        gathered = await asyncio.gather(*tasks, return_exceptions=True)
        for agent, result in zip(agents, gathered, strict=False):
            if isinstance(result, Exception):
                logger.exception(
                    "AgentSet.do_async failed for agent %s method '%s'",
                    agent,
                    method,
                    exc_info=result,
                )
                if _PARALLEL_ON_ERROR == "raise":
                    raise result
        return gathered

    return _run()


AgentSet.do_async = _agentset_do_async
