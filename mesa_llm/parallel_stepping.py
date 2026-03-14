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

_PARALLEL_ON_ERROR = "continue"  # "continue" | "raise"
_PARALLEL_TIMEOUT: float | None = None


@dataclass
class AgentStepResult:
    """Result of a single agent's step during parallel execution."""

    agent: Agent | LLMAgent
    success: bool
    exception: BaseException | None = None


async def step_agents_parallel(
    agents: list[Agent | LLMAgent],
    *,
    on_error: str | None = None,
    timeout: float | None = None,
) -> list[AgentStepResult]:
    """Step all agents in parallel using async/await with error isolation."""
    effective_on_error = on_error if on_error is not None else _PARALLEL_ON_ERROR
    effective_timeout = timeout if timeout is not None else _PARALLEL_TIMEOUT

    coros = []
    agent_list = []
    for agent in agents:
        if hasattr(agent, "astep"):
            coro = agent.astep()
        elif hasattr(agent, "step"):
            coro = _sync_step(agent)
        else:
            continue
        if effective_timeout is not None:
            coro = asyncio.wait_for(coro, timeout=effective_timeout)
        coros.append(coro)
        agent_list.append(agent)

    outcomes = await asyncio.gather(*coros, return_exceptions=True)

    results = []
    for agent, outcome in zip(agent_list, outcomes):
        if isinstance(outcome, BaseException):
            logger.error(
                "Agent %s (id=%s) failed during parallel step: %s",
                agent.__class__.__name__,
                getattr(agent, "unique_id", "?"),
                outcome,
            )
            results.append(
                AgentStepResult(agent=agent, success=False, exception=outcome)
            )
            if effective_on_error == "raise":
                raise outcome
        else:
            results.append(AgentStepResult(agent=agent, success=True))

    return results


async def _sync_step(agent: Agent) -> None:
    """Run synchronous step in async context."""
    agent.step()


def step_agents_multithreaded(
    agents: list[Agent | LLMAgent],
    *,
    on_error: str | None = None,
    timeout: float | None = None,
) -> list[AgentStepResult]:
    """Step all agents in parallel using threads with error isolation."""
    effective_on_error = on_error if on_error is not None else _PARALLEL_ON_ERROR
    effective_timeout = timeout if timeout is not None else _PARALLEL_TIMEOUT

    results = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        agent_futures = []
        for agent in agents:
            if hasattr(agent, "astep"):
                future = executor.submit(lambda agent=agent: asyncio.run(agent.astep()))
            elif hasattr(agent, "step"):
                future = executor.submit(agent.step)
            else:
                continue
            agent_futures.append((agent, future))

        for agent, future in agent_futures:
            try:
                future.result(timeout=effective_timeout)
                results.append(AgentStepResult(agent=agent, success=True))
            except Exception as exc:
                logger.error(
                    "Agent %s (id=%s) failed during threaded step: %s",
                    agent.__class__.__name__,
                    getattr(agent, "unique_id", "?"),
                    exc,
                )
                results.append(
                    AgentStepResult(agent=agent, success=False, exception=exc)
                )
                if effective_on_error == "raise":
                    raise
    return results


def step_agents_parallel_sync(
    agents: list[Agent | LLMAgent],
    *,
    on_error: str | None = None,
    timeout: float | None = None,
) -> list[AgentStepResult]:
    """Synchronous wrapper for parallel stepping using the global mode."""
    if _PARALLEL_STEPPING_MODE == "asyncio":
        try:
            asyncio.get_running_loop()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    lambda: asyncio.run(
                        step_agents_parallel(agents, on_error=on_error, timeout=timeout)
                    )
                )
                return future.result()
        except RuntimeError:
            return asyncio.run(
                step_agents_parallel(agents, on_error=on_error, timeout=timeout)
            )
    elif _PARALLEL_STEPPING_MODE == "threading":
        return step_agents_multithreaded(agents, on_error=on_error, timeout=timeout)
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
    mode: str = "asyncio",
    *,
    on_error: str = "continue",
    timeout: float | None = None,
):
    """Enable automatic parallel stepping with selectable mode and error handling."""
    global _PARALLEL_STEPPING_MODE, _PARALLEL_ON_ERROR, _PARALLEL_TIMEOUT  # noqa: PLW0603
    if mode not in ("asyncio", "threading"):
        raise ValueError("mode must be either 'asyncio' or 'threading'")
    if on_error not in ("continue", "raise"):
        raise ValueError("on_error must be either 'continue' or 'raise'")
    _PARALLEL_STEPPING_MODE = mode
    _PARALLEL_ON_ERROR = on_error
    _PARALLEL_TIMEOUT = timeout
    AgentSet.shuffle_do = _enhanced_shuffle_do


def disable_automatic_parallel_stepping():
    """Restore original shuffle_do behavior and reset config."""
    global _PARALLEL_ON_ERROR, _PARALLEL_TIMEOUT  # noqa: PLW0603
    AgentSet.shuffle_do = _original_shuffle_do
    _PARALLEL_ON_ERROR = "continue"
    _PARALLEL_TIMEOUT = None


# --- Monkey-patch AgentSet with do_async for async parallel method calls ---


def _agentset_do_async(self, method: str, *args, **kwargs):
    """
    Call the given async method on all agents in the set in parallel.
    Usage: await agents.do_async("async_function")
    """
    logger.info("Running async method '%s' on %d agents", method, len(self))

    async def _run():
        tasks = []
        for agent in self:
            fn = getattr(agent, method, None)
            if fn is not None and asyncio.iscoroutinefunction(fn):
                tasks.append(fn(*args, **kwargs))
            else:
                raise AttributeError(
                    f"Agent {agent} does not have async method '{method}'"
                )
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for agent, result in zip(self, results):
            if isinstance(result, BaseException):
                logger.error(
                    "Agent %s (id=%s) failed during do_async('%s'): %s",
                    agent.__class__.__name__,
                    getattr(agent, "unique_id", "?"),
                    method,
                    result,
                )
        return results

    return _run()


AgentSet.do_async = _agentset_do_async
