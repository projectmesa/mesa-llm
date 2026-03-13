"""
Automatic parallel stepping for Mesa-LLM simulations with performance optimizations.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import threading
from typing import TYPE_CHECKING, Dict

from mesa.agent import Agent, AgentSet

if TYPE_CHECKING:
    from .llm_agent import LLMAgent

logger = logging.getLogger(__name__)

# Global variable to control parallel stepping mode
_PARALLEL_STEPPING_MODE = "asyncio"  # or "threading"


class EventLoopManager:
    """Manages event loops for different threads."""
    
    def __init__(self):
        self.loops: Dict[int, asyncio.AbstractEventLoop] = {}
        self.lock = threading.Lock()
    
    def get_loop_for_thread(self) -> asyncio.AbstractEventLoop:
        """Get or create event loop for current thread."""
        thread_id = threading.get_ident()
        
        with self.lock:
            if thread_id not in self.loops:
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                self.loops[thread_id] = loop
            return self.loops[thread_id]
    
    def cleanup(self):
        """Cleanup all event loops."""
        with self.lock:
            for loop in self.loops.values():
                if loop.is_running():
                    loop.call_soon_threadsafe(loop.stop)
            self.loops.clear()


class SemaphorePool:
    """Manages semaphores for concurrency control."""
    
    def __init__(self, max_concurrent: int = 10):
        self.max_concurrent = max_concurrent
        self._semaphores: Dict[str, asyncio.Semaphore] = {}
        self._lock = threading.Lock()
    
    def get_semaphore(self, key: str = "default") -> asyncio.Semaphore:
        """Get or create a semaphore for concurrency control."""
        thread_id = threading.get_ident()
        semaphore_key = f"{thread_id}:{key}"
        
        with self._lock:
            if semaphore_key not in self._semaphores:
                # For Python 3.9+, loop parameter is not needed
                self._semaphores[semaphore_key] = asyncio.Semaphore(self.max_concurrent)
            return self._semaphores[semaphore_key]


# Global managers
_loop_manager = EventLoopManager()
_semaphore_pool = SemaphorePool()


async def _sync_step(agent: Agent) -> None:
    """Run synchronous step in async context."""
    agent.step()


async def step_agents_parallel(agents: list[Agent | LLMAgent]) -> None:
    """
    Optimized parallel agent stepping with proper concurrency control.
    """
    semaphore = _semaphore_pool.get_semaphore()
    
    async def step_with_semaphore(agent):
        async with semaphore:
            try:
                if hasattr(agent, "astep"):
                    await agent.astep()
                elif hasattr(agent, "step"):
                    # Run sync step in thread pool
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(None, agent.step)
            except Exception as e:
                logger.error(f"Error stepping agent {getattr(agent, 'unique_id', 'unknown')}: {e}")
    
    tasks = [step_with_semaphore(agent) for agent in agents]
    await asyncio.gather(*tasks, return_exceptions=True)


def step_agents_multithreaded(agents: list[Agent | LLMAgent], max_workers: int = None) -> None:
    """
    Optimized multithreaded agent stepping with proper resource management.
    """
    max_workers = max_workers or min(32, len(agents))

    async_agents: list[Agent | LLMAgent] = []
    sync_agents: list[Agent | LLMAgent] = []
    for agent in agents:
        if hasattr(agent, "astep"):
            async_agents.append(agent)
        elif hasattr(agent, "step"):
            sync_agents.append(agent)

    def _run_all_async() -> None:
        if not async_agents:
            return
        # Create new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(asyncio.gather(*[a.astep() for a in async_agents]))
        finally:
            loop.close()

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures: list[concurrent.futures.Future] = []

        # Run sync agents concurrently
        for agent in sync_agents:
            futures.append(executor.submit(agent.step))

        # Run all async agents on one event loop (one loop, not per-agent)
        if async_agents:
            futures.append(executor.submit(_run_all_async))

        # Wait with timeout and error handling
        for future in concurrent.futures.as_completed(futures, timeout=300):
            try:
                future.result()
            except Exception as e:
                logger.error(f"Error in multithreaded stepping: {e}")


def step_agents_parallel_sync(agents: list[Agent | LLMAgent]) -> None:
    """Synchronous wrapper for parallel stepping using the global mode."""
    if _PARALLEL_STEPPING_MODE == "asyncio":
        try:
            asyncio.get_running_loop()
            # If in event loop, use thread
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    lambda: asyncio.run(step_agents_parallel(agents))
                )
                future.result()
        except RuntimeError:
            # No event loop - create one
            asyncio.run(step_agents_parallel(agents))
    elif _PARALLEL_STEPPING_MODE == "threading":
        step_agents_multithreaded(agents)
    else:
        raise ValueError(f"Unknown parallel stepping mode: {_PARALLEL_STEPPING_MODE}")


# Patch Mesa's shuffle_do for automatic parallel detection
_original_shuffle_do = AgentSet.shuffle_do


def enable_automatic_parallel_stepping(
    mode: str = "asyncio", 
    max_concurrent: int = 10,
    request_timeout: float = 30.0
) -> None:
    """
    Enable optimized automatic parallel stepping with enhanced controls.
    
    Args:
        mode: Execution mode ('asyncio' or 'threading')
        max_concurrent: Maximum number of concurrent operations
        request_timeout: Timeout for operations in seconds
    """
    global _PARALLEL_STEPPING_MODE
    if mode not in ("asyncio", "threading"):
        raise ValueError("mode must be either 'asyncio' or 'threading'")
    
    _PARALLEL_STEPPING_MODE = mode
    
    # Update semaphore pool with new max concurrent
    global _semaphore_pool
    _semaphore_pool = SemaphorePool(max_concurrent=max_concurrent)
    
    # Enhanced shuffle_do with optimized stepping
    def _enhanced_shuffle_do_optimized(self, method: str, *args, **kwargs):
        if method == "step" and self:
            agent = next(iter(self))
            if hasattr(agent, "model") and getattr(agent.model, "parallel_stepping", False):
                if mode == "asyncio":
                    # Use optimized async stepping with proper event loop management
                    try:
                        asyncio.get_running_loop()
                        # We're in an event loop, but shuffle_do is sync. To preserve
                        # Mesa semantics (step completes before returning), run the
                        # coroutine to completion in a dedicated thread.
                        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                            future = executor.submit(lambda: asyncio.run(step_agents_parallel(list(self))))
                            future.result(timeout=request_timeout)
                    except RuntimeError:
                        # No event loop - create one and run
                        asyncio.run(step_agents_parallel(list(self)))
                    except Exception as e:
                        logger.error(f"Optimized parallel stepping failed: {e}")
                        # Fallback to original method
                        _original_shuffle_do(self, method, *args, **kwargs)
                    return
                elif mode == "threading":
                    step_agents_multithreaded(list(self))
                    return
        _original_shuffle_do(self, method, *args, **kwargs)
    
    AgentSet.shuffle_do = _enhanced_shuffle_do_optimized


def enable_automatic_parallel_stepping_optimized(
    mode: str = "asyncio",
    max_concurrent: int = 10,
    request_timeout: float = 30.0
) -> None:
    """
    Legacy function - use enable_automatic_parallel_stepping instead.
    """
    enable_automatic_parallel_stepping(mode, max_concurrent, request_timeout)


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
        for agent in self:
            fn = getattr(agent, method, None)
            if fn is not None and asyncio.iscoroutinefunction(fn):
                tasks.append(fn(*args, **kwargs))
            else:
                raise AttributeError(
                    f"Agent {agent} does not have async method '{method}'"
                )
        return await asyncio.gather(*tasks)

    return _run()


AgentSet.do_async = _agentset_do_async
