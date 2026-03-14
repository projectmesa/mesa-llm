import asyncio

import pytest
from mesa.agent import Agent, AgentSet
from mesa.model import Model

from mesa_llm.parallel_stepping import (
    AgentStepResult,
    disable_automatic_parallel_stepping,
    enable_automatic_parallel_stepping,
    step_agents_multithreaded,
    step_agents_parallel,
    step_agents_parallel_sync,
)


class DummyModel(Model):
    def __init__(self):
        super().__init__(seed=42)
        self.parallel_stepping = False


class SyncAgent(Agent):
    def __init__(self, model):
        super().__init__(model)
        self.counter = 0

    def step(self):
        self.counter += 1


class AsyncAgent(Agent):
    def __init__(self, model):
        super().__init__(model)
        self.counter = 0

    async def astep(self):
        self.counter += 1


class FailingAsyncAgent(Agent):
    def __init__(self, model):
        super().__init__(model)

    async def astep(self):
        raise RuntimeError("LLM timeout")


class FailingSyncAgent(Agent):
    def __init__(self, model):
        super().__init__(model)

    def step(self):
        raise ValueError("bad response")


class SlowAsyncAgent(Agent):
    def __init__(self, model):
        super().__init__(model)
        self.counter = 0

    async def astep(self):
        await asyncio.sleep(10)
        self.counter += 1


# --- Async path tests ---


@pytest.mark.asyncio
async def test_parallel_all_succeed_returns_results():
    m = DummyModel()
    a1 = AsyncAgent(m)
    a2 = SyncAgent(m)
    results = await step_agents_parallel([a1, a2])
    assert len(results) == 2
    assert all(isinstance(r, AgentStepResult) for r in results)
    assert all(r.success for r in results)
    assert all(r.exception is None for r in results)
    assert a1.counter == 1
    assert a2.counter == 1


@pytest.mark.asyncio
async def test_parallel_one_fails_others_continue():
    m = DummyModel()
    good = AsyncAgent(m)
    bad = FailingAsyncAgent(m)
    results = await step_agents_parallel([bad, good])
    assert good.counter == 1
    failed = [r for r in results if not r.success]
    succeeded = [r for r in results if r.success]
    assert len(failed) == 1
    assert len(succeeded) == 1
    assert isinstance(failed[0].exception, RuntimeError)
    assert failed[0].agent is bad


@pytest.mark.asyncio
async def test_parallel_on_error_raise():
    m = DummyModel()
    good = AsyncAgent(m)
    bad = FailingAsyncAgent(m)
    with pytest.raises(RuntimeError, match="LLM timeout"):
        await step_agents_parallel([bad, good], on_error="raise")


def test_parallel_on_error_invalid_value():
    with pytest.raises(ValueError, match="on_error"):
        enable_automatic_parallel_stepping(on_error="ignore")


@pytest.mark.asyncio
async def test_parallel_timeout_slow_agent():
    m = DummyModel()
    fast = AsyncAgent(m)
    slow = SlowAsyncAgent(m)
    results = await step_agents_parallel([slow, fast], timeout=0.1)
    assert fast.counter == 1
    assert slow.counter == 0
    failed = [r for r in results if not r.success]
    assert len(failed) == 1
    assert isinstance(failed[0].exception, asyncio.TimeoutError)


@pytest.mark.asyncio
async def test_parallel_no_timeout_all_complete():
    m = DummyModel()
    a1 = AsyncAgent(m)
    a2 = AsyncAgent(m)
    results = await step_agents_parallel([a1, a2], timeout=5.0)
    assert all(r.success for r in results)
    assert a1.counter == 1
    assert a2.counter == 1


# --- Threaded path tests ---


def test_threaded_one_fails_others_continue():
    m = DummyModel()
    good = AsyncAgent(m)
    bad = FailingAsyncAgent(m)
    results = step_agents_multithreaded([bad, good])
    assert good.counter == 1
    failed = [r for r in results if not r.success]
    succeeded = [r for r in results if r.success]
    assert len(failed) == 1
    assert len(succeeded) == 1
    assert isinstance(failed[0].exception, RuntimeError)


def test_threaded_on_error_raise():
    m = DummyModel()
    good = AsyncAgent(m)
    bad = FailingAsyncAgent(m)
    with pytest.raises(RuntimeError, match="LLM timeout"):
        step_agents_multithreaded([bad, good], on_error="raise")


def test_threaded_timeout():
    m = DummyModel()
    fast = SyncAgent(m)
    slow = SlowAsyncAgent(m)
    results = step_agents_multithreaded([slow, fast], timeout=0.1)
    assert fast.counter == 1
    failed = [r for r in results if not r.success]
    assert len(failed) == 1


# --- Sync wrapper tests ---


def test_sync_wrapper_forwards_params():
    m = DummyModel()
    good = AsyncAgent(m)
    bad = FailingAsyncAgent(m)
    results = step_agents_parallel_sync([bad, good], on_error="continue")
    assert good.counter == 1
    failed = [r for r in results if not r.success]
    assert len(failed) == 1
    assert isinstance(failed[0].exception, RuntimeError)


# --- do_async tests ---


@pytest.mark.asyncio
async def test_do_async_one_fails_others_continue():
    m = DummyModel()
    good = AsyncAgent(m)
    bad = FailingAsyncAgent(m)
    agents = AgentSet([bad, good], random=m.random)
    results = await agents.do_async("astep")
    assert good.counter == 1
    exceptions = [r for r in results if isinstance(r, BaseException)]
    assert len(exceptions) == 1
    assert isinstance(exceptions[0], RuntimeError)


# --- Config tests ---


def test_enable_config_sets_globals():
    import mesa_llm.parallel_stepping as ps

    enable_automatic_parallel_stepping(on_error="raise", timeout=5.0)
    assert ps._PARALLEL_ON_ERROR == "raise"
    assert ps._PARALLEL_TIMEOUT == 5.0

    disable_automatic_parallel_stepping()
    assert ps._PARALLEL_ON_ERROR == "continue"
    assert ps._PARALLEL_TIMEOUT is None


def test_backward_compat_no_params():
    m = DummyModel()
    a1 = AsyncAgent(m)
    a2 = SyncAgent(m)
    results = step_agents_parallel_sync([a1, a2])
    assert a1.counter == 1
    assert a2.counter == 1
    assert len(results) == 2
    assert all(r.success for r in results)


# --- Coverage gap tests ---


class _Bare:
    """Object with neither step() nor astep() — should be skipped."""


@pytest.mark.asyncio
async def test_parallel_skips_agent_without_step_or_astep():
    m = DummyModel()
    good = AsyncAgent(m)
    noop = _Bare()
    results = await step_agents_parallel([noop, good])
    assert len(results) == 1
    assert results[0].success
    assert good.counter == 1


def test_threaded_skips_agent_without_step_or_astep():
    m = DummyModel()
    good = SyncAgent(m)
    noop = _Bare()
    results = step_agents_multithreaded([noop, good])
    assert len(results) == 1
    assert results[0].success
    assert good.counter == 1


def test_sync_wrapper_unknown_mode_raises():
    import mesa_llm.parallel_stepping as ps

    original = ps._PARALLEL_STEPPING_MODE
    try:
        ps._PARALLEL_STEPPING_MODE = "unknown"
        with pytest.raises(ValueError, match="Unknown parallel stepping mode"):
            step_agents_parallel_sync([])
    finally:
        ps._PARALLEL_STEPPING_MODE = original
