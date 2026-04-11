import asyncio

import pytest
from mesa.agent import Agent, AgentSet
from mesa.model import Model

from mesa_llm.parallel_stepping import (
    disable_automatic_parallel_stepping,
    enable_automatic_parallel_stepping,
    step_agents_multithreaded,
    step_agents_parallel,
    step_agents_parallel_sync,
)


class DummyModel(Model):
    def __init__(self):
        super().__init__(rng=42)
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


@pytest.mark.asyncio
async def test_step_agents_parallel():
    m = DummyModel()
    a1 = SyncAgent(m)
    a2 = AsyncAgent(m)
    await step_agents_parallel([a1, a2])
    assert a1.counter == 1
    assert a2.counter == 1


def test_step_agents_multithreaded():
    m = DummyModel()
    a1 = SyncAgent(m)
    a2 = AsyncAgent(m)
    step_agents_multithreaded([a1, a2])
    assert a1.counter == 1
    assert a2.counter == 1


def test_automatic_parallel_shuffle_do():
    """
    verify that enable_automatic_parallel_stepping
    monkey patches AgentSet.shuffle_do and ends up
    using step_agents_parallel_sync
    """
    disable_automatic_parallel_stepping()  # Ensure clean state
    m = DummyModel()
    m.parallel_stepping = True

    # SyncAgent that will be called by AgentSet.shuffle_do
    a1 = SyncAgent(m)
    agents = AgentSet([a1], random=m.random)

    # enable patch
    enable_automatic_parallel_stepping("asyncio")

    # shuffle_do should now call step_agents_parallel_sync
    # instead of individual step, so the counter still ends up 1
    agents.shuffle_do("step")
    assert a1.counter == 1

    # disable patch and check that shuffle_do calls default (and will step again)
    disable_automatic_parallel_stepping()
    agents.shuffle_do("step")
    assert a1.counter == 2
    disable_automatic_parallel_stepping()


def test_step_agents_parallel_sync_in_running_loop():
    # ensure no exception is raised if we call the sync wrapper
    # while an event loop is already running
    m = DummyModel()
    a1 = SyncAgent(m)
    a2 = AsyncAgent(m)

    async def wrapper():
        # running inside an event loop
        step_agents_parallel_sync([a1, a2])

    asyncio.run(wrapper())
    assert a1.counter == 1
    assert a2.counter == 1
