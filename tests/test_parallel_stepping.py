import asyncio

import pytest
from mesa.agent import Agent, AgentSet
from mesa.experimental.meta_agents.meta_agent import MetaAgent
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


@pytest.fixture(autouse=True)
def manage_parallel_stepping_patch():
    # Helper to clean up parallel stepping state if tests fail
    yield
    disable_automatic_parallel_stepping()


# --- Meta Agent Parallel Conflict Tests ---


class ConflictWorker(Agent):
    def __init__(self, model):
        super().__init__(model)
        self.step_count = 0

    def step(self):
        self.step_count += 1


class ConflictManager(MetaAgent):
    def step(self):
        for agent in self.agents:
            agent.step()


class ConflictBusinessModel(Model):
    def __init__(self):
        super().__init__(rng=42)
        self.parallel_stepping = True
        self.worker = ConflictWorker(self)
        self.manager = ConflictManager(self, agents={self.worker})

    def step(self):
        self.agents.shuffle_do("step")


def test_meta_agent_parallel_conflict_fix():
    """
    Test that a constituent agent is only stepped ONCE when
    parallel stepping is enabled, because it is skipped by
    the scheduler and only stepped by its MetaAgent.
    """
    enable_automatic_parallel_stepping("asyncio")
    model = ConflictBusinessModel()

    # Run one step
    model.step()

    # Assert worker was only stepped once
    # If the fix failed, this would be 2
    assert model.worker.step_count == 1


def test_meta_agent_multithreaded_conflict_fix():
    """Test same logic but with multithreaded mode."""
    disable_automatic_parallel_stepping()
    enable_automatic_parallel_stepping("threading")

    model = ConflictBusinessModel()
    model.step()

    assert model.worker.step_count == 1


def test_agent_becomes_independent_again():
    """Test that an agent removed from MetaAgent is stepped by scheduler again."""
    enable_automatic_parallel_stepping("asyncio")
    model = ConflictBusinessModel()

    # 1. Initially it's a component
    assert model.worker.is_component is True

    # 2. Remove it from manager
    model.manager.remove_constituting_agents({model.worker})
    assert model.worker.is_component is False

    # 3. Model step should now step the worker directly
    model.step()
    assert model.worker.step_count == 1
