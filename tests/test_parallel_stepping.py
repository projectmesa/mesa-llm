import asyncio
import time

import pytest
from mesa.agent import Agent, AgentSet
from mesa.model import Model

from mesa_llm.parallel_stepping import (
    EventLoopManager,
    SemaphorePool,
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


class MockAgent(Agent):
    """Mock agent for testing parallel stepping."""

    def __init__(self, model, agent_id):
        super().__init__(model)
        self.agent_id = agent_id
        self.steps_taken = 0
        self.async_steps_taken = 0

    async def astep(self):
        """Async step that simulates work."""
        await asyncio.sleep(0.01)  # Simulate 10ms of work
        self.async_steps_taken += 1

    def step(self):
        """Sync step that simulates work."""
        time.sleep(0.01)  # Simulate 10ms of work
        self.steps_taken += 1


# === Test Helper Functions ===


def create_mock_model(num_agents=10, enable_parallel_stepping=True):
    """Create a standardized mock model for testing."""
    model = DummyModel()
    model.parallel_stepping = enable_parallel_stepping
    model.custom_agents = [MockAgent(model, i) for i in range(num_agents)]
    return model


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


# === Performance Optimization Tests ===


@pytest.mark.asyncio
async def test_step_agents_parallel_optimized():
    """Test optimized parallel stepping works correctly."""
    model = create_mock_model(num_agents=5)
    agents = model.custom_agents

    await step_agents_parallel(agents)

    # All agents should have stepped
    for agent in agents:
        assert agent.async_steps_taken == 1


@pytest.mark.asyncio
async def test_parallel_vs_sequential_performance():
    """Test that parallel execution is faster than sequential."""
    agent_counts = [5, 10]

    for num_agents in agent_counts:
        model = create_mock_model(num_agents=num_agents)
        agents = model.custom_agents

        # Sequential execution
        start_time = time.time()
        for agent in agents:
            agent.step()
        sequential_time = time.time() - start_time

        # Reset agents
        for agent in agents:
            agent.steps_taken = 0

        # Parallel execution
        start_time = time.time()
        await step_agents_parallel(agents)
        parallel_time = time.time() - start_time

        # Parallel should be faster or equal
        assert parallel_time <= sequential_time, (
            f"Parallel ({parallel_time:.3f}s) should be <= sequential ({sequential_time:.3f}s) for {num_agents} agents"
        )


def test_step_agents_multithreaded_optimized():
    """Test optimized multithreaded stepping."""
    model = create_mock_model(num_agents=5)
    agents = model.custom_agents

    step_agents_multithreaded(agents)

    # All agents should have stepped (either sync or async)
    for agent in agents:
        # Should have either sync steps or async steps
        total_steps = agent.steps_taken + agent.async_steps_taken
        assert total_steps == 1, (
            f"Agent {agent.agent_id} should have 1 total step, got {total_steps}"
        )


def test_event_loop_manager():
    """Test event loop manager functionality."""
    manager = EventLoopManager()

    # Get loop for current thread
    loop = manager.get_loop_for_thread()
    assert loop is not None

    # Should return same loop for same thread
    loop2 = manager.get_loop_for_thread()
    assert loop is loop2

    # Cleanup
    manager.cleanup()
    assert len(manager.loops) == 0


def test_semaphore_pool():
    """Test semaphore pool functionality."""
    pool = SemaphorePool(max_concurrent=5)

    # Get semaphore
    semaphore = pool.get_semaphore()
    assert semaphore is not None
    assert semaphore._value == 5  # Max concurrent

    # Should return same semaphore for same thread
    semaphore2 = pool.get_semaphore()
    assert semaphore is semaphore2

    # Test with custom key
    custom_semaphore = pool.get_semaphore("custom")
    assert custom_semaphore is not None
    assert custom_semaphore is not semaphore


def test_enable_automatic_parallel_stepping_optimized():
    """Test enabling optimized automatic parallel stepping."""
    # Should not raise any errors
    enable_automatic_parallel_stepping(
        mode="asyncio", max_concurrent=10, request_timeout=30.0
    )

    # Test with threading mode
    enable_automatic_parallel_stepping(
        mode="threading", max_concurrent=5, request_timeout=15.0
    )

    # Test invalid mode raises error
    with pytest.raises(ValueError):
        enable_automatic_parallel_stepping(mode="invalid")
