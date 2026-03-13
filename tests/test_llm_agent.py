# tests/test_llm_agent.py

import json
import re
from unittest.mock import patch

import pytest
from mesa.discrete_space import OrthogonalMooreGrid
from mesa.model import Model
from mesa.space import ContinuousSpace, MultiGrid, SingleGrid

from mesa_llm import Plan
from mesa_llm.llm_agent import LLMAgent
from mesa_llm.memory.st_memory import ShortTermMemory
from mesa_llm.reasoning.react import ReActReasoning
from mesa_llm.reasoning.reasoning import Reasoning

# === Test Helper Functions ===


class MockReasoning(Reasoning):
    """Mock reasoning class for testing without API calls."""

    def plan(self, prompt, obs, selected_tools=None):
        return Plan(step=1, llm_plan="Mock plan response")

    async def aplan(self, prompt, obs, selected_tools=None):
        return Plan(step=1, llm_plan="Mock async plan response")


def create_dummy_model(seed=42, grid_type="MultiGrid", **grid_kwargs):
    """Create a standardized dummy model for testing."""

    class DummyModel(Model):
        def __init__(self, seed=seed):
            super().__init__(seed=seed)
            if grid_type == "MultiGrid":
                self.grid = MultiGrid(**grid_kwargs)
            elif grid_type == "SingleGrid":
                self.grid = SingleGrid(**grid_kwargs)
            elif grid_type == "OrthogonalMooreGrid":
                self.grid = OrthogonalMooreGrid(
                    dimensions=grid_kwargs["dimensions"], random=self.random
                )
            elif grid_type == "ContinuousSpace":
                self.space = ContinuousSpace(**grid_kwargs)
            # No grid/space if grid_type is None

    return DummyModel(seed)


def create_test_agent(
    model,
    pos=None,
    reasoning=ReActReasoning,
    system_prompt="You are an agent in a simulation.",
    vision=-1,
    internal_state=["test_state"],
    memory_config=None,
):
    """Create a standardized test agent."""
    agents = LLMAgent.create_agents(
        model,
        n=1,
        reasoning=reasoning,
        system_prompt=system_prompt,
        vision=vision,
        internal_state=internal_state,
    )
    agent = agents.to_list()[0]

    # Set up memory
    memory_config = memory_config or {"n": 5, "display": True}
    agent.memory = ShortTermMemory(agent=agent, **memory_config)

    # Place agent if position provided and grid exists
    if pos and hasattr(model, "grid"):
        model.grid.place_agent(agent, pos)

    return agent


def create_two_agent_model(seed=45):
    """Create a model with two agents for message testing."""
    model = create_dummy_model(
        seed=seed, grid_type="MultiGrid", width=3, height=3, torus=False
    )

    sender = create_test_agent(
        model,
        pos=(0, 0),
        reasoning=lambda agent: None,
        system_prompt="Test",
        vision=-1,
        internal_state=[],
    )
    sender.unique_id = 10

    recipient = create_test_agent(
        model,
        pos=(1, 1),
        reasoning=lambda agent: None,
        system_prompt="Test",
        vision=-1,
        internal_state=[],
    )
    recipient.unique_id = 20

    return sender, recipient


class MockCell:
    """Minimal mock of a CellAgent cell with just a coordinate attribute."""

    def __init__(self, coordinate):
        self.coordinate = coordinate


def test_apply_plan_adds_to_memory(monkeypatch):
    model = create_dummy_model(
        seed=42, grid_type="MultiGrid", width=3, height=3, torus=False
    )
    agent = create_test_agent(model, pos=(1, 1))

    # fake response returned by the tool manager
    fake_response = [{"tool": "foo", "argument": "bar"}]

    # monkeypatch the tool manager so no real tool calls are made
    monkeypatch.setattr(
        agent.tool_manager, "call_tools", lambda agent, llm_response: fake_response
    )

    plan = Plan(step=0, llm_plan="do something")

    resp = agent.apply_plan(plan)

    assert resp == fake_response

    assert {
        "tool": "foo",
        "argument": "bar",
    } in agent.memory.step_content.values() or agent.memory.step_content == {
        "tool": "foo",
        "argument": "bar",
    }


def test_generate_obs_with_one_neighbor(monkeypatch):
    model = create_dummy_model(
        seed=45, grid_type="MultiGrid", width=3, height=3, torus=False
    )

    agent = create_test_agent(model, pos=(1, 1))
    agent.unique_id = 1

    neighbor = create_test_agent(model, pos=(1, 2))
    neighbor.unique_id = 2
    monkeypatch.setattr(agent.memory, "add_to_memory", lambda *args, **kwargs: None)

    obs = agent.generate_obs()

    assert obs.self_state["agent_unique_id"] == 1

    # we should have exactly one neighboring agent in local_state
    assert len(obs.local_state) == 1

    # extract the neighbor
    key = next(iter(obs.local_state.keys()))
    assert key == "LLMAgent 2"

    entry = obs.local_state[key]
    assert entry["position"] == (1, 2)
    assert entry["internal_state"] == ["test_state"]


def test_send_message_updates_both_agents_memory(monkeypatch):
    sender, recipient = create_two_agent_model(seed=45)

    # Track how many times add_to_memory is called
    call_counter = {"count": 0}

    def fake_add_to_memory(*args, **kwargs):
        call_counter["count"] += 1

    # monkeypatch both agents' memory modules
    monkeypatch.setattr(sender.memory, "add_to_memory", fake_add_to_memory)
    monkeypatch.setattr(recipient.memory, "add_to_memory", fake_add_to_memory)

    result = sender.send_message("hello", recipients=[recipient])
    pattern = r"LLMAgent 10 → \[<mesa_llm\.llm_agent\.LLMAgent object at 0x[0-9A-Fa-f]+>\] : hello"
    assert re.match(pattern, result)

    # sender + recipient memory => should be called twice
    assert call_counter["count"] == 2


@pytest.mark.asyncio
async def test_aapply_plan_adds_to_memory(monkeypatch):
    model = create_dummy_model(
        seed=42, grid_type="MultiGrid", width=3, height=3, torus=False
    )
    agent = create_test_agent(model, pos=(1, 1))

    # optional: you can replace with async memory stub
    async def fake_aadd_to_memory(*args, **kwargs):
        pass

    monkeypatch.setattr(agent.memory, "aadd_to_memory", fake_aadd_to_memory)

    # fake async tool response
    fake_response = [{"tool": "foo", "argument": "bar"}]

    async def fake_acall_tools(agent, llm_response):
        return fake_response

    monkeypatch.setattr(agent.tool_manager, "acall_tools", fake_acall_tools)

    plan = Plan(step=0, llm_plan="do something")

    resp = await agent.aapply_plan(plan)

    assert resp == fake_response


@pytest.mark.asyncio
async def test_agenerate_obs_with_one_neighbor(monkeypatch):
    model = create_dummy_model(
        seed=45, grid_type="MultiGrid", width=3, height=3, torus=False
    )

    agent = create_test_agent(model, pos=(1, 1))
    neighbor = create_test_agent(model, pos=(1, 2))

    agent.unique_id = 1
    neighbor.unique_id = 2

    async def fake_aadd_to_memory(*args, **kwargs):
        pass

    monkeypatch.setattr(agent.memory, "aadd_to_memory", fake_aadd_to_memory)

    obs = await agent.agenerate_obs()

    assert obs.self_state["agent_unique_id"] == 1
    assert len(obs.local_state) == 1

    key = next(iter(obs.local_state.keys()))
    assert key == "LLMAgent 2"

    entry = obs.local_state[key]
    assert entry["position"] == (1, 2)
    assert entry["internal_state"] == ["test_state"]


@pytest.mark.asyncio
async def test_async_wrapper_calls_pre_and_post(monkeypatch):
    class CustomAgent(LLMAgent):
        async def astep(self):
            self.user_called = True
            return "done"

    model = create_dummy_model(
        seed=1, grid_type="MultiGrid", width=3, height=3, torus=False
    )

    agent = CustomAgent.create_agents(
        model,
        n=1,
        reasoning=lambda agent: None,
        system_prompt="test",
        vision=-1,
        internal_state=[],
    ).to_list()[0]

    calls = {"pre": 0, "post": 0}

    async def fake_aprocess_step(pre_step=False):
        if pre_step:
            calls["pre"] += 1
        else:
            calls["post"] += 1

    monkeypatch.setattr(agent.memory, "aprocess_step", fake_aprocess_step)

    result = await agent.astep()

    assert result == "done"
    assert calls["pre"] == 1
    assert calls["post"] == 1
    assert agent.user_called is True


def _make_agent(model, vision=0, internal_state=None):
    """Helper: create one LLMAgent and attach fresh ShortTermMemory."""
    return create_test_agent(
        model, vision=vision, internal_state=internal_state or ["test"]
    )


def test_safer_cell_access_agent_with_cell_no_pos(monkeypatch):
    """Agent location falls back to cell.coordinate when pos=None."""
    model = Model(seed=42)
    agent = _make_agent(model)
    agent.pos = None
    agent.cell = MockCell(coordinate=(3, 4))
    monkeypatch.setattr(agent.memory, "add_to_memory", lambda *a, **kw: None)

    obs = agent.generate_obs()

    assert obs.self_state["location"] == (3, 4)


def test_safer_cell_access_agent_without_cell_or_pos(monkeypatch):
    """Agent location returns None gracefully when neither pos nor cell exists."""
    model = Model(seed=42)
    agent = _make_agent(model)
    agent.pos = None
    if hasattr(agent, "cell"):
        delattr(agent, "cell")
    monkeypatch.setattr(agent.memory, "add_to_memory", lambda *a, **kw: None)

    obs = agent.generate_obs()

    assert obs.self_state["location"] is None


def test_safer_cell_access_neighbor_with_cell_no_pos(monkeypatch):
    """Neighbor position uses cell.coordinate when neighbor.pos=None."""

    class GridModel(Model):
        def __init__(self):
            super().__init__(seed=42)
            self.grid = MultiGrid(3, 3, torus=False)

    model = GridModel()
    agents = LLMAgent.create_agents(
        model,
        n=2,
        reasoning=ReActReasoning,
        system_prompt="Test",
        vision=-1,
        internal_state=["test"],
    )
    agent, neighbor = agents
    agent.unique_id = 1
    neighbor.unique_id = 2
    agent.memory = ShortTermMemory(agent=agent, n=5, display=True)
    neighbor.memory = ShortTermMemory(agent=neighbor, n=5, display=True)

    model.grid.place_agent(agent, (1, 1))
    neighbor.pos = None
    neighbor.cell = MockCell(coordinate=(2, 2))

    monkeypatch.setattr(agent.memory, "add_to_memory", lambda *a, **kw: None)
    obs = agent.generate_obs()

    assert obs.local_state["LLMAgent 2"]["position"] == (2, 2)


def test_safer_cell_access_neighbor_without_cell_or_pos(monkeypatch):
    """Neighbor position returns None when neighbor has neither pos nor cell."""

    class GridModel(Model):
        def __init__(self):
            super().__init__(seed=42)
            self.grid = MultiGrid(3, 3, torus=False)

    model = GridModel()
    agents = LLMAgent.create_agents(
        model,
        n=2,
        reasoning=ReActReasoning,
        system_prompt="Test",
        vision=-1,
        internal_state=["test"],
    )
    agent, neighbor = agents
    agent.unique_id = 1
    neighbor.unique_id = 2
    agent.memory = ShortTermMemory(agent=agent, n=5, display=True)
    neighbor.memory = ShortTermMemory(agent=neighbor, n=5, display=True)

    model.grid.place_agent(agent, (1, 1))
    neighbor.pos = None
    if hasattr(neighbor, "cell"):
        delattr(neighbor, "cell")

    monkeypatch.setattr(agent.memory, "add_to_memory", lambda *a, **kw: None)
    obs = agent.generate_obs()

    assert obs.local_state["LLMAgent 2"]["position"] is None


def test_generate_obs_with_continuous_space(monkeypatch):
    """Agents within vision radius are included; those outside are not."""

    class ContModel(Model):
        def __init__(self):
            super().__init__(seed=42)
            self.space = ContinuousSpace(x_max=10.0, y_max=10.0, torus=False)

    model = ContModel()
    agents = LLMAgent.create_agents(
        model,
        n=3,
        reasoning=ReActReasoning,
        system_prompt="Test",
        vision=2.0,
        internal_state=["test"],
    )
    agent, nearby, far = agents
    agent.unique_id = 1
    nearby.unique_id = 2
    far.unique_id = 3
    for a in agents:
        a.memory = ShortTermMemory(agent=a, n=5, display=True)

    model.space.place_agent(agent, (5.0, 5.0))
    model.space.place_agent(nearby, (6.0, 5.0))  # distance ≈ 1.0
    model.space.place_agent(far, (9.0, 9.0))  # distance ≈ 5.66

    monkeypatch.setattr(agent.memory, "add_to_memory", lambda *a, **kw: None)
    obs = agent.generate_obs()

    assert len(obs.local_state) == 1
    assert "LLMAgent 2" in obs.local_state
    assert "LLMAgent 3" not in obs.local_state


def test_generate_obs_vision_all_agents(monkeypatch):
    """vision=-1 returns all other agents regardless of position."""

    class GridModel(Model):
        def __init__(self):
            super().__init__(seed=42)
            self.grid = MultiGrid(10, 10, torus=False)

    model = GridModel()
    agents = LLMAgent.create_agents(
        model,
        n=4,
        reasoning=ReActReasoning,
        system_prompt="Test",
        vision=-1,
        internal_state=["test"],
    )
    for idx, a in enumerate(agents):
        a.unique_id = idx + 1
        a.memory = ShortTermMemory(agent=a, n=5, display=True)
        model.grid.place_agent(a, (idx, idx))

    agent = agents.to_list()[0]
    monkeypatch.setattr(agent.memory, "add_to_memory", lambda *a, **kw: None)
    obs = agent.generate_obs()

    # Should see all 3 other agents
    assert len(obs.local_state) == 3
    assert "LLMAgent 2" in obs.local_state
    assert "LLMAgent 3" in obs.local_state
    assert "LLMAgent 4" in obs.local_state


def test_generate_obs_no_grid_with_vision(monkeypatch):
    """When the model has no grid/space, generate_obs falls back to empty neighbors."""
    model = Model(seed=42)  # no grid, no space
    agents = LLMAgent.create_agents(
        model,
        n=2,
        reasoning=ReActReasoning,
        system_prompt="Test",
        vision=5,
        internal_state=["test"],
    )
    agent = agents.to_list()[0]
    agent.unique_id = 1
    agent.memory = ShortTermMemory(agent=agent, n=5, display=True)
    monkeypatch.setattr(agent.memory, "add_to_memory", lambda *a, **kw: None)

    obs = agent.generate_obs()

    assert len(obs.local_state) == 0


def test_generate_obs_standard_grid_with_vision_radius(monkeypatch):
    """
    Tests spatial neighborhood lookup for an LLMAgent on a SingleGrid
    when a positive vision radius is set.

    Verifies that:
    - Agents within the specified vision distance are detected.
    - The observation includes nearby agents in local_state.
    - The SingleGrid neighbor lookup branch is executed.
    """

    class GridModel(Model):
        def __init__(self):
            super().__init__(seed=42)
            # Reverted to width/height for SingleGrid
            self.grid = SingleGrid(width=5, height=5, torus=False)

    model = GridModel()
    agent = LLMAgent(model=model, reasoning=ReActReasoning, vision=1)
    neighbor = LLMAgent(model=model, reasoning=ReActReasoning)

    # Place agents within vision distance
    model.grid.place_agent(agent, (2, 2))
    model.grid.place_agent(neighbor, (2, 3))

    # Mock memory to bypass API logic
    monkeypatch.setattr(agent.memory, "add_to_memory", lambda *args, **kwargs: None)

    obs = agent.generate_obs()

    assert len(obs.local_state) == 1
    assert "LLMAgent" in str(obs.local_state)


def test_generate_obs_orthogonal_grid_branches(monkeypatch):
    """
    Tests the OrthogonalMooreGrid-specific observation logic in generate_obs().

    Checks the following:
    - When the agent is properly added to a cell, its location is correctly detected and included in self_state.
    - When the agent is not present in any grid cell, generate_obs() handles the situation gracefully and returns an empty local_state without errors.

    Covers Orthogonal grid-specific branches including
    cell-based lookup and fallback behavior.
    """

    class OrthoModel(Model):
        def __init__(self):
            super().__init__(seed=42)
            # Pass self.random to ensure reproducibility
            self.grid = OrthogonalMooreGrid(dimensions=(5, 5), random=self.random)

    model = OrthoModel()
    agent = LLMAgent(model=model, reasoning=ReActReasoning, vision=1)

    # Mock memory to bypass API logic
    monkeypatch.setattr(agent.memory, "add_to_memory", lambda *args, **kwargs: None)

    agent_cell = next(
        cell for cell in model.grid.all_cells if cell.coordinate == (2, 2)
    )
    agent_cell.add_agent(agent)
    agent.pos = (2, 2)

    obs = agent.generate_obs()
    assert obs.self_state["location"] == (2, 2)

    agent_cell.remove_agent(agent)
    obs = agent.generate_obs()

    assert len(obs.local_state) == 0


# ---------------------------------------------------------------------------
# send_message / asend_message - store unique_ids, not Agent objects (#156)
# ---------------------------------------------------------------------------


def _make_send_message_model(monkeypatch):
    """Shared setup: two-agent MultiGrid model with ShortTermMemory."""
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")
    return create_two_agent_model(seed=45)


def test_send_message_stores_serializable_ids(monkeypatch):
    """send_message stores sender/recipients as unique_ids, not Agent objects."""
    sender, recipient = _make_send_message_model(monkeypatch)

    captured = {}

    def capture_content(type, content):
        captured.update(content)

    monkeypatch.setattr(recipient.memory, "add_to_memory", capture_content)
    monkeypatch.setattr(sender.memory, "add_to_memory", lambda *a, **kw: None)

    sender.send_message("hello", recipients=[recipient])

    assert captured["sender"] == 10
    assert captured["recipients"] == [20]
    assert captured["message"] == "hello"

    # Must not raise TypeError when serializing
    data = json.loads(json.dumps(captured))
    assert data["sender"] == 10
    assert data["recipients"] == [20]


@pytest.mark.asyncio
async def test_asend_message_stores_serializable_ids(monkeypatch):
    """asend_message stores sender/recipients as unique_ids, not Agent objects."""
    sender, recipient = _make_send_message_model(monkeypatch)

    captured = {}

    async def capture_content(type, content):
        captured.update(content)

    async def noop(*a, **kw):
        pass

    monkeypatch.setattr(recipient.memory, "aadd_to_memory", capture_content)

    await sender.asend_message("hello", recipients=[recipient])

    assert captured["sender"] == 10
    assert captured["recipients"] == [20]
    assert captured["message"] == "hello"

    data = json.loads(json.dumps(captured))
    assert data["sender"] == 10
    assert data["recipients"] == [20]


# === Performance Optimization Integration Tests ===


@pytest.mark.asyncio
async def test_llm_agent_with_optimizations():
    """Test that LLMAgent properly initializes with optimizations."""
    mock_response = "Mock LLM response"

    with (
        patch("mesa_llm.module_llm.completion", return_value=mock_response),
        patch("mesa_llm.module_llm.acompletion", return_value=mock_response),
    ):
        model = create_dummy_model(
            seed=42, grid_type="MultiGrid", width=3, height=3, torus=False
        )

        agent = LLMAgent(
            model=model,
            reasoning=MockReasoning,
            llm_model="test/gpt-4",
            system_prompt="Test system prompt",
        )

        # Verify optimizations are enabled
        assert hasattr(agent, "llm")
        assert hasattr(agent.llm, "enable_caching")
        assert agent.llm.enable_caching is True
        assert hasattr(agent.llm, "enable_batching")
        assert agent.llm.enable_batching is True
        assert hasattr(agent.llm, "cache")
        assert hasattr(agent.llm, "batcher")
        assert hasattr(agent.llm, "connection_pool")

        # Test performance stats
        stats = agent.llm.get_performance_stats()
        assert "request_count" in stats
        assert "cache_hits" in stats
        assert "cache_hit_rate" in stats

        await agent.llm.cleanup()


def test_backward_compatibility():
    """Test that optimizations don't break existing functionality."""
    # Test that agents still work without explicit optimization parameters
    model = create_dummy_model(
        seed=42, grid_type="MultiGrid", width=3, height=3, torus=False
    )

    agent = LLMAgent(model=model, reasoning=MockReasoning, llm_model="test/gpt-4")

    # Should still work with default optimizations enabled
    assert hasattr(agent, "llm")
    assert hasattr(agent, "reasoning")
    assert hasattr(agent, "memory")
    assert hasattr(agent, "tool_manager")

    # Should have optimizations by default
    assert agent.llm.enable_caching is True
    assert agent.llm.enable_batching is True
