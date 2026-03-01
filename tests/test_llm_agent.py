# tests/test_llm_agent.py

import re

import pytest
from mesa.discrete_space import OrthogonalMooreGrid
from mesa.model import Model
from mesa.space import ContinuousSpace, MultiGrid, SingleGrid

from mesa_llm import Plan
from mesa_llm.llm_agent import LLMAgent
from mesa_llm.memory.st_memory import ShortTermMemory
from mesa_llm.reasoning.react import ReActReasoning


def test_apply_plan_adds_to_memory(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")

    class DummyModel(Model):
        def __init__(self):
            super().__init__(seed=42)
            self.grid = MultiGrid(3, 3, torus=False)

        def add_agent(self, pos):
            system_prompt = "You are an agent in a simulation."
            agents = LLMAgent.create_agents(
                self,
                n=1,
                reasoning=ReActReasoning,
                system_prompt=system_prompt,
                vision=-1,
                internal_state=["test_state"],
            )

            x, y = pos

            agent = agents.to_list()[0]
            self.grid.place_agent(agent, (x, y))
            return agent

    model = DummyModel()
    agent = model.add_agent((1, 1))
    agent.memory = ShortTermMemory(
        agent=agent,
        n=5,
        display=True,
    )

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
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")

    class DummyModel(Model):
        def __init__(self):
            super().__init__(seed=45)
            self.grid = MultiGrid(3, 3, torus=False)

        def add_agent(self, pos, agent_class=LLMAgent):
            system_prompt = "You are an agent in a simulation."
            agents = agent_class.create_agents(
                self,
                n=1,
                reasoning=ReActReasoning,
                system_prompt=system_prompt,
                vision=-1,
                internal_state=["test_state"],
            )
            x, y = pos
            agent = agents.to_list()[0]
            self.grid.place_agent(agent, (x, y))
            return agent

    model = DummyModel()

    agent = model.add_agent((1, 1))
    agent.memory = ShortTermMemory(
        agent=agent,
        n=5,
        display=True,
    )
    agent.unique_id = 1

    neighbor = model.add_agent((1, 2))
    neighbor.memory = ShortTermMemory(
        agent=agent,
        n=5,
        display=True,
    )
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
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")

    class DummyModel(Model):
        def __init__(self):
            super().__init__(seed=45)
            self.grid = MultiGrid(3, 3, torus=False)

        def add_agent(self, pos, agent_class=LLMAgent):
            system_prompt = "You are an agent in a simulation."
            agents = agent_class.create_agents(
                self,
                n=1,
                reasoning=lambda agent: None,
                system_prompt=system_prompt,
                vision=-1,
                internal_state=["test_state"],
            )
            x, y = pos
            agent = agents.to_list()[0]
            self.grid.place_agent(agent, (x, y))
            return agent

    model = DummyModel()
    sender = model.add_agent((0, 0))
    sender.memory = ShortTermMemory(
        agent=sender,
        n=5,
        display=True,
    )
    sender.unique_id = 1

    recipient = model.add_agent((1, 1))
    recipient.memory = ShortTermMemory(
        agent=recipient,
        n=5,
        display=True,
    )
    recipient.unique_id = 2

    # Track how many times add_to_memory is called
    call_counter = {"count": 0}

    def fake_add_to_memory(*args, **kwargs):
        call_counter["count"] += 1

    # monkeypatch both agents' memory modules
    monkeypatch.setattr(sender.memory, "add_to_memory", fake_add_to_memory)
    monkeypatch.setattr(recipient.memory, "add_to_memory", fake_add_to_memory)

    result = sender.send_message("hello", recipients=[recipient])
    pattern = r"LLMAgent 1 → \[<mesa_llm\.llm_agent\.LLMAgent object at 0x[0-9A-Fa-f]+>\] : hello"
    assert re.match(pattern, result)

    # sender + recipient memory => should be called twice
    assert call_counter["count"] == 2


@pytest.mark.asyncio
async def test_aapply_plan_adds_to_memory(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")

    class DummyModel(Model):
        def __init__(self):
            super().__init__(seed=42)
            self.grid = MultiGrid(3, 3, torus=False)

        def add_agent(self, pos):
            system_prompt = "You are an agent in a simulation."
            agents = LLMAgent.create_agents(
                self,
                n=1,
                reasoning=ReActReasoning,
                system_prompt=system_prompt,
                vision=-1,
                internal_state=["test_state"],
            )

            x, y = pos
            agent = agents.to_list()[0]
            self.grid.place_agent(agent, (x, y))
            return agent

    model = DummyModel()
    agent = model.add_agent((1, 1))

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
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")

    class DummyModel(Model):
        def __init__(self):
            super().__init__(seed=45)
            self.grid = MultiGrid(3, 3, torus=False)

        def add_agent(self, pos):
            agents = LLMAgent.create_agents(
                self,
                n=1,
                reasoning=ReActReasoning,
                system_prompt="You are an agent.",
                vision=-1,
                internal_state=["test_state"],
            )
            x, y = pos
            agent = agents.to_list()[0]
            self.grid.place_agent(agent, (x, y))
            return agent

    model = DummyModel()

    agent = model.add_agent((1, 1))
    neighbor = model.add_agent((1, 2))

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
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")

    class CustomAgent(LLMAgent):
        async def astep(self):
            self.user_called = True
            return "done"

    class DummyModel(Model):
        def __init__(self):
            super().__init__(seed=1)
            self.grid = MultiGrid(3, 3, torus=False)

    model = DummyModel()

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


class MockCell:
    """Minimal mock of a CellAgent cell with just a coordinate attribute."""

    def __init__(self, coordinate):
        self.coordinate = coordinate


def _make_agent(model, vision=0, internal_state=None):
    """Helper: create one LLMAgent and attach fresh ShortTermMemory."""
    agents = LLMAgent.create_agents(
        model,
        n=1,
        reasoning=ReActReasoning,
        system_prompt="Test",
        vision=vision,
        internal_state=internal_state or ["test"],
    )
    agent = agents.to_list()[0]
    agent.memory = ShortTermMemory(agent=agent, n=5, display=True)
    return agent


def test_safer_cell_access_agent_with_cell_no_pos(monkeypatch):
    """Agent location falls back to cell.coordinate when pos=None."""
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")
    model = Model(seed=42)
    agent = _make_agent(model)
    agent.pos = None
    agent.cell = MockCell(coordinate=(3, 4))
    monkeypatch.setattr(agent.memory, "add_to_memory", lambda *a, **kw: None)

    obs = agent.generate_obs()

    assert obs.self_state["location"] == (3, 4)


def test_safer_cell_access_agent_without_cell_or_pos(monkeypatch):
    """Agent location returns None gracefully when neither pos nor cell exists."""
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")
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
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")

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
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")

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
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")

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
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")

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
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")
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
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")

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
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")

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
