from __future__ import annotations

from types import SimpleNamespace

from mesa.discrete_space import OrthogonalMooreGrid
from mesa.space import MultiGrid, SingleGrid

from mesa_llm.tools.inbuilt_tools import (
    move_one_step,
    speak_to,
    teleport_to_location,
)


class DummyModel:
    def __init__(self):
        self.grid = None
        self.space = None
        self.agents = []


class DummyAgent:
    def __init__(self, unique_id: int, model: DummyModel):
        self.unique_id = unique_id
        self.model = model
        self.pos = None


def test_move_one_step_on_singlegrid():
    model = DummyModel()
    model.grid = SingleGrid(width=5, height=5, torus=False)

    agent = DummyAgent(unique_id=1, model=model)
    model.agents.append(agent)
    model.grid.place_agent(agent, (2, 2))

    result = move_one_step(agent, "North")

    assert agent.pos == (2, 3)
    assert result == "agent 1 moved to (2, 3)."


def test_teleport_to_location_on_multigrid():
    model = DummyModel()
    model.grid = MultiGrid(width=4, height=4, torus=False)

    agent = DummyAgent(unique_id=7, model=model)
    model.agents.append(agent)
    model.grid.place_agent(agent, (0, 0))

    out = teleport_to_location(agent, [3, 2])

    assert agent.pos == (3, 2)
    assert out == "agent 7 moved to (3, 2)."


def test_teleport_to_location_on_orthogonal_grid_without_constructor():
    # Create an instance of a subclass of OrthogonalMooreGrid without invoking its __init__
    class _DummyOrthogonalGrid(OrthogonalMooreGrid):
        pass

    orth_grid = object.__new__(_DummyOrthogonalGrid)
    target = (1, 1)
    dummy_cell = SimpleNamespace(coordinate=target, agents=[])
    orth_grid._cells = {target: dummy_cell}

    model = DummyModel()
    model.grid = orth_grid

    agent = DummyAgent(unique_id=9, model=model)
    model.agents.append(agent)

    out = teleport_to_location(agent, [1, 1])

    assert getattr(agent, "cell", None) is dummy_cell
    assert out == "agent 9 moved to (1, 1)."


def test_speak_to_records_on_recipients(mocker):
    model = DummyModel()

    # Sender and two recipients
    sender = DummyAgent(unique_id=10, model=model)
    r1 = DummyAgent(unique_id=11, model=model)
    r2 = DummyAgent(unique_id=12, model=model)

    # Attach mock memories to recipients
    r1.memory = SimpleNamespace(add_to_memory=mocker.Mock())
    r2.memory = SimpleNamespace(add_to_memory=mocker.Mock())

    model.agents = [sender, r1, r2]

    message = "Hello there"
    ret = speak_to(sender, [10, 11, 12], message)

    # Sender should not get message recorded, recipients should
    r1.memory.add_to_memory.assert_called_once()
    r2.memory.add_to_memory.assert_called_once()

    # Verify payload structure for one recipient
    _, kwargs = r1.memory.add_to_memory.call_args
    assert kwargs["type"] == "message"
    content = kwargs["content"]
    assert content["message"] == message
    assert content["sender"] == sender.unique_id
    assert set(content["recipients"]) == {11, 12}

    # Return string contains sender and recipients list
    assert "10" in ret and "11" in ret and "12" in ret and message in ret
