from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
from mesa.discrete_space import OrthogonalMooreGrid, OrthogonalVonNeumannGrid
from mesa.space import ContinuousSpace, MultiGrid, SingleGrid

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
    orth_grid.torus = False
    orth_grid.dimensions = (3, 3)
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


def test_move_one_step_on_orthogonal_grid_without_constructor():
    class _DummyOrthogonalGrid(OrthogonalMooreGrid):
        pass

    orth_grid = object.__new__(_DummyOrthogonalGrid)
    orth_grid.torus = False
    orth_grid.dimensions = (5, 5)
    start_target = (1, 1)
    # mesa.discrete_space grids use (row, col), so North decrements row.
    end_target = (0, 1)
    start_cell = SimpleNamespace(
        coordinate=start_target, agents=[], connections={}, is_full=False
    )
    end_cell = SimpleNamespace(
        coordinate=end_target, agents=[], connections={}, is_full=False
    )
    start_cell.connections[(-1, 0)] = end_cell
    orth_grid._cells = {start_target: start_cell, end_target: end_cell}

    model = DummyModel()
    model.grid = orth_grid

    agent = DummyAgent(unique_id=10, model=model)
    agent.cell = start_cell
    model.agents.append(agent)

    out = move_one_step(agent, "North")

    assert getattr(agent, "cell", None) is end_cell
    assert out == "agent 10 moved to (0, 1)."


def test_move_one_step_east_on_orthogonal_grid_without_constructor():
    class _DummyOrthogonalGrid(OrthogonalMooreGrid):
        pass

    orth_grid = object.__new__(_DummyOrthogonalGrid)
    orth_grid.torus = False
    orth_grid.dimensions = (5, 5)
    start_target = (1, 1)
    # mesa.discrete_space grids use (row, col), so East increments col.
    end_target = (1, 2)
    start_cell = SimpleNamespace(
        coordinate=start_target, agents=[], connections={}, is_full=False
    )
    end_cell = SimpleNamespace(
        coordinate=end_target, agents=[], connections={}, is_full=False
    )
    start_cell.connections[(0, 1)] = end_cell
    orth_grid._cells = {start_target: start_cell, end_target: end_cell}

    model = DummyModel()
    model.grid = orth_grid

    agent = DummyAgent(unique_id=11, model=model)
    agent.cell = start_cell
    model.agents.append(agent)

    out = move_one_step(agent, "East")

    assert getattr(agent, "cell", None) is end_cell
    assert out == "agent 11 moved to (1, 2)."


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
    assert "recipients" not in content
    assert ret == "sent message 'Hello there' to [11, 12]"


def test_move_one_step_invalid_direction():
    model = DummyModel()
    model.grid = MultiGrid(width=4, height=4, torus=False)

    agent = DummyAgent(unique_id=3, model=model)
    model.agents.append(agent)
    model.grid.place_agent(agent, (2, 2))

    with pytest.raises(ValueError):
        move_one_step(agent, "north east")


def test_move_one_step_unsupported_environment():
    model = DummyModel()
    model.grid = None
    model.space = None

    agent = DummyAgent(unique_id=4, model=model)
    model.agents.append(agent)
    agent.pos = (1, 1)

    with pytest.raises(ValueError, match="Unsupported environment"):
        move_one_step(agent, "North")


def test_move_one_step_unsupported_non_none_environment():
    class _UnsupportedGrid:
        pass

    class _UnsupportedSpace:
        pass

    model = DummyModel()
    model.grid = _UnsupportedGrid()
    model.space = _UnsupportedSpace()

    agent = DummyAgent(unique_id=32, model=model)
    model.agents.append(agent)
    agent.pos = (1, 1)

    with pytest.raises(ValueError, match="Unsupported environment"):
        move_one_step(agent, "North")


def test_teleport_to_location_unsupported_environment():
    model = DummyModel()
    model.grid = None
    model.space = None

    agent = DummyAgent(unique_id=8, model=model)
    model.agents.append(agent)
    agent.pos = (1, 1)

    with pytest.raises(ValueError, match="Unsupported environment"):
        teleport_to_location(agent, [2, 2])


def test_teleport_to_location_unsupported_non_none_environment():
    class _UnsupportedGrid:
        pass

    class _UnsupportedSpace:
        pass

    model = DummyModel()
    model.grid = _UnsupportedGrid()
    model.space = _UnsupportedSpace()

    agent = DummyAgent(unique_id=33, model=model)
    model.agents.append(agent)
    agent.pos = (1, 1)

    with pytest.raises(ValueError, match="Unsupported environment"):
        teleport_to_location(agent, [2, 2])


def test_teleport_to_location_on_continuousspace():
    model = DummyModel()
    model.grid = None
    model.space = ContinuousSpace(x_max=10.0, y_max=10.0, torus=False)

    agent = DummyAgent(unique_id=5, model=model)
    model.agents.append(agent)
    model.space.place_agent(agent, (1.0, 1.0))

    out = teleport_to_location(agent, [5.0, 7.0])

    assert agent.pos == (5.0, 7.0)
    assert out == "agent 5 moved to (5.0, 7.0)."


def test_teleport_to_location_singlegrid_occupied_target_raises():
    model = DummyModel()
    model.grid = SingleGrid(width=4, height=4, torus=False)

    moving_agent = DummyAgent(unique_id=34, model=model)
    blocking_agent = DummyAgent(unique_id=35, model=model)
    model.agents.extend([moving_agent, blocking_agent])
    model.grid.place_agent(moving_agent, (1, 1))
    model.grid.place_agent(blocking_agent, (1, 2))

    with pytest.raises(Exception, match="Cell not empty"):
        teleport_to_location(moving_agent, [1, 2])


def test_teleport_to_location_singlegrid_out_of_bounds_raises():
    model = DummyModel()
    model.grid = SingleGrid(width=4, height=4, torus=False)

    agent = DummyAgent(unique_id=36, model=model)
    model.agents.append(agent)
    model.grid.place_agent(agent, (1, 1))

    with pytest.raises(Exception, match="Point out of bounds"):
        teleport_to_location(agent, [-1, 1])


def test_teleport_to_location_orthogonal_missing_cell_raises_keyerror():
    class _DummyOrthogonalGrid(OrthogonalMooreGrid):
        pass

    orth_grid = object.__new__(_DummyOrthogonalGrid)
    orth_grid.torus = False
    orth_grid.dimensions = (3, 3)
    start = (1, 1)
    start_cell = SimpleNamespace(coordinate=start, agents=[], is_full=False)
    orth_grid._cells = {start: start_cell}

    model = DummyModel()
    model.grid = orth_grid

    agent = DummyAgent(unique_id=37, model=model)
    agent.cell = start_cell
    model.agents.append(agent)

    with pytest.raises(KeyError):
        teleport_to_location(agent, [0, 1])


def test_move_one_step_on_continuousspace():
    """move_one_step delegates to teleport_to_location, verify it works on ContinuousSpace too."""
    model = DummyModel()
    model.grid = None
    model.space = ContinuousSpace(x_max=10.0, y_max=10.0, torus=False)

    agent = DummyAgent(unique_id=6, model=model)
    model.agents.append(agent)
    model.space.place_agent(agent, (2.0, 2.0))

    result = move_one_step(agent, "North")

    assert agent.pos == (2.0, 3.0)
    assert result == "agent 6 moved to (2.0, 3.0)."


def test_move_one_step_boundary_on_continuousspace():
    model = DummyModel()
    model.grid = None
    model.space = ContinuousSpace(x_max=10.0, y_max=10.0, torus=False)

    agent = DummyAgent(unique_id=30, model=model)
    model.agents.append(agent)
    model.space.place_agent(agent, (2.0, 9.0))

    result = move_one_step(agent, "North")

    assert agent.pos == (2.0, 9.0)
    assert "boundary" in result.lower()
    assert "North" in result


def test_move_one_step_torus_wrap_on_continuousspace():
    model = DummyModel()
    model.grid = None
    model.space = ContinuousSpace(x_max=10.0, y_max=10.0, torus=True)

    agent = DummyAgent(unique_id=31, model=model)
    model.agents.append(agent)
    model.space.place_agent(agent, (2.0, 9.0))

    result = move_one_step(agent, "North")

    assert agent.pos == (2.0, 0.0)
    assert result == "agent 31 moved to (2.0, 0.0)."


def test_move_one_step_boundary_singlegrid_north():
    """Agent at top edge of SingleGrid trying to go North gets a clear message."""
    model = DummyModel()
    model.grid = SingleGrid(width=5, height=5, torus=False)

    agent = DummyAgent(unique_id=20, model=model)
    model.agents.append(agent)
    model.grid.place_agent(agent, (2, 4))  # y=4 is the top edge

    result = move_one_step(agent, "North")

    # agent should not have moved
    assert agent.pos == (2, 4)
    assert "boundary" in result.lower()
    assert "North" in result


def test_move_one_step_torus_wrap_singlegrid_north():
    model = DummyModel()
    model.grid = SingleGrid(width=5, height=5, torus=True)

    agent = DummyAgent(unique_id=23, model=model)
    model.agents.append(agent)
    model.grid.place_agent(agent, (2, 4))

    result = move_one_step(agent, "North")

    assert agent.pos == (2, 0)
    assert result == "agent 23 moved to (2, 0)."


def test_move_one_step_boundary_multigrid_west():
    """Agent at left edge of MultiGrid trying to go West gets a clear message."""
    model = DummyModel()
    model.grid = MultiGrid(width=5, height=5, torus=False)

    agent = DummyAgent(unique_id=21, model=model)
    model.agents.append(agent)
    model.grid.place_agent(agent, (0, 2))  # x=0 is the left edge

    result = move_one_step(agent, "West")

    assert agent.pos == (0, 2)
    assert "boundary" in result.lower()
    assert "West" in result


def test_move_one_step_torus_wrap_multigrid_west():
    model = DummyModel()
    model.grid = MultiGrid(width=5, height=5, torus=True)

    agent = DummyAgent(unique_id=24, model=model)
    model.agents.append(agent)
    model.grid.place_agent(agent, (0, 2))

    result = move_one_step(agent, "West")

    assert agent.pos == (4, 2)
    assert result == "agent 24 moved to (4, 2)."


def test_move_one_step_singlegrid_occupied_target():
    model = DummyModel()
    model.grid = SingleGrid(width=5, height=5, torus=False)

    moving_agent = DummyAgent(unique_id=25, model=model)
    blocking_agent = DummyAgent(unique_id=26, model=model)
    model.agents.extend([moving_agent, blocking_agent])
    model.grid.place_agent(moving_agent, (2, 2))
    model.grid.place_agent(blocking_agent, (2, 3))

    result = move_one_step(moving_agent, "North")

    assert moving_agent.pos == (2, 2)
    assert blocking_agent.pos == (2, 3)
    assert "occupied" in result.lower()
    assert "North" in result


def test_move_one_step_boundary_orthogonal_grid():
    """Agent at edge of OrthogonalMooreGrid with no cell in that direction gets a clear message."""

    class _DummyOrthogonalGrid(OrthogonalMooreGrid):
        pass

    orth_grid = object.__new__(_DummyOrthogonalGrid)
    orth_grid.torus = False
    orth_grid.dimensions = (5, 5)
    start = (0, 1)
    start_cell = SimpleNamespace(
        coordinate=start, agents=[], connections={}, is_full=False
    )
    orth_grid._cells = {start: start_cell}

    model = DummyModel()
    model.grid = orth_grid

    agent = DummyAgent(unique_id=22, model=model)
    agent.cell = start_cell
    model.agents.append(agent)

    result = move_one_step(agent, "North")

    # cell should be unchanged
    assert agent.cell is start_cell
    assert "boundary" in result.lower()
    assert "North" in result


def test_move_one_step_boundary_orthogonal_torus_missing_wrapped_cell():
    class _DummyOrthogonalGrid(OrthogonalMooreGrid):
        pass

    orth_grid = object.__new__(_DummyOrthogonalGrid)
    orth_grid.torus = True
    orth_grid.dimensions = (3, 3)
    start = (0, 0)
    start_cell = SimpleNamespace(coordinate=start, agents=[], is_full=False)
    # Wrapped target for North would be (2, 0), but it is intentionally absent.
    orth_grid._cells = {start: start_cell}

    model = DummyModel()
    model.grid = orth_grid

    agent = DummyAgent(unique_id=38, model=model)
    agent.cell = start_cell
    model.agents.append(agent)

    result = move_one_step(agent, "North")

    assert agent.cell is start_cell
    assert "boundary" in result.lower()
    assert "North" in result


def test_move_one_step_full_target_orthogonal_grid():
    class _DummyOrthogonalGrid(OrthogonalMooreGrid):
        pass

    orth_grid = object.__new__(_DummyOrthogonalGrid)
    orth_grid.torus = False
    orth_grid.dimensions = (5, 5)
    start = (1, 1)
    end = (0, 1)
    start_cell = SimpleNamespace(
        coordinate=start, agents=[], connections={}, is_full=False
    )
    full_target_cell = SimpleNamespace(
        coordinate=end,
        agents=[SimpleNamespace(unique_id=99)],
        connections={},
        is_full=True,
    )
    start_cell.connections[(-1, 0)] = full_target_cell
    orth_grid._cells = {start: start_cell, end: full_target_cell}

    model = DummyModel()
    model.grid = orth_grid

    agent = DummyAgent(unique_id=27, model=model)
    agent.cell = start_cell
    model.agents.append(agent)

    result = move_one_step(agent, "North")

    assert agent.cell is start_cell
    assert "full" in result.lower()
    assert "North" in result


def test_move_one_step_diagonal_on_orthogonal_vonneumann_grid():
    class _DummyOrthogonalVonNeumannGrid(OrthogonalVonNeumannGrid):
        pass

    orth_grid = object.__new__(_DummyOrthogonalVonNeumannGrid)
    orth_grid.torus = False
    orth_grid.dimensions = (5, 5)
    start = (2, 2)
    end = (1, 3)  # NorthEast
    start_cell = SimpleNamespace(coordinate=start, agents=[], is_full=False)
    end_cell = SimpleNamespace(coordinate=end, agents=[], is_full=False)
    orth_grid._cells = {start: start_cell, end: end_cell}

    model = DummyModel()
    model.grid = orth_grid

    agent = DummyAgent(unique_id=28, model=model)
    agent.cell = start_cell
    model.agents.append(agent)

    result = move_one_step(agent, "NorthEast")

    assert agent.cell is end_cell
    assert result == "agent 28 moved to (1, 3)."


def test_move_one_step_torus_wrap_orthogonal_grid():
    class _DummyOrthogonalGrid(OrthogonalMooreGrid):
        pass

    orth_grid = object.__new__(_DummyOrthogonalGrid)
    orth_grid.torus = True
    orth_grid.dimensions = (3, 3)
    start = (0, 0)
    end = (2, 2)  # NorthWest wraps on torus
    start_cell = SimpleNamespace(coordinate=start, agents=[], is_full=False)
    wrapped_cell = SimpleNamespace(coordinate=end, agents=[], is_full=False)
    orth_grid._cells = {start: start_cell, end: wrapped_cell}

    model = DummyModel()
    model.grid = orth_grid

    agent = DummyAgent(unique_id=29, model=model)
    agent.cell = start_cell
    model.agents.append(agent)

    result = move_one_step(agent, "NorthWest")

    assert agent.cell is wrapped_cell
    assert result == "agent 29 moved to (2, 2)."


def test_speak_to_skips_non_llm_recipient(mocker):
    """
    speak_to must not crash when a recipient has no memory attribute.

    This covers the case where a non-LLM (rule-based) agent is listed as a
    recipient.
    """
    model = DummyModel()

    sender = DummyAgent(unique_id=1, model=model)
    llm_recipient = DummyAgent(unique_id=2, model=model)
    rule_recipient = DummyAgent(unique_id=3, model=model)

    llm_recipient.memory = SimpleNamespace(add_to_memory=mocker.Mock())

    model.agents = [sender, llm_recipient, rule_recipient]

    ret = speak_to(sender, [2, 3], "Hello both")

    llm_recipient.memory.add_to_memory.assert_called_once()
    call_kwargs = llm_recipient.memory.add_to_memory.call_args[1]
    assert call_kwargs["type"] == "message"
    assert call_kwargs["content"]["message"] == "Hello both"
    assert "recipients" not in call_kwargs["content"]

    assert ret == (
        "sent message 'Hello both' to [2]; skipped [3] because they have no `memory` attribute"
    )


def test_speak_to_warns_for_non_llm_recipient(mocker, caplog):
    model = DummyModel()
    sender = DummyAgent(unique_id=10, model=model)
    rule_recipient = DummyAgent(unique_id=11, model=model)  # no .memory

    model.agents = [sender, rule_recipient]

    with caplog.at_level(logging.WARNING, logger="mesa_llm.tools.inbuilt_tools"):
        ret = speak_to(sender, [11], "Test message")

    assert any(
        "11" in record.message and "memory" in record.message
        for record in caplog.records
    )
    assert ret == "skipped [11] because they have no `memory` attribute"


def test_speak_to_returns_clear_message_when_no_valid_recipients():
    model = DummyModel()
    sender = DummyAgent(unique_id=20, model=model)

    model.agents = [sender]

    ret = speak_to(sender, [20, 999], "Anyone there?")

    assert (
        ret == "Could not send message 'Anyone there?': no matching recipients found."
    )
