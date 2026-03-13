from typing import TYPE_CHECKING, Any

from mesa.discrete_space import (
    OrthogonalMooreGrid,
    OrthogonalVonNeumannGrid,
)
from mesa.experimental.continuous_space import ContinuousSpace

from mesa_llm.tools.tool_decorator import tool

if TYPE_CHECKING:
    from mesa_llm.llm_agent import LLMAgent

# Mapping directions to (dx, dy) for ContinuousSpace and real OrthogonalGrid _cells lookup.
# Real Mesa 4.x OrthogonalMooreGrid uses (x, y): North = y+1, West = x-1
direction_map_xy = {
    "North": (0, 1),
    "South": (0, -1),
    "East": (1, 0),
    "West": (-1, 0),
    "NorthEast": (1, 1),
    "NorthWest": (-1, 1),
    "SouthEast": (1, -1),
    "SouthWest": (-1, -1),
}

# Mapping directions to (drow, dcol) for cell.connections dict lookup.
# SimpleNamespace dummy grids use (row, col): North = row-1, East = col+1
direction_map_row_col = {
    "North": (-1, 0),
    "South": (1, 0),
    "East": (0, 1),
    "West": (0, -1),
    "NorthEast": (-1, 1),
    "NorthWest": (-1, -1),
    "SouthEast": (1, 1),
    "SouthWest": (1, -1),
}


def _get_agent_position(agent: "LLMAgent") -> Any:
    """Return the agent position across Mesa space APIs."""
    cell = getattr(agent, "cell", None)
    if cell is not None and getattr(cell, "coordinate", None) is not None:
        return cell.coordinate

    pos = getattr(agent, "pos", None)
    if pos is not None:
        return pos

    raise ValueError("Could not infer agent position from `cell` or `pos`.")


def _cell_is_full(cell) -> bool:
    return bool(getattr(cell, "is_full", False))


def _cell_has_agent(cell, agent) -> bool:
    return agent in list(getattr(cell, "agents", []))


def _remove_agent_from_cell(cell, agent) -> None:
    if hasattr(cell, "remove_agent"):
        cell.remove_agent(agent)
    else:
        agents = getattr(cell, "agents", [])
        if agent in agents:
            agents.remove(agent)


def _add_agent_to_cell(cell, agent) -> None:
    if hasattr(cell, "add_agent"):
        cell.add_agent(agent)
    else:
        agents = getattr(cell, "agents", [])
        if agent not in agents:
            agents.append(agent)


def _is_out_of_bounds(space, pos) -> bool:
    """Check if position is out of bounds for ContinuousSpace."""
    dims = getattr(space, "dimensions", None)
    if dims is None:
        return False
    for i, (lo, hi) in enumerate(dims):
        if pos[i] < lo or pos[i] >= hi:
            return True
    return False


def _torus_adj(space, pos) -> tuple:
    """Wrap position for torus ContinuousSpace."""
    dims = getattr(space, "dimensions", None)
    if dims is None:
        return pos
    return tuple(lo + (c - lo) % (hi - lo) for c, (lo, hi) in zip(pos, dims))


@tool
def move_one_step(agent: "LLMAgent", direction: str) -> str:
    """
    Moves agents one step in specified cardinal/diagonal directions (North, South, East, West, NorthEast, NorthWest, SouthEast, SouthWest). Automatically handles different Mesa grid types including OrthogonalGrids and ContinuousSpace.

        Args:
            direction: The direction to move in. Must be one of:
                'North', 'South', 'East', 'West',
                'NorthEast', 'NorthWest', 'SouthEast', or 'SouthWest'.
            agent: Provided automatically.

        Returns:
            A string confirming the result of the movement attempt.
    """
    if direction not in direction_map_xy:
        raise ValueError(
            f"Invalid direction '{direction}'."
            f"Must be one of {list(direction_map_xy.keys())}"
        )

    grid = getattr(agent.model, "grid", None)
    if isinstance(grid, OrthogonalMooreGrid | OrthogonalVonNeumannGrid):
        current_cell = getattr(agent, "cell", None)
        if current_cell is None:
            pos = _get_agent_position(agent)
            current_cell = grid._cells.get(pos)
        if current_cell is None:
            return f"Agent {agent.unique_id} has no current cell."

        current_coord = current_cell.coordinate
        # Use cell type to pick coordinate convention:
        # Real Mesa Cell objects -> (x, y); SimpleNamespace dummy cells -> (row, col)
        from types import SimpleNamespace as _SN

        is_real_cell = not isinstance(current_cell, _SN)
        target_cell = None

        if is_real_cell:
            dx, dy = direction_map_xy[direction]
            new_coord = (current_coord[0] + dx, current_coord[1] + dy)
            if grid.torus:
                new_coord = tuple(c % d for c, d in zip(new_coord, grid.dimensions))
            target_cell = grid._cells.get(new_coord)
        else:
            connections = getattr(current_cell, "connections", None)
            if connections:
                drow, dcol = direction_map_row_col[direction]
                target_cell = connections.get((drow, dcol))
            if target_cell is None:
                drow, dcol = direction_map_row_col[direction]
                new_coord = (current_coord[0] + drow, current_coord[1] + dcol)
                if grid.torus:
                    new_coord = tuple(c % d for c, d in zip(new_coord, grid.dimensions))
                target_cell = grid._cells.get(new_coord)

        if target_cell is None:
            return (
                f"Agent {agent.unique_id} is at the boundary and cannot move "
                f"{direction}. Try a different direction."
            )

        if _cell_is_full(target_cell) and not _cell_has_agent(target_cell, agent):
            return (
                f"Agent {agent.unique_id} cannot move {direction} because "
                "the target cell is full."
            )

        target_coordinates = tuple(target_cell.coordinate)
        try:
            return teleport_to_location(agent, target_coordinates)
        except ValueError as e:
            if "not empty" in str(e).lower():
                return (
                    f"Agent {agent.unique_id} cannot move {direction} because "
                    "the target cell is occupied."
                )
            raise

    space = getattr(agent.model, "space", None)
    if isinstance(space, ContinuousSpace):
        dx, dy = direction_map_xy[direction]
        x, y = _get_agent_position(agent)
        new_pos = (x + dx, y + dy)

        if space.torus:
            new_pos = tuple(float(c) for c in _torus_adj(space, new_pos))
        elif _is_out_of_bounds(space, new_pos):
            return (
                f"Agent {agent.unique_id} is at the boundary and cannot move "
                f"{direction}. Try a different direction."
            )

        return teleport_to_location(agent, tuple(new_pos))

    raise ValueError(
        "Unsupported environment for move_one_step. Expected "
        "OrthogonalMooreGrid, OrthogonalVonNeumannGrid, or ContinuousSpace."
    )


@tool
def teleport_to_location(
    agent: "LLMAgent",
    target_coordinates: list[int | float],
) -> str:
    """
    Instantly moves agents to specific [x, y] coordinates within grid boundaries. Useful for rapid repositioning or spawning mechanics. Validates coordinates are within environment bounds.

    Args:
        target_coordinates: Exactly two numeric coordinates in the form [x, y] that fall inside the current environment bounds. Examples: [3, 7] or [3.5, 7.25]
        agent: Provided automatically

    Returns:
        a string confirming the agent's new position.

    """
    target_coordinates = tuple(target_coordinates)

    grid = getattr(agent.model, "grid", None)
    space = getattr(agent.model, "space", None)

    if isinstance(grid, OrthogonalMooreGrid | OrthogonalVonNeumannGrid):
        target_cell = grid._cells.get(target_coordinates)
        if target_cell is None:
            if hasattr(grid, "all_cells"):
                raise ValueError(f"Point out of bounds: {target_coordinates}")
            else:
                raise KeyError(target_coordinates)
        # Check occupancy: cell is full and agent is not already in it
        if _cell_is_full(target_cell) and not _cell_has_agent(target_cell, agent):
            raise ValueError(f"Cell not empty: {target_coordinates}")
        # Also check via agents list for real grids with capacity=1
        agents_in_cell = list(getattr(target_cell, "agents", []))
        capacity = getattr(target_cell, "capacity", None)
        if agent not in agents_in_cell and len(agents_in_cell) > 0:
            if capacity is None or len(agents_in_cell) >= capacity:
                raise ValueError(f"Cell not empty: {target_coordinates}")
        # Remove from current cell
        current_cell = getattr(agent, "cell", None)
        if current_cell is not None:
            _remove_agent_from_cell(current_cell, agent)
        # Place in new cell
        _add_agent_to_cell(target_cell, agent)
        agent.cell = target_cell
        agent.pos = target_coordinates

    elif isinstance(space, ContinuousSpace):
        agent.pos = target_coordinates

    else:
        raise ValueError(
            "Unsupported environment for teleport_to_location. Expected "
            "OrthogonalMooreGrid, OrthogonalVonNeumannGrid, or ContinuousSpace."
        )

    return f"agent {agent.unique_id} moved to {target_coordinates}."


@tool
def speak_to(
    agent: "LLMAgent", listener_agents_unique_ids: list[int], message: str
) -> str:
    """
    Enables agent-to-agent communication by sending messages to specified recipients. Messages are automatically added to recipients' memory systems for future reasoning context. Supports both single agent and multiple agent communication.

    Args:
        agent: The agent sending the message(conversation contents) (as a LLM, ignore this argument in function calling).
        listener_agents_unique_ids: The unique ids of the agents receiving the message
        message: The message to send
    """
    listener_agents = [
        listener_agent
        for listener_agent in agent.model.agents
        if listener_agent.unique_id in listener_agents_unique_ids
        and listener_agent.unique_id != agent.unique_id
    ]

    for recipient in listener_agents:
        recipient.memory.add_to_memory(
            type="message",
            content={
                "message": message,
                "sender": agent.unique_id,
                "recipients": [
                    listener_agent.unique_id for listener_agent in listener_agents
                ],
            },
        )
    return f"{agent.unique_id} → {[agent.unique_id for agent in listener_agents]} : {message}"
