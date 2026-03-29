from typing import TYPE_CHECKING, Any

from mesa.discrete_space import (
    OrthogonalMooreGrid,
    OrthogonalVonNeumannGrid,
)
from mesa.experimental.continuous_space import ContinuousSpace

from mesa_llm.tools.tool_decorator import tool

if TYPE_CHECKING:
    from mesa_llm.llm_agent import LLMAgent

# Mapping directions to (dx, dy) for Cartesian-style spaces.
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


# Mapping directions to (drow, dcol) for mesa.discrete_space orthogonal grids.
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

    position = getattr(agent, "position", None)
    if position is not None:
        return position

    raise ValueError(
        "Could not infer agent position from `cell`, `pos`, or `position`."
    )


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
        row, col = _get_agent_position(agent)
        drow, dcol = direction_map_row_col[direction]
        new_pos = (row + drow, col + dcol)

        if grid.torus:
            dimensions = grid.dimensions
            if len(dimensions) == len(new_pos):
                new_pos = tuple(coord % dim for coord, dim in zip(new_pos, dimensions))
        elif new_pos not in grid._cells:
            return (
                f"Agent {agent.unique_id} is at the boundary and cannot move "
                f"{direction}. Try a different direction."
            )

        target_cell = grid._cells.get(new_pos)
        if target_cell is None:
            return (
                f"Agent {agent.unique_id} is at the boundary and cannot move "
                f"{direction}. Try a different direction."
            )

        if target_cell.is_full:
            return (
                f"Agent {agent.unique_id} cannot move {direction} because "
                "the target cell is full."
            )

        target_coordinates = tuple(target_cell.coordinate)
        return teleport_to_location(agent, target_coordinates)

    space = getattr(agent.model, "space", None)
    if isinstance(space, ContinuousSpace):
        dx, dy = direction_map_xy[direction]
        x, y = _get_agent_position(agent)
        new_pos = (x + dx, y + dy)

        if space.torus:
            new_pos = space.torus_adj(new_pos)
        elif space.out_of_bounds(new_pos):
            return (
                f"Agent {agent.unique_id} is at the boundary and cannot move "
                f"{direction}. Try a different direction."
            )

        target_coordinates = tuple(new_pos)
        return teleport_to_location(agent, target_coordinates)

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

    if isinstance(agent.model.grid, OrthogonalMooreGrid | OrthogonalVonNeumannGrid):
        agent.cell = agent.model.grid._cells[target_coordinates]

    elif isinstance(agent.model.space, ContinuousSpace):
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
