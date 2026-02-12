from typing import TYPE_CHECKING

from mesa.discrete_space import (
    OrthogonalMooreGrid,
    OrthogonalVonNeumannGrid,
)
from mesa.space import (
    ContinuousSpace,
    MultiGrid,
    SingleGrid,
)

from mesa_llm.tools.tool_decorator import tool

if TYPE_CHECKING:
    from mesa_llm.llm_agent import LLMAgent

# Mapping directions to (dx, dy)
direction_map = {
    "North": (0, 1),
    "South": (0, -1),
    "East": (1, 0),
    "West": (-1, 0),
    "NorthEast": (1, 1),
    "NorthWest": (-1, 1),
    "SouthEast": (1, -1),
    "SouthWest": (-1, -1),
}


@tool
def move_one_step(agent: "LLMAgent", direction: str) -> str:
    """
    Moves agents one step in specified cardinal/diagonal directions (North, South, East, West, NorthEast, NorthWest, SouthEast, SouthWest). Automatically handles different Mesa grid types including SingleGrid, MultiGrid, OrthogonalGrids, and ContinuousSpace.

        Args:
            direction: The direction to move in. Must be one of:
                'North', 'South', 'East', 'West',
                'NorthEast', 'NorthWest', 'SouthEast', or 'SouthWest'.
            agent: Provided automatically.

        Returns:
            A string confirming the result of the movement attempt.
    """
    if direction not in direction_map:
        raise ValueError(
            f"Invalid direction '{direction}'."
            f"Must be one of {list(direction_map.keys())}"
        )
    dx, dy = direction_map[direction]
    x, y = agent.pos
    new_pos = (x + dx, y + dy)
    target_coordinates = tuple(new_pos)
    teleport_to_location(agent, target_coordinates)
    return f"agent {agent.unique_id} moved to {target_coordinates}."


@tool
def teleport_to_location(
    agent: "LLMAgent",
    target_coordinates: list[int],
) -> str:
    """
    Instantly moves agents to specific [x, y] coordinates within grid boundaries. Useful for rapid repositioning or spawning mechanics. Validates coordinates are within environment bounds.

    Args:
        target_coordinates: Exactly two integers in the form [x, y] that fall inside the current environment bounds. Example: [3, 7]
        agent: Provided automatically

    Returns:
        a string confirming the agent's new position.

    """
    target_coordinates = tuple(target_coordinates)

    if isinstance(agent.model.grid, SingleGrid | MultiGrid):
        agent.model.grid.move_agent(agent, target_coordinates)

    elif isinstance(agent.model.grid, OrthogonalMooreGrid | OrthogonalVonNeumannGrid):
        cell = agent.model.grid._cells[target_coordinates]
        agent.cell = cell

    elif isinstance(agent.model.space, ContinuousSpace):
        agent.model.space.move_agent(agent.model.space, agent, target_coordinates)

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
