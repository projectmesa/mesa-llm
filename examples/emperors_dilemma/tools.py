from typing import TYPE_CHECKING

from mesa_llm.tools.tool_manager import ToolManager
from mesa_llm.tools.tool_decorator import tool

if TYPE_CHECKING:
    from mesa_llm.llm_agent import LLMAgent

emperor_tool_manager = ToolManager()


@tool(tool_manager=emperor_tool_manager)
def set_compliance(agent: "LLMAgent", comply: bool) -> str:
    """
    Sets the agent's public compliance state.
    Call this to decide whether the agent publicly complies with the norm or resists it.

    Args:
        agent: The EmperorLLMAgent making the decision.
        comply: True if the agent publicly complies with the norm, False if they resist.

    Returns:
        str: A confirmation of the compliance decision.
    """
    if comply:
        agent.compliance = 1  # publicly comply with the norm
    else:
        agent.compliance = agent.private_belief  # stay true to private belief

    status = "complying with" if comply else "resisting"
    return f"Agent is now publicly {status} the norm."


@tool(tool_manager=emperor_tool_manager)
def set_enforcement(agent: "LLMAgent", enforce: bool) -> str:
    """
    Sets the agent's enforcement state — whether they actively pressure neighbors to comply.
    Call this after deciding on compliance.

    Args:
        agent: The EmperorLLMAgent making the decision.
        enforce: True if the agent actively enforces the norm on neighbors, False otherwise.

    Returns:
        str: A confirmation of the enforcement decision.
    """
    if enforce:
        agent.enforcement = 1  # enforce the norm on neighbors
    else:
        agent.enforcement = 0  # stay quiet, do not enforce

    status = "enforcing the norm on neighbors" if enforce else "not enforcing"
    return f"Agent is {status}."
