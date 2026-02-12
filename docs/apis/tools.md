# Tools System

The tools system in Mesa-LLM enables agents to interact with their environment and other agents through a structured function-calling interface. Tools represent the concrete actions agents can perform, from basic movement to complex domain-specific behaviors, and are automatically integrated with LLM reasoning through JSON schemas. The tools module provides decorators, managers, and built-in functionality for creating LLM-callable agent actions.

## Usage in Mesa Simulations

```python
from mesa_llm.llm_agent import LLMAgent
from mesa_llm.tools.tool_decorator import tool
from mesa_llm.tools.tool_manager import ToolManager

# Creating custom tools
@tool
def change_state(agent: "LLMAgent", new_state: str) -> str:
   """
   Change the agent's internal state.

   Args:
      agent: The agent whose state to change (provided automatically)
      new_state: The new state value to set

   Returns:
      Confirmation message of the state change
   """
   agent.internal_state.append(f"State changed to: {new_state}")
   return f"Agent {agent.unique_id} state changed to {new_state}"

@tool
def arrest_citizen(agent: "LLMAgent", target_agent_id: int) -> str:
   """
   Arrest a citizen agent if they are active and nearby.

   Args:
      agent: The arresting agent (provided automatically)
      target_agent_id: Unique ID of the citizen to arrest

   Returns:
      Result of the arrest attempt
   """
   target = next((a for a in agent.model.agents if a.unique_id == target_agent_id), None)
   if target and target.state == CitizenState.ACTIVE:
      target.jail_sentence_left = 2.0
      target.state = CitizenState.ARRESTED
      return f"Citizen {target_agent_id} has been arrested"
   return f"Could not arrest citizen {target_agent_id}"

# Agent-specific tool configuration
citizen_tool_manager = ToolManager()
cop_tool_manager = ToolManager()

class Citizen(LLMAgent):
   def __init__(self, model, **kwargs):
      super().__init__(model, **kwargs)
      self.tool_manager = citizen_tool_manager  # Limited tool access

class Cop(LLMAgent):
   def __init__(self, model, **kwargs):
      super().__init__(model, **kwargs)
      self.tool_manager = cop_tool_manager  # Full tool access

# Tool selection in reasoning
def step(self):
   obs = self.generate_obs()
   plan = self.reasoning.plan(
      obs=obs,
      selected_tools=["move_one_step", "change_state"]  # Restrict available tools for LLM calling
   )
   self.apply_plan(plan)
```

## Tool manager

```{eval-rst}
.. automodule:: mesa_llm.tools.tool_manager
   :members:
   :undoc-members:
   :show-inheritance:
```

## Tool decorator

```{eval-rst}
.. automodule:: mesa_llm.tools.tool_decorator
   :members:
   :undoc-members:
   :show-inheritance:
```

## Built-in tools

```{eval-rst}
.. automodule:: mesa_llm.tools.inbuilt_tools
   :members:
   :undoc-members:
   :show-inheritance:
```
