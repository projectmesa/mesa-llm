# Tools System

The tools system in Mesa-LLM exposes explicit LLM-callable helper functions through JSON schemas. Tool exposure is opt-in: `LLMAgent(..., tools=None)` and `LLMAgent(..., tools=[])` expose no tools. Pass explicit callables or a tool-set factory when a simulation should expose tools.

Current built-ins such as `move_one_step`, `teleport_to_location`, and `speak_to` are available only through explicit configuration such as `legacy_tools()`.

## Tool Sets

Tool sets are factory functions that return tuples. They are preferred over mutable constants and live in `mesa_llm.tools.defaults`.

- `default_tools()` returns exactly `()` because no safe read-only built-ins are available yet.
- `legacy_tools()` returns exactly `(move_one_step, teleport_to_location, speak_to)` for simulations migrating from old implicit built-ins.
- `math_tools()` returns `()`.
- `spatial_tools()` returns `()`.
- `environment_tools()` returns `()`.
- `social_query_tools()` returns `()`.
- `external_tools()` returns `()`.

`selected_tools` remains only as a deprecated compatibility alias for one transition window. New code should use `tools=`.

`agent.tool_manager` remains only as a deprecated compatibility property. New code should configure tools with `LLMAgent(..., tools=...)` or pass per-call overrides with `reasoning.plan(..., tools=...)`.

When working with a `ToolManager` directly, use `get_tools_schema(...)` to
retrieve schemas for the configured tool set or a narrowed `tools=` selection.
The selector accepts omitted `tools` for configured tools, `tools=None` or
`tools=[]` for no tools, or a configured callable, string name, list, or tuple.
`get_all_tools_schema(...)` and `get_tool_schema(...)` are deprecated
compatibility aliases.

## Usage in Mesa Simulations

```python
from mesa_llm.llm_agent import LLMAgent
from mesa_llm.reasoning.cot import CoTReasoning
from mesa_llm.tools.tool_decorator import tool

@tool
def inspect_status(agent: "LLMAgent") -> str:
   """
   Inspect the agent status.

   Args:
      agent: The agent making the request (provided automatically)

   Returns:
      A compact status string.
   """
   return f"Agent {agent.unique_id} is at {getattr(agent, 'pos', None)}"

class MyAgent(LLMAgent):
   def __init__(self, model, **kwargs):
      super().__init__(
         model=model,
         reasoning=CoTReasoning,
         tools=[inspect_status],
         **kwargs,
      )

   def step(self):
      obs = self.generate_obs()
      plan = self.reasoning.plan(obs=obs)
      self.apply_plan(plan)
```

Per-call `tools=` narrows the agent's configured tools. Omitting `tools` inherits the agent's configured tools; explicit `tools=None` or `tools=[]` exposes no tools for that call. A single configured callable or string name is accepted, as is a list or tuple of configured callables or names.

```python
plan = self.reasoning.plan(obs=obs, tools=[inspect_status])
plan_without_tools = self.reasoning.plan(obs=obs, tools=None)
```

Per-call selections are also used for execution of the returned plan. Any tool passed as `tools=[inspect_status]` must already be configured on the agent or manager; per-call `tools=[...]` cannot add unconfigured tools.

To migrate code that relied on the old implicit built-ins, opt in explicitly:

```python
from mesa_llm.tools.defaults import legacy_tools

agent = LLMAgent(model, reasoning=CoTReasoning, tools=legacy_tools())
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
