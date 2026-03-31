import asyncio
import concurrent.futures
import inspect
import json
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, ClassVar

from terminal_style import style

from mesa_llm.tools.tool_decorator import _GLOBAL_TOOL_REGISTRY, add_tool_callback

if TYPE_CHECKING:
    from mesa_llm.llm_agent import LLMAgent

logger = logging.getLogger(__name__)


class ToolManager:
    """
    Manager for registering, organizing, and executing LLM-callable tools with per-agent customization. Supports both global tool registration and per-agent tool customization while maintaining a central registry. There can be multiple instances of ToolManager for different group of agents.

    Attributes:
        - tools: A dictionary of tools of the form {tool_name: tool_function}. E.g. {"get_current_weather": get_current_weather}.
        - **instances** (class-level list) - All ToolManager instances for global tool distribution

    Methods:
        - **register(fn)** - Register tool function to this manager
        - **add_tool_to_all(fn)** - Add tool to all ToolManager instances
        - **get_all_tools_schema(selected_tools=None)** → *list[dict]* - Get OpenAI-compatible schemas
        - **call_tools(agent, llm_response)** → *list[dict]* - Execute LLM-recommended tools
        - **has_tool(name)** → *bool* - Check if tool is registered

    Tool Execution Flow:
        1. **Tool Registration**: Functions decorated with `@tool` are automatically registered in the global registry
        2. **Schema Generation**: Tool decorators analyze function signatures and docstrings to create function calling schemas
        3. **LLM Integration**: Reasoning strategies receive tool schemas and can request specific tool calls
        4. **Argument Validation**: ToolManager validates LLM-provided arguments against function signatures with automatic type coercion
        5. **Execution**: Tools are called with validated arguments, including automatic agent parameter injection
        6. **Result Handling**: Tool outputs are captured and added to agent memory for future reasoning
    """

    instances: ClassVar[list["ToolManager"]] = []

    def __init__(self, extra_tools: dict[str, Callable] | None = None):
        # start from everything that was decorated
        ToolManager.instances.append(self)
        self.tools = dict(_GLOBAL_TOOL_REGISTRY)
        # allow per-agent overrides / reductions
        if extra_tools:
            self.tools.update(extra_tools)

    def register(self, fn: Callable):
        """Register a tool function by name"""
        name = fn.__name__
        self.tools[name] = fn  # storing the name & function pair as a dictionary

    @classmethod
    def add_tool_to_all(cls, fn: Callable):
        """Add a tool to all instances"""
        for instance in cls.instances:
            instance.register(fn)

    def get_tool_schema(self, fn: Callable, schema_name: str) -> dict:
        return getattr(fn, "__tool_schema__", None) or {
            "error": f"Tool {schema_name} missing __tool_schema__"
        }

    def get_all_tools_schema(
        self, selected_tools: list[str] | None = None
    ) -> list[dict]:
        if selected_tools:
            selected_tools_schema = [
                self.tools[tool].__tool_schema__ for tool in selected_tools
            ]
            return selected_tools_schema

        else:
            return [fn.__tool_schema__ for fn in self.tools.values()]

    def call(self, name: str, arguments: dict) -> str:
        """Call a registered tool with validated args"""
        if name not in self.tools:
            raise ValueError(style(f"Tool '{name}' not found", color="red"))
        return self.tools[name](**arguments)

    def has_tool(self, name: str) -> bool:
        return name in self.tools

    async def _process_tool_call(
        self, agent: "LLMAgent", tool_call: Any, index: int
    ) -> dict:
        """
        Internal helper to process a single tool call consistently.
        Supports both synchronous and asynchronous tool functions.
        """

        # Safe extraction
        function_obj = getattr(tool_call, "function", None)
        function_name = getattr(function_obj, "name", "unknown")
        tool_call_id = getattr(tool_call, "id", "unknown")
        raw_args = getattr(function_obj, "arguments", "{}")

        try:
            # Validate tool existence
            if function_name not in self.tools:
                raise ValueError(
                    style(
                        f"Function '{function_name}' not found in ToolManager",
                        color="red",
                    )
                )

            # Parse JSON arguments safely
            try:
                function_args = json.loads(raw_args or "{}")
            except json.JSONDecodeError as e:
                raise ValueError(
                    style(f"Invalid JSON in function arguments: {e}", color="red")
                ) from e

            function_to_call = self.tools[function_name]

            # Inspect signature BEFORE calling
            sig = inspect.signature(function_to_call)
            expects_agent = "agent" in sig.parameters

            # Filter arguments to only those accepted
            filtered_args = {
                k: v for k, v in function_args.items() if k in sig.parameters
            }

            if expects_agent:
                filtered_args["agent"] = agent

            # Execute (sync or async)
            if inspect.iscoroutinefunction(function_to_call):
                function_response = await function_to_call(**filtered_args)
            else:
                function_response = function_to_call(**filtered_args)

            # Only treat None as empty
            if function_response is None:
                function_response = f"{function_name} executed successfully"

            return {
                "tool_call_id": tool_call_id,
                "role": "tool",
                "name": function_name,
                "response": str(function_response),
            }

        except Exception as e:
            logger.exception(
                "Error executing tool call %s (%s): %s",
                index + 1,
                function_name,
                e,
            )
            return {
                "tool_call_id": tool_call_id,
                "role": "tool",
                "name": function_name,
                "response": f"Error: {e!s}",
            }

    def call_tools(self, agent: "LLMAgent", llm_response: Any) -> list[dict]:
        """
        Synchronous tool execution with safe async bridge.
        """

        tool_calls = getattr(llm_response, "tool_calls", [])
        if not tool_calls:
            return []

        async def _run_all():
            tasks = [
                self._process_tool_call(agent, tc, i) for i, tc in enumerate(tool_calls)
            ]
            return await asyncio.gather(*tasks)

        try:
            return asyncio.run(_run_all())
        except RuntimeError:
            # Fallback if event loop already running
            with concurrent.futures.ThreadPoolExecutor() as executor:
                return executor.submit(lambda: asyncio.run(_run_all())).result()

    async def acall_tools(self, agent: "LLMAgent", llm_response: Any) -> list[dict]:
        """
        Asynchronous tool execution (parallel via asyncio.gather).
        """

        tool_calls = getattr(llm_response, "tool_calls", [])
        if not tool_calls:
            return []

        tasks = [
            self._process_tool_call(agent, tc, i) for i, tc in enumerate(tool_calls)
        ]

        return await asyncio.gather(*tasks)


# Register callback to automatically add new tools to all ToolManager instances
add_tool_callback(ToolManager.add_tool_to_all)
