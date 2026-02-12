import inspect
import json
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from terminal_style import sprint, style

from mesa_llm.tools.tool_decorator import _GLOBAL_TOOL_REGISTRY, add_tool_callback

if TYPE_CHECKING:
    from mesa_llm.llm_agent import LLMAgent


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

    instances: list["ToolManager"] = []

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

    def call_tools(self, agent: "LLMAgent", llm_response: Any) -> list[dict]:
        """
        Calls the tools, recommended by the LLM. If the tool has an output it returns the name of the tool and the output else, it returns the name
        and output as successfully executed.

        Args:
            llm_response: The raw response from the LLM.

        Returns:
            A list of tool results
        """

        try:
            # Extract response message and tool calls
            tool_calls = llm_response.tool_calls

            # Check if tool_calls exists and is not None
            if not tool_calls:
                sprint("No tool calls in LLM response", color="red")
                return []

            tool_results = []

            # Process each tool call
            for i, tool_call in enumerate(tool_calls):
                try:
                    # Extract function details
                    function_name = tool_call.function.name
                    function_args_str = tool_call.function.arguments
                    tool_call_id = tool_call.id

                    # Validate function exists in tool_manager
                    if function_name not in self.tools:
                        raise ValueError(
                            style(
                                f"Function '{function_name}' not found in ToolManager",
                                color="red",
                            )
                        )

                    # Parse function arguments
                    try:
                        function_args = json.loads(function_args_str)
                    except json.JSONDecodeError as e:
                        raise json.JSONDecodeError(
                            style(
                                f"Invalid JSON in function arguments: {e}", color="red"
                            )
                        ) from e

                    # Get the actual function to call from tool_manager
                    function_to_call = self.tools[function_name]

                    # Call the function with unpacked arguments
                    try:
                        function_response = function_to_call(
                            agent=agent, **function_args
                        )
                    except TypeError as e:
                        # If function arguments don't match function signature :
                        sprint(
                            f"Warning: Function call failed with TypeError: {e}",
                            color="yellow",
                        )
                        sprint(
                            "Attempting to call with filtered arguments...",
                            color="yellow",
                        )

                        # Try to filter arguments to match function signature

                        sig = inspect.signature(function_to_call)
                        expects_agent = "agent" in sig.parameters
                        filtered_args = {
                            k: v
                            for k, v in function_args.items()
                            if k in sig.parameters
                        }

                        if expects_agent:
                            function_response = function_to_call(
                                agent=agent, **filtered_args
                            )
                        else:
                            function_response = function_to_call(**filtered_args)

                    if not function_response:
                        function_response = f"{function_name} executed successfully"

                    # Create tool result message
                    tool_result = {
                        "tool_call_id": tool_call_id,
                        "role": "tool",
                        "name": function_name,
                        "response": str(function_response),
                    }

                    tool_results.append(tool_result)

                except Exception as e:
                    # Handle individual tool call errors
                    sprint(
                        f"Error executing tool call {i + 1} ({function_name}): {e!s}",
                        color="red",
                    )

                    # Create error response
                    error_result = {
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "response": f"Error: {e!s}",
                    }

                    tool_results.append(error_result)
            return tool_results

        except AttributeError as e:
            sprint(f"Error accessing LLM response structure: {e}", color="red")
            return []
        except Exception as e:
            sprint(f"Unexpected error in call_tools: {e}", color="red")
            return []


# Register callback to automatically add new tools to all ToolManager instances
add_tool_callback(ToolManager.add_tool_to_all)
