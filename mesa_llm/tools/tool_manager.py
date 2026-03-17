import asyncio
import concurrent.futures
import inspect
import json
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from terminal_style import style

from mesa_llm.tools.tool_decorator import _GLOBAL_TOOL_REGISTRY, add_tool_callback

if TYPE_CHECKING:
    from mesa_llm.llm_agent import LLMAgent

logger = logging.getLogger(__name__)


class ToolExecutionError(Exception):
    """Custom exception for tool execution failures."""

    def __init__(
        self,
        message: str,
        tool_name: str | None = None,
        original_error: Exception | None = None,
    ):
        super().__init__(message)
        self.tool_name = tool_name
        self.original_error = original_error


class ToolValidationError(Exception):
    """Custom exception for tool validation failures."""

    def __init__(
        self,
        message: str,
        validation_type: str | None = None,
        field_name: str | None = None,
    ):
        super().__init__(message)
        self.validation_type = validation_type
        self.field_name = field_name


class ToolTimeoutError(Exception):
    """Custom exception for tool timeout failures."""

    def __init__(
        self,
        message: str,
        tool_name: str | None = None,
        timeout_duration: float | None = None,
    ):
        super().__init__(message)
        self.tool_name = tool_name
        self.timeout_duration = timeout_duration


class ToolManager:
    """
    Manager for registering, organizing, and executing LLM-callable tools with per-agent customization. Supports both global tool registration and per-agent tool customization while maintaining a central registry. There can be multiple instances of ToolManager for different group of agents.

    Attributes:
        - tools: A dictionary of tools of the form {tool_name: tool_function}. E.g. {"get_current_weather": get_current_weather}.
        - **instances** (class-level list) - All ToolManager instances for global tool distribution
        - execution_stats: Statistics for tool execution monitoring

    Methods:
        - **register(fn)** - Register tool function to this manager
        - **add_tool_to_all(fn)** - Add tool to all ToolManager instances
        - **get_all_tools_schema(selected_tools=None)** → *list[dict]* - Get OpenAI-compatible schemas
        - **call_tools(agent, llm_response)** → *list[dict]* - Execute LLM-recommended tools
        - **has_tool(name)** → *bool* - Check if tool is registered

    Tool Execution Flow:
        1. **Tool Registration**: Functions decorated with `@tool` are automatically registered in global registry
        2. **Schema Generation**: Tool decorators analyze function signatures and docstrings to create function calling schemas
        3. **LLM Integration**: Reasoning strategies receive tool schemas and can request specific tool calls
        4. **Argument Validation**: ToolManager validates LLM-provided arguments against function signatures with automatic type coercion
        5. **Execution**: Tools are called with validated arguments, including automatic agent parameter injection
        6. **Result Handling**: Tool outputs are captured and added to agent memory for future reasoning
        7. **Error Isolation**: Each tool execution is isolated to prevent cascade failures
    """

    instances: list["ToolManager"] = []

    def __init__(
        self, extra_tools: dict[str, Callable] | None = None, tool_timeout: float = 30.0
    ):
        # start from everything that was decorated
        ToolManager.instances.append(self)
        self.tools = dict(_GLOBAL_TOOL_REGISTRY)
        # allow per-agent overrides / reductions
        if extra_tools:
            self.tools.update(extra_tools)

        # Error handling and monitoring
        self.tool_timeout = tool_timeout
        self.execution_stats = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "timeout_errors": 0,
            "validation_errors": 0,
            "execution_errors": 0,
        }

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
        Internal helper to process a single tool call with comprehensive error isolation.
        Supports both synchronous and asynchronous tool functions with timeout protection.
        """

        # Safe extraction
        function_obj = getattr(tool_call, "function", None)
        function_name = getattr(function_obj, "name", "unknown")
        tool_call_id = getattr(tool_call, "id", "unknown")
        raw_args = getattr(function_obj, "arguments", "{}")

        # Update execution stats
        self.execution_stats["total_calls"] += 1

        try:
            # Validate tool existence
            if function_name not in self.tools:
                self.execution_stats["validation_errors"] += 1
                raise ToolValidationError(
                    style(
                        f"Function '{function_name}' not found in ToolManager",
                        color="red",
                    ),
                    validation_type="tool_existence",
                    field_name="function_name",
                )

            # Parse JSON arguments safely
            try:
                function_args = json.loads(raw_args or "{}")
            except json.JSONDecodeError as e:
                self.execution_stats["validation_errors"] += 1
                raise ToolValidationError(
                    style(f"Invalid JSON in function arguments: {e}", color="red"),
                    validation_type="json_parsing",
                    field_name="arguments",
                ) from e

            # Validate argument types against function signature
            function_to_call = self.tools[function_name]
            sig = inspect.signature(function_to_call)
            expects_agent = "agent" in sig.parameters

            # Filter arguments to only those accepted
            filtered_args = {
                k: v for k, v in function_args.items() if k in sig.parameters
            }

            if expects_agent:
                filtered_args["agent"] = agent

            # Execute with timeout protection
            try:
                if inspect.iscoroutinefunction(function_to_call):
                    function_response = await asyncio.wait_for(
                        function_to_call(**filtered_args), timeout=self.tool_timeout
                    )
                else:
                    # Run sync function in executor to avoid blocking
                    loop = asyncio.get_running_loop()
                    function_response = await loop.run_in_executor(
                        None, lambda: function_to_call(**filtered_args)
                    )
            except TimeoutError:
                self.execution_stats["timeout_errors"] += 1
                raise ToolTimeoutError(
                    style(
                        f"Tool '{function_name}' timed out after {self.tool_timeout}s",
                        color="red",
                    ),
                    tool_name=function_name,
                    timeout_duration=self.tool_timeout,
                ) from None
            except Exception as e:
                # Catch any other execution errors
                self.execution_stats["execution_errors"] += 1
                raise ToolExecutionError(
                    style(f"Tool '{function_name}' execution failed: {e}", color="red"),
                    tool_name=function_name,
                    original_error=e,
                ) from e

            # Only treat None as empty
            if function_response is None:
                function_response = f"{function_name} executed successfully"

            # Update success stats
            self.execution_stats["successful_calls"] += 1

            return {
                "tool_call_id": tool_call_id,
                "role": "tool",
                "name": function_name,
                "response": str(function_response),
                "success": True,
            }

        except (ToolValidationError, ToolTimeoutError, ToolExecutionError) as e:
            self.execution_stats["failed_calls"] += 1
            if isinstance(e, ToolValidationError):
                logger.error(
                    "Tool validation error for %s (%s): %s [type: %s, field: %s]",
                    index + 1,
                    function_name,
                    e,
                    getattr(e, "validation_type", "unknown"),
                    getattr(e, "field_name", "unknown"),
                )
            elif isinstance(e, ToolTimeoutError):
                logger.error(
                    "Tool timeout error for %s (%s): %s [tool: %s, duration: %s]",
                    index + 1,
                    function_name,
                    e,
                    getattr(e, "tool_name", "unknown"),
                    getattr(e, "timeout_duration", "unknown"),
                )
            elif isinstance(e, ToolExecutionError):
                logger.error(
                    "Tool execution error for %s (%s): %s [tool: %s, original: %s]",
                    index + 1,
                    function_name,
                    e,
                    getattr(e, "tool_name", "unknown"),
                    getattr(e, "original_error", e),
                )

            return {
                "tool_call_id": tool_call_id,
                "role": "tool",
                "name": function_name,
                "response": f"Error: {e!s}",
                "success": False,
                "error_type": type(e).__name__,
                "error_details": {
                    "validation_type": getattr(e, "validation_type", None),
                    "field_name": getattr(e, "field_name", None),
                    "tool_name": getattr(e, "tool_name", None),
                    "timeout_duration": getattr(e, "timeout_duration", None),
                    "original_error": str(getattr(e, "original_error", None)),
                },
            }

        except Exception as e:
            if not isinstance(
                e, (ToolValidationError, ToolTimeoutError, ToolExecutionError)
            ):
                self.execution_stats["failed_calls"] += 1
                self.execution_stats["execution_errors"] += 1
            logger.exception(
                "Unexpected error executing tool call %s (%s): %s",
                index + 1,
                function_name,
                e,
            )
            return {
                "tool_call_id": tool_call_id,
                "role": "tool",
                "name": function_name,
                "response": f"Unexpected error: {e!s}",
                "success": False,
                "error_type": "UnexpectedError",
            }

    def get_execution_stats(self) -> dict:
        """Get comprehensive tool execution statistics."""
        stats = self.execution_stats.copy()
        if stats["total_calls"] > 0:
            stats["success_rate"] = stats["successful_calls"] / stats["total_calls"]
        else:
            stats["success_rate"] = 0.0
        return stats

    def reset_execution_stats(self) -> None:
        """Reset execution statistics."""
        self.execution_stats = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "timeout_errors": 0,
            "validation_errors": 0,
            "execution_errors": 0,
        }

    async def _execute_tools_with_isolation(
        self, agent: "LLMAgent", tool_calls: list
    ) -> list[dict]:
        """
        Execute tools with complete error isolation - one tool failure doesn't affect others.
        """
        if not tool_calls:
            return []

        # Execute all tools in parallel with individual error isolation
        tasks = [
            self._process_tool_call(agent, tc, i) for i, tc in enumerate(tool_calls)
        ]

        # Use return_exceptions=True to ensure all tools complete regardless of individual failures
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results, converting any exceptions to error responses
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Convert unexpected exceptions to error responses
                processed_results.append(
                    {
                        "tool_call_id": getattr(tool_calls[i], "id", "unknown"),
                        "role": "tool",
                        "name": getattr(
                            getattr(tool_calls[i], "function", {}), "name", "unknown"
                        ),
                        "response": f"Critical error: {result!s}",
                        "success": False,
                        "error_type": "CriticalError",
                    }
                )
            else:
                processed_results.append(result)

        return processed_results

    def call_tools(self, agent: "LLMAgent", llm_response: Any) -> list[dict]:
        """
        Synchronous tool execution with safe async bridge and error isolation.
        """

        tool_calls = getattr(llm_response, "tool_calls", [])
        if not tool_calls:
            return []

        async def _run_all():
            return await self._execute_tools_with_isolation(agent, tool_calls)

        try:
            return asyncio.run(_run_all())
        except RuntimeError:
            # Fallback if event loop already running
            with concurrent.futures.ThreadPoolExecutor() as executor:
                return executor.submit(lambda: asyncio.run(_run_all())).result()

    async def acall_tools(self, agent: "LLMAgent", llm_response: Any) -> list[dict]:
        """
        Asynchronous tool execution with complete error isolation.
        """

        tool_calls = getattr(llm_response, "tool_calls", [])
        if not tool_calls:
            return []

        return await self._execute_tools_with_isolation(agent, tool_calls)


# Register callback to automatically add new tools to all ToolManager instances
add_tool_callback(ToolManager.add_tool_to_all)
