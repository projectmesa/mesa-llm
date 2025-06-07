from unittest.mock import Mock

import pytest

from mesa_llm.tools.tool_decorator import _GLOBAL_TOOL_REGISTRY, tool
from mesa_llm.tools.tool_manager import ToolManager


class TestToolManager:
    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Clear global registry to start fresh
        _GLOBAL_TOOL_REGISTRY.clear()
        # Clear instances list
        ToolManager.instances.clear()

    def teardown_method(self):
        """Clean up after each test method."""
        _GLOBAL_TOOL_REGISTRY.clear()
        ToolManager.instances.clear()

    def test_init_empty(self):
        """Test initialization with no tools."""
        manager = ToolManager()
        assert isinstance(manager.tools, dict)
        assert len(manager.tools) == 0
        assert manager in ToolManager.instances

    def test_init_with_global_tools(self):
        """Test initialization with global tools."""

        # Register a tool globally first
        @tool
        def test_global_tool(param1: str) -> str:
            """Test global tool.
            Args:
                param1: A test parameter.
            Returns:
                The input parameter.
            """
            return param1

        manager = ToolManager()
        assert "test_global_tool" in manager.tools
        assert manager.tools["test_global_tool"] == test_global_tool

    def test_init_with_extra_tools(self):
        """Test initialization with extra tools."""

        def extra_tool(x: int) -> int:
            """Extra tool.
            Args:
                x: Input number.
            Returns:
                The input number.
            """
            return x

        extra_tool.__tool_schema__ = {
            "type": "function",
            "function": {
                "name": "extra_tool",
                "description": "Extra tool returns: The input number.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "x": {"type": "integer", "description": "Input number."}
                    },
                    "required": ["x"],
                },
            },
        }

        extra_tools = {"extra_tool": extra_tool}
        manager = ToolManager(extra_tools=extra_tools)
        assert "extra_tool" in manager.tools

    def test_register_tool(self):
        """Test registering a tool manually."""
        manager = ToolManager()

        def manual_tool(text: str) -> str:
            """Manual tool.
            Args:
                text: Input text.
            Returns:
                The input text.
            """
            return text

        manager.register(manual_tool)
        assert "manual_tool" in manager.tools
        assert manager.tools["manual_tool"] == manual_tool

    def test_add_tool_to_all(self):
        """Test adding a tool to all manager instances."""
        manager1 = ToolManager()
        manager2 = ToolManager()

        def shared_tool(value: str) -> str:
            """Shared tool.
            Args:
                value: Input value.
            Returns:
                The input value.
            """
            return value

        ToolManager.add_tool_to_all(shared_tool)

        assert "shared_tool" in manager1.tools
        assert "shared_tool" in manager2.tools

    def test_get_tool_schema(self):
        """Test getting tool schema."""
        manager = ToolManager()

        @tool
        def schema_test_tool(param: str) -> str:
            """Schema test tool.
            Args:
                param: A parameter.
            Returns:
                The parameter.
            """
            return param

        manager.register(schema_test_tool)
        schema = manager.get_tool_schema(schema_test_tool, "schema_test_tool")

        assert "type" in schema
        assert "function" in schema
        assert schema["function"]["name"] == "schema_test_tool"

    def test_get_tool_schema_missing(self):
        """Test getting schema for tool without schema."""
        manager = ToolManager()

        def no_schema_tool():
            return "test"

        schema = manager.get_tool_schema(no_schema_tool, "no_schema_tool")
        assert "error" in schema

    def test_get_all_tools_schema(self):
        """Test getting all tools schemas."""

        @tool
        def tool1(x: int) -> int:
            """Tool 1.
            Args:
                x: Input.
            Returns:
                Output.
            """
            return x

        @tool
        def tool2(y: str) -> str:
            """Tool 2.
            Args:
                y: Input.
            Returns:
                Output.
            """
            return y

        manager = ToolManager()
        schemas = manager.get_all_tools_schema()

        assert len(schemas) == 2
        assert all("function" in schema for schema in schemas)

    def test_call_tool_success(self):
        """Test successfully calling a tool."""
        manager = ToolManager()

        def callable_tool(message: str) -> str:
            """Callable tool.
            Args:
                message: Input message.
            Returns:
                The message with prefix.
            """
            return f"Result: {message}"

        manager.register(callable_tool)
        result = manager.call("callable_tool", {"message": "test"})
        assert result == "Result: test"

    def test_call_tool_not_found(self):
        """Test calling a non-existent tool."""
        manager = ToolManager()

        with pytest.raises(ValueError, match="Tool 'nonexistent' not found"):
            manager.call("nonexistent", {})

    def test_has_tool(self):
        """Test checking if a tool exists."""
        manager = ToolManager()

        def existing_tool():
            """Existing tool.
            Returns:
                Test result.
            """
            return "exists"

        manager.register(existing_tool)

        assert manager.has_tool("existing_tool") is True
        assert manager.has_tool("nonexistent_tool") is False

    def test_call_tools_no_tool_calls(self):
        """Test call_tools with no tool calls in response."""
        manager = ToolManager()
        mock_agent = Mock()

        # Mock LLM response with no tool calls
        mock_response = Mock()
        mock_response.tool_calls = None

        result = manager.call_tools(mock_agent, mock_response)
        assert result == []

    def test_call_tools_empty_tool_calls(self):
        """Test call_tools with empty tool calls list."""
        manager = ToolManager()
        mock_agent = Mock()

        # Mock LLM response with empty tool calls
        mock_response = Mock()
        mock_response.tool_calls = []

        result = manager.call_tools(mock_agent, mock_response)
        assert result == []

    def test_call_tools_success(self):
        """Test successful tool calling from LLM response."""
        manager = ToolManager()
        mock_agent = Mock()

        # Register a test tool
        def test_tool(agent, param1: str) -> str:
            """Test tool.
            Args:
                agent: The agent.
                param1: Test parameter.
            Returns:
                Test result.
            """
            return f"Tool result: {param1}"

        manager.register(test_tool)

        # Mock LLM response with tool calls
        mock_tool_call = Mock()
        mock_tool_call.id = "call_123"
        mock_tool_call.function.name = "test_tool"
        mock_tool_call.function.arguments = '{"param1": "test_value"}'

        mock_response = Mock()
        mock_response.tool_calls = [mock_tool_call]

        result = manager.call_tools(mock_agent, mock_response)

        assert len(result) == 1
        assert result[0]["tool_call_id"] == "call_123"
        assert result[0]["role"] == "tool"
        assert result[0]["name"] == "test_tool"
        assert "Tool result: test_value" in result[0]["response"]

    def test_call_tools_function_not_found(self):
        """Test call_tools with non-existent function."""
        manager = ToolManager()
        mock_agent = Mock()

        # Mock LLM response with non-existent tool
        mock_tool_call = Mock()
        mock_tool_call.id = "call_123"
        mock_tool_call.function.name = "nonexistent_tool"
        mock_tool_call.function.arguments = "{}"

        mock_response = Mock()
        mock_response.tool_calls = [mock_tool_call]

        result = manager.call_tools(mock_agent, mock_response)

        assert len(result) == 1
        assert result[0]["tool_call_id"] == "call_123"
        assert "Error:" in result[0]["response"]
        assert "not found in ToolManager" in result[0]["response"]

    def test_call_tools_invalid_json(self):
        """Test call_tools with invalid JSON arguments."""
        manager = ToolManager()
        mock_agent = Mock()

        def test_tool(agent, param1: str) -> str:
            """Test tool.
            Args:
                agent: The agent.
                param1: Test parameter.
            Returns:
                Test result.
            """
            return f"Tool result: {param1}"

        manager.register(test_tool)

        # Mock LLM response with invalid JSON
        mock_tool_call = Mock()
        mock_tool_call.id = "call_123"
        mock_tool_call.function.name = "test_tool"
        mock_tool_call.function.arguments = "invalid json"

        mock_response = Mock()
        mock_response.tool_calls = [mock_tool_call]

        result = manager.call_tools(mock_agent, mock_response)

        assert len(result) == 1
        assert result[0]["tool_call_id"] == "call_123"
        assert "Error:" in result[0]["response"]

    def test_call_tools_type_error_handling(self):
        """Test call_tools handling TypeError with argument filtering."""
        manager = ToolManager()
        mock_agent = Mock()

        def strict_tool(agent, required_param: str) -> str:
            """Strict tool.
            Args:
                agent: The agent.
                required_param: Required parameter.
            Returns:
                Test result.
            """
            return f"Result: {required_param}"

        manager.register(strict_tool)

        # Mock LLM response with extra arguments that don't match function signature
        mock_tool_call = Mock()
        mock_tool_call.id = "call_123"
        mock_tool_call.function.name = "strict_tool"
        mock_tool_call.function.arguments = (
            '{"required_param": "test", "extra_param": "ignored"}'
        )

        mock_response = Mock()
        mock_response.tool_calls = [mock_tool_call]

        result = manager.call_tools(mock_agent, mock_response)

        # This test should show that the tool call fails since agent parameter filtering
        # in the current implementation doesn't properly handle the agent parameter
        assert len(result) == 1
        assert result[0]["tool_call_id"] == "call_123"
        assert "Error:" in result[0]["response"]

    def test_call_tools_successful_argument_filtering(self):
        """Test call_tools successfully filtering extra arguments for functions without agent parameter."""
        manager = ToolManager()
        mock_agent = Mock()

        def simple_tool(required_param: str) -> str:
            """Simple tool that doesn't need agent parameter.
            Args:
                required_param: Required parameter.
            Returns:
                Test result.
            """
            return f"Result: {required_param}"

        manager.register(simple_tool)

        # Mock LLM response with extra arguments
        mock_tool_call = Mock()
        mock_tool_call.id = "call_123"
        mock_tool_call.function.name = "simple_tool"
        mock_tool_call.function.arguments = (
            '{"required_param": "test", "extra_param": "ignored"}'
        )

        mock_response = Mock()
        mock_response.tool_calls = [mock_tool_call]

        result = manager.call_tools(mock_agent, mock_response)

        assert len(result) == 1
        assert result[0]["tool_call_id"] == "call_123"
        assert "Result: test" in result[0]["response"]

    def test_call_tools_no_response(self):
        """Test call_tools when function returns None."""
        manager = ToolManager()
        mock_agent = Mock()

        def silent_tool(agent) -> None:
            """Silent tool.
            Args:
                agent: The agent.
            Returns:
                Nothing.
            """
            # Returns None

        manager.register(silent_tool)

        # Mock LLM response
        mock_tool_call = Mock()
        mock_tool_call.id = "call_123"
        mock_tool_call.function.name = "silent_tool"
        mock_tool_call.function.arguments = "{}"

        mock_response = Mock()
        mock_response.tool_calls = [mock_tool_call]

        result = manager.call_tools(mock_agent, mock_response)

        assert len(result) == 1
        assert result[0]["response"] == "silent_tool executed successfully"

    def test_call_tools_general_exception(self):
        """Test call_tools with general exception during processing."""
        manager = ToolManager()
        mock_agent = Mock()

        # Create a mock response that will cause a general exception
        # by making tool_calls raise an exception when accessed
        mock_response = Mock()
        mock_response.tool_calls = Mock(side_effect=Exception("General error"))

        result = manager.call_tools(mock_agent, mock_response)
        assert result == []
