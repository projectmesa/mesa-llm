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
        def test_global_tool(agent, param1: str) -> str:
            """Test global tool.
            Args:
                agent: The agent making the request (provided automatically)
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

        def extra_tool(agent, x: int) -> int:
            """Extra tool.
            Args:
                agent: The agent making the request (provided automatically)
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

        def manual_tool(agent, text: str) -> str:
            """Manual tool.
            Args:
                agent: The agent making the request (provided automatically)
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

        def shared_tool(agent, value: str) -> str:
            """Shared tool.
            Args:
                agent: The agent making the request (provided automatically)
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
        def schema_test_tool(agent, param: str) -> str:
            """Schema test tool.
            Args:
                agent: The agent making the request (provided automatically)
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
        def tool1(agent, x: int) -> int:
            """Tool 1.
            Args:
                agent: The agent making the request (provided automatically)
                x: Input.
            Returns:
                Output.
            """
            return x

        @tool
        def tool2(agent, y: str) -> str:
            """Tool 2.
            Args:
                agent: The agent making the request (provided automatically)
                y: Input.
            Returns:
                Output.
            """
            return y

        manager = ToolManager()
        schemas = manager.get_all_tools_schema()

        assert len(schemas) == 2
        assert all("function" in schema for schema in schemas)

    def test_get_all_tools_schema_with_selected_tools(self):
        """Test getting schemas for selected tools only."""

        @tool
        def tool_a(agent, x: int) -> int:
            """Tool A.
            Args:
                agent: The agent making the request (provided automatically)
                x: Input.
            Returns:
                Output.
            """
            return x

        @tool
        def tool_b(agent, y: str) -> str:
            """Tool B.
            Args:
                agent: The agent making the request (provided automatically)
                y: Input.
            Returns:
                Output.
            """
            return y

        @tool
        def tool_c(agent, z: float) -> float:
            """Tool C.
            Args:
                agent: The agent making the request (provided automatically)
                z: Input.
            Returns:
                Output.
            """
            return z

        manager = ToolManager()

        # Test selecting specific tools
        selected_tools = ["tool_a", "tool_c"]
        schemas = manager.get_all_tools_schema(selected_tools)

        assert len(schemas) == 2
        tool_names = [schema["function"]["name"] for schema in schemas]
        assert "tool_a" in tool_names
        assert "tool_c" in tool_names
        assert "tool_b" not in tool_names

    def test_get_all_tools_schema_empty_list(self):
        """Test that empty list returns no tools (selected_tools=[] means 'no tools')."""

        @tool
        def test_tool(agent, x: int) -> int:
            """Test tool.
            Args:
                agent: The agent making the request (provided automatically)
                x: Input.
            Returns:
                Output.
            """
            return x

        manager = ToolManager()

        # Empty list should return no tools — the user explicitly asked for none
        empty_list_schemas = manager.get_all_tools_schema([])

        assert len(empty_list_schemas) == 0

    def test_get_all_tools_schema_none(self):
        """Test that None returns all tools."""

        @tool
        def test_tool(agent, x: int) -> int:
            """Test tool.
            Args:
                agent: The agent making the request (provided automatically)
                x: Input.
            Returns:
                Output.
            """
            return x

        manager = ToolManager()

        all_schemas = manager.get_all_tools_schema()
        none_schemas = manager.get_all_tools_schema(None)

        assert len(none_schemas) == len(all_schemas)

    def test_get_all_tools_schema_nonexistent_tools(self):
        """Test that requesting nonexistent tools raises appropriate errors."""

        @tool
        def existing_tool(agent, x: int) -> int:
            """Existing tool.
            Args:
                agent: The agent making the request (provided automatically)
                x: Input.
            Returns:
                Output.
            """
            return x

        manager = ToolManager()

        # Test with nonexistent tools
        selected_tools = ["existing_tool", "nonexistent_tool"]

        with pytest.raises(ValueError, match="Unknown tool name"):
            manager.get_all_tools_schema(selected_tools)

    def test_get_all_tools_schema_single_tool(self):
        """Test selecting a single tool."""

        @tool
        def single_tool(agent, x: int) -> int:
            """Single tool.
            Args:
                agent: The agent making the request (provided automatically)
                x: Input.
            Returns:
                Output.
            """
            return x

        @tool
        def other_tool(agent, y: str) -> str:
            """Other tool.
            Args:
                agent: The agent making the request (provided automatically)
                y: Input.
            Returns:
                Output.
            """
            return y

        manager = ToolManager()

        schemas = manager.get_all_tools_schema(["single_tool"])

        assert len(schemas) == 1
        assert schemas[0]["function"]["name"] == "single_tool"

    def test_call_tool_success(self):
        """Test successfully calling a tool."""
        manager = ToolManager()

        def callable_tool(agent, message: str) -> str:
            """Callable tool.
            Args:
                agent: The agent making the request (provided automatically)
                message: Input message.
            Returns:
                The message with prefix.
            """
            return f"Result: {message}"

        manager.register(callable_tool)
        result = manager.call("callable_tool", {"agent": Mock(), "message": "test"})
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
            return "test"

        manager.register(existing_tool)

        assert manager.has_tool("existing_tool") is True
        assert manager.has_tool("nonexistent_tool") is False

    def test_call_tools_no_tool_calls(self):
        """Test call_tools with response that has no tool_calls."""
        manager = ToolManager()
        mock_agent = Mock()

        mock_response = Mock()
        mock_response.tool_calls = None

        result = manager.call_tools(mock_agent, mock_response)
        assert result == []

    def test_call_tools_empty_tool_calls(self):
        """Test call_tools with empty tool_calls list."""
        manager = ToolManager()
        mock_agent = Mock()

        mock_response = Mock()
        mock_response.tool_calls = []

        result = manager.call_tools(mock_agent, mock_response)
        assert result == []

    def test_call_tools_success(self):
        """Test successful tool calling."""
        manager = ToolManager()

        @tool
        def test_tool(agent, param1: str) -> str:
            """Test tool for call_tools.
            Args:
                agent: The agent making the request (provided automatically)
                param1: Test parameter.
            Returns:
                Processed parameter.
            """
            return f"Processed: {param1}"

        # Mock agent
        mock_agent = Mock()

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
        assert "Processed: test_value" in result[0]["response"]

    def test_call_tools_function_not_found(self):
        """Test call_tools with non-existent function."""
        manager = ToolManager()
        mock_agent = Mock()

        mock_tool_call = Mock()
        mock_tool_call.id = "call_123"
        mock_tool_call.function.name = "nonexistent_function"
        mock_tool_call.function.arguments = '{"param": "value"}'

        mock_response = Mock()
        mock_response.tool_calls = [mock_tool_call]

        result = manager.call_tools(mock_agent, mock_response)

        assert len(result) == 1
        assert result[0]["tool_call_id"] == "call_123"
        assert "Error:" in result[0]["response"]

    def test_call_tools_invalid_json(self):
        """Test call_tools with invalid JSON arguments."""
        manager = ToolManager()

        @tool
        def test_tool(agent, param1: str) -> str:
            """Test tool.
            Args:
                agent: The agent making the request (provided automatically)
                param1: Test parameter.
            Returns:
                Processed parameter.
            """
            return f"Processed: {param1}"

        mock_agent = Mock()

        mock_tool_call = Mock()
        mock_tool_call.id = "call_123"
        mock_tool_call.function.name = "test_tool"
        mock_tool_call.function.arguments = '{"param1": invalid_json}'

        mock_response = Mock()
        mock_response.tool_calls = [mock_tool_call]

        result = manager.call_tools(mock_agent, mock_response)

        assert len(result) == 1
        assert result[0]["tool_call_id"] == "call_123"
        assert "Error:" in result[0]["response"]

    def test_call_tools_successful_argument_filtering(self):
        """Test call_tools with argument filtering when function signature doesn't match."""
        manager = ToolManager()

        def simple_tool(required_param: str) -> str:
            """Simple tool that only takes required_param.
            Args:
                required_param: The only parameter this function accepts.
            Returns:
                Processed parameter.
            """
            return f"Simple: {required_param}"

        # Register tool manually without using decorator to test filtering
        manager.register(simple_tool)

        mock_agent = Mock()

        mock_tool_call = Mock()
        mock_tool_call.id = "call_123"
        mock_tool_call.function.name = "simple_tool"
        # Include extra parameters that the function doesn't accept
        mock_tool_call.function.arguments = (
            '{"required_param": "test", "extra_param": "ignored"}'
        )

        mock_response = Mock()
        mock_response.tool_calls = [mock_tool_call]

        result = manager.call_tools(mock_agent, mock_response)

        assert len(result) == 1
        assert result[0]["tool_call_id"] == "call_123"
        assert "Simple: test" in result[0]["response"]

    def test_call_tools_no_response(self):
        """Test call_tools when tool returns None."""
        manager = ToolManager()

        @tool
        def silent_tool(agent) -> None:
            """Tool that returns None.
            Args:
                agent: The agent making the request (provided automatically)
            Returns:
                None
            """
            return None

        mock_agent = Mock()

        mock_tool_call = Mock()
        mock_tool_call.id = "call_123"
        mock_tool_call.function.name = "silent_tool"
        mock_tool_call.function.arguments = "{}"

        mock_response = Mock()
        mock_response.tool_calls = [mock_tool_call]

        result = manager.call_tools(mock_agent, mock_response)

        assert len(result) == 1
        assert result[0]["tool_call_id"] == "call_123"
        assert "silent_tool executed successfully" in result[0]["response"]

    def test_call_tools_general_exception(self):
        """Test call_tools handling of general exceptions."""
        manager = ToolManager()
        mock_agent = Mock()

        # Create a mock response that will cause an AttributeError
        mock_response = Mock()
        # Remove the tool_calls attribute to cause AttributeError
        del mock_response.tool_calls

        result = manager.call_tools(mock_agent, mock_response)
        assert result == []

    def test_selected_tools_consistency(self):
        """Test that selected_tools parameter works consistently."""

        @tool
        def consistency_tool_a(agent, x: int) -> int:
            """Consistency tool A.
            Args:
                agent: The agent making the request (provided automatically)
                x: Input.
            Returns:
                Output.
            """
            return x

        @tool
        def consistency_tool_b(agent, y: str) -> str:
            """Consistency tool B.
            Args:
                agent: The agent making the request (provided automatically)
                y: Input.
            Returns:
                Output.
            """
            return y

        @tool
        def consistency_tool_c(agent, z: float) -> float:
            """Consistency tool C.
            Args:
                agent: The agent making the request (provided automatically)
                z: Input.
            Returns:
                Output.
            """
            return z

        manager = ToolManager()

        # Test that same selected_tools always returns same schemas
        selected_tools = ["consistency_tool_a", "consistency_tool_c"]

        schemas1 = manager.get_all_tools_schema(selected_tools)
        schemas2 = manager.get_all_tools_schema(selected_tools)

        names1 = sorted([schema["function"]["name"] for schema in schemas1])
        names2 = sorted([schema["function"]["name"] for schema in schemas2])

        assert names1 == names2
        assert len(schemas1) == len(schemas2) == 2

        # Test order independence
        reversed_tools = list(reversed(selected_tools))
        schemas3 = manager.get_all_tools_schema(reversed_tools)
        names3 = sorted([schema["function"]["name"] for schema in schemas3])

        assert names1 == names3

    def test_selected_tools_duplicate_handling(self):
        """Test how selected_tools handles duplicates."""

        @tool
        def duplicate_test_tool(agent, x: int) -> int:
            """Duplicate test tool.
            Args:
                agent: The agent making the request (provided automatically)
                x: Input.
            Returns:
                Output.
            """
            return x

        manager = ToolManager()

        # Test with duplicate tool names
        selected_tools = ["duplicate_test_tool", "duplicate_test_tool"]
        schemas = manager.get_all_tools_schema(selected_tools)

        # Should return schemas for each request (may include duplicates)
        assert len(schemas) == 2

    def test_multiple_managers_selected_tools(self):
        """Test selected_tools functionality with multiple ToolManager instances."""

        @tool
        def shared_tool_1(agent, x: int) -> int:
            """Shared tool 1.
            Args:
                agent: The agent making the request (provided automatically)
                x: Input.
            Returns:
                Output.
            """
            return x

        @tool
        def shared_tool_2(agent, y: str) -> str:
            """Shared tool 2.
            Args:
                agent: The agent making the request (provided automatically)
                y: Input.
            Returns:
                Output.
            """
            return y

        manager1 = ToolManager()
        manager2 = ToolManager()

        # Both managers should have the same tools
        all_schemas_1 = manager1.get_all_tools_schema()
        all_schemas_2 = manager2.get_all_tools_schema()

        assert len(all_schemas_1) == len(all_schemas_2)

        # Selected tools should work the same on both managers
        selected_tools = ["shared_tool_1"]
        schemas_1 = manager1.get_all_tools_schema(selected_tools)
        schemas_2 = manager2.get_all_tools_schema(selected_tools)

        assert len(schemas_1) == len(schemas_2) == 1
        assert schemas_1[0]["function"]["name"] == schemas_2[0]["function"]["name"]

    @pytest.mark.asyncio
    async def test_acall_tools_success(self, monkeypatch):
        """
        This test validates the full async tool execution pipeline by ensuring that:

        - Tool calls are correctly extracted from the LLM response object.
        - JSON-formatted arguments are parsed without error.
        - The `agent` parameter is automatically injected when required by the tool's function signature.
        - The tool function is executed successfully.
        - The result is wrapped in the expected structured response format:
            {
                "tool_call_id": <tool_call_id>,
                "role": "tool",
                "name": <tool_name>,
                "response": <stringified_result>
            }
        - The asynchronous execution path using `asyncio.gather`
        returns the correct results.
        """
        manager = ToolManager()
        mock_agent = Mock()

        @tool
        def async_test_tool(agent, value: str) -> str:
            """Async test tool.

            Args:
                agent: The agent making the request (provided automatically)
                value: Input value.

            Returns:
                Processed value.
            """
            return f"Async: {value}"

        mock_tool_call = Mock()
        mock_tool_call.id = "call_async"
        mock_tool_call.function.name = "async_test_tool"
        mock_tool_call.function.arguments = '{"value": "hello"}'

        mock_response = Mock()
        mock_response.tool_calls = [mock_tool_call]

        result = await manager.acall_tools(mock_agent, mock_response)

        assert len(result) == 1
        assert result[0]["tool_call_id"] == "call_async"
        assert "Async: hello" in result[0]["response"]
