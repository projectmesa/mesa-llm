import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from mesa_llm.reasoning.reasoning import (
    _UNSET,
    Observation,
    Plan,
    Reasoning,
)
from mesa_llm.tools.tool_decorator import tool
from mesa_llm.tools.tool_manager import ToolManager


class TestObservation:
    """Test the Observation dataclass."""

    def test_observation_creation(self):
        """Test creating an Observation with valid data."""
        obs = Observation(
            step=1,
            self_state={"position": (0, 0), "health": 100},
            local_state={"Agent_1": {"position": (1, 1), "health": 90}},
        )

        assert obs.step == 1
        assert obs.self_state["position"] == (0, 0)
        assert obs.self_state["health"] == 100
        assert "Agent_1" in obs.local_state
        assert obs.local_state["Agent_1"]["position"] == (1, 1)


class TestPlan:
    """Test the Plan dataclass."""

    def test_plan_creation(self):
        """Test creating a Plan with valid data."""
        mock_llm_response = Mock()
        mock_llm_response.content = "Test plan content"

        plan = Plan(step=1, llm_plan=mock_llm_response, ttl=3)

        assert plan.step == 1
        assert plan.llm_plan == mock_llm_response
        assert plan.ttl == 3


class TestReasoningBase:
    """Tests for the abstract Reasoning base class."""

    def test_execute_tool_call_generates_plan(self, llm_response_factory, mock_agent):
        """Test that the base execute_tool_call method produces a Plan."""
        # 1. Setup a mock agent with all necessary components
        mock_agent.model.steps = 5

        mock_llm_response = llm_response_factory(content="Final LLM message")
        mock_agent.llm.generate.return_value = mock_llm_response

        # Mock the Tool Manager
        mock_agent.tool_manager.get_tools_schema.return_value = [{"schema": "example"}]

        # 2. Instantiate a concrete implementation of Reasoning to test the base method
        class ConcreteReasoning(Reasoning):
            def plan(
                self,
                prompt=None,
                obs=None,
                ttl=1,
                tools=_UNSET,
                tool_calls="auto",
                selected_tools=_UNSET,
            ):
                pass  # Not needed for this test

        reasoning = ConcreteReasoning(agent=mock_agent)

        # 3. Call the method we want to test
        chaining_message = "Execute the plan."
        result_plan = reasoning.execute_tool_call(chaining_message, tools=["tool1"])

        # 4. Assert the results
        # Assert that the LLM was called with the correct parameters
        mock_agent.llm.generate.assert_called_once_with(
            prompt=chaining_message,
            tool_schema=[{"schema": "example"}],
            tool_choice="auto",
            system_prompt="You are an executor that executes the plan given to you in the prompt through tool calls. If the plan concludes that no action should be taken, do not call any tool.",
        )
        # Assert that the tool manager was asked for the correct schema
        mock_agent.tool_manager.get_tools_schema.assert_called_once_with(
            tools=["tool1"]
        )
        # Assert that the output is a correctly formed Plan object
        assert isinstance(result_plan, Plan)
        assert result_plan.step == 5
        assert result_plan.llm_plan.content == "Final LLM message"
        assert result_plan.ttl == 1
        mock_agent.memory.add_to_memory.assert_called_once_with(
            type="plan_execution",
            content={"content": str(result_plan)},
        )

    def test_execute_tool_call_does_not_mutate_llm_prompt(
        self, llm_response_factory, mock_agent
    ):
        """Executor prompt should be scoped to call kwargs, not shared state."""
        mock_agent.model.steps = 5
        mock_agent.llm.system_prompt = "base-system-prompt"
        mock_agent.llm.generate.return_value = llm_response_factory(content="ok")
        mock_agent.tool_manager.get_tools_schema.return_value = []

        class ConcreteReasoning(Reasoning):
            def plan(
                self,
                prompt=None,
                obs=None,
                ttl=1,
                tools=_UNSET,
                tool_calls="auto",
                selected_tools=_UNSET,
            ):
                pass

        reasoning = ConcreteReasoning(agent=mock_agent)
        reasoning.execute_tool_call("Execute the plan.")

        assert mock_agent.llm.system_prompt == "base-system-prompt"

    @pytest.mark.asyncio
    async def test_aexecute_tool_call_does_not_mutate_llm_prompt(
        self, llm_response_factory, mock_agent
    ):
        """Async executor prompt should be scoped to call kwargs, not shared state."""
        mock_agent.model.steps = 5
        mock_agent.llm.system_prompt = "base-system-prompt"
        mock_agent.llm.agenerate = AsyncMock(
            return_value=llm_response_factory(content="ok")
        )
        mock_agent.memory.aadd_to_memory = AsyncMock()
        mock_agent.tool_manager.get_tools_schema.return_value = []

        class ConcreteReasoning(Reasoning):
            def plan(
                self,
                prompt=None,
                obs=None,
                ttl=1,
                tools=_UNSET,
                tool_calls="auto",
                selected_tools=_UNSET,
            ):
                pass

        reasoning = ConcreteReasoning(agent=mock_agent)
        await reasoning.aexecute_tool_call("Execute the plan.")

        assert mock_agent.llm.system_prompt == "base-system-prompt"

    def test_execute_tool_call_propagates_ttl(self):
        """Test that execute_tool_call propagates caller-provided TTL."""
        mock_agent = Mock()
        mock_agent.model.steps = 5
        mock_llm_response = Mock()
        mock_llm_response.choices = [Mock()]
        mock_llm_response.choices[0].message = "Final LLM message"
        mock_agent.llm.generate.return_value = mock_llm_response
        mock_agent.tool_manager.get_tools_schema.return_value = []

        class ConcreteReasoning(Reasoning):
            def plan(
                self,
                prompt=None,
                obs=None,
                ttl=1,
                tools=_UNSET,
                tool_calls="auto",
                selected_tools=_UNSET,
            ):
                pass

        reasoning = ConcreteReasoning(agent=mock_agent)
        result_plan = reasoning.execute_tool_call("Execute the plan.", ttl=7)

        assert isinstance(result_plan, Plan)
        assert result_plan.ttl == 7

    def test_execute_tool_call_respects_tool_calls_override(
        self, llm_response_factory, mock_agent
    ):
        """Test that execute_tool_call forwards the caller's tool choice."""
        mock_agent.model.steps = 5
        mock_llm_response = llm_response_factory(content="Final LLM message")
        mock_agent.llm.generate.return_value = mock_llm_response
        mock_agent.tool_manager.get_tools_schema.return_value = [{"schema": "example"}]

        class ConcreteReasoning(Reasoning):
            def plan(
                self,
                prompt=None,
                obs=None,
                ttl=1,
                tools=_UNSET,
                tool_calls="auto",
                selected_tools=_UNSET,
            ):
                pass

        reasoning = ConcreteReasoning(agent=mock_agent)
        reasoning.execute_tool_call(
            "Execute the plan.",
            tools=["tool1"],
            tool_calls="auto",
        )

        mock_agent.llm.generate.assert_called_once_with(
            prompt="Execute the plan.",
            tool_schema=[{"schema": "example"}],
            tool_choice="auto",
            system_prompt="You are an executor that executes the plan given to you in the prompt through tool calls. If the plan concludes that no action should be taken, do not call any tool.",
        )

    def test_execute_tool_call_tools_sentinel_semantics(
        self, llm_response_factory, mock_agent
    ):
        """Omitted tools inherit; explicit None/list overrides per call."""

        @tool
        def inherited_reasoning_tool(agent, x: int) -> int:
            """Inherited reasoning tool.
            Args:
                agent: The agent making the request (provided automatically)
                x: Input.
            Returns:
                Output.
            """
            return x

        @tool
        def override_reasoning_tool(agent, y: int) -> int:
            """Override reasoning tool.
            Args:
                agent: The agent making the request (provided automatically)
                y: Input.
            Returns:
                Output.
            """
            return y

        mock_agent.model.steps = 5
        mock_agent.llm.generate.return_value = llm_response_factory(content="ok")
        mock_agent._tool_manager = ToolManager(
            tools=[inherited_reasoning_tool, override_reasoning_tool]
        )

        class ConcreteReasoning(Reasoning):
            def plan(
                self,
                prompt=None,
                obs=None,
                ttl=1,
                tools=_UNSET,
                tool_calls="auto",
                selected_tools=_UNSET,
            ):
                pass

        reasoning = ConcreteReasoning(agent=mock_agent)

        reasoning.execute_tool_call("inherit")
        schemas = mock_agent.llm.generate.call_args.kwargs["tool_schema"]
        assert [schema["function"]["name"] for schema in schemas] == [
            "inherited_reasoning_tool",
            "override_reasoning_tool",
        ]

        reasoning.execute_tool_call("none", tools=None)
        assert mock_agent.llm.generate.call_args.kwargs["tool_schema"] == []

        reasoning.execute_tool_call("empty", tools=[])
        assert mock_agent.llm.generate.call_args.kwargs["tool_schema"] == []

        reasoning.execute_tool_call("override", tools=[override_reasoning_tool])
        schemas = mock_agent.llm.generate.call_args.kwargs["tool_schema"]
        assert [schema["function"]["name"] for schema in schemas] == [
            "override_reasoning_tool"
        ]

    def test_execute_tool_call_selected_tools_alias_warns(
        self, llm_response_factory, mock_agent
    ):
        """selected_tools remains a deprecated compatibility alias."""

        @tool
        def alias_reasoning_tool(agent, x: int) -> int:
            """Alias reasoning tool.
            Args:
                agent: The agent making the request (provided automatically)
                x: Input.
            Returns:
                Output.
            """
            return x

        mock_agent.model.steps = 5
        mock_agent.llm.generate.return_value = llm_response_factory(content="ok")
        mock_agent._tool_manager = ToolManager(tools=[alias_reasoning_tool])

        class ConcreteReasoning(Reasoning):
            def plan(
                self,
                prompt=None,
                obs=None,
                ttl=1,
                tools=_UNSET,
                tool_calls="auto",
                selected_tools=_UNSET,
            ):
                pass

        reasoning = ConcreteReasoning(agent=mock_agent)

        with pytest.warns(DeprecationWarning, match="selected_tools"):
            reasoning.execute_tool_call(
                "alias",
                selected_tools=[alias_reasoning_tool],
            )

        schemas = mock_agent.llm.generate.call_args.kwargs["tool_schema"]
        assert [schema["function"]["name"] for schema in schemas] == [
            "alias_reasoning_tool"
        ]

        with pytest.warns(DeprecationWarning, match="selected_tools"):
            reasoning.execute_tool_call(
                "alias none inherits",
                selected_tools=None,
            )

        schemas = mock_agent.llm.generate.call_args.kwargs["tool_schema"]
        assert [schema["function"]["name"] for schema in schemas] == [
            "alias_reasoning_tool"
        ]

    def test_execute_tool_call_rejects_unconfigured_per_call_tools(self, mock_agent):
        """Per-call tools narrow configured tools and cannot inject tools."""

        @tool
        def configured_reasoning_tool(agent, x: int) -> int:
            """Configured reasoning tool.
            Args:
                agent: The agent making the request (provided automatically)
                x: Input.
            Returns:
                Output.
            """
            return x

        @tool
        def unconfigured_reasoning_tool(agent, y: int) -> int:
            """Unconfigured reasoning tool.
            Args:
                agent: The agent making the request (provided automatically)
                y: Input.
            Returns:
                Output.
            """
            return y

        mock_agent._tool_manager = ToolManager(tools=[configured_reasoning_tool])
        mock_agent.llm.generate.reset_mock()

        class ConcreteReasoning(Reasoning):
            def plan(
                self,
                prompt=None,
                obs=None,
                ttl=1,
                tools=_UNSET,
                tool_calls="auto",
                selected_tools=_UNSET,
            ):
                pass

        reasoning = ConcreteReasoning(agent=mock_agent)

        with pytest.raises(ValueError, match="Unknown tool name"):
            reasoning.execute_tool_call(
                "reject callable", tools=[unconfigured_reasoning_tool]
            )

        with pytest.raises(ValueError, match="Unknown tool name"):
            reasoning.execute_tool_call(
                "reject name", tools=["unconfigured_reasoning_tool"]
            )

        mock_agent.llm.generate.assert_not_called()

    def test_aexecute_tool_call_records_plan_execution(
        self, llm_response_factory, mock_agent
    ):
        """Test that aexecute_tool_call logs plan_execution to memory."""
        mock_agent.model.steps = 5
        mock_llm_response = llm_response_factory(content="Async final LLM message")
        mock_agent.llm.agenerate = AsyncMock(return_value=mock_llm_response)
        mock_agent.tool_manager.get_tools_schema.return_value = [{"schema": "example"}]
        mock_agent.memory.aadd_to_memory = AsyncMock()

        class ConcreteReasoning(Reasoning):
            def plan(
                self,
                prompt=None,
                obs=None,
                ttl=1,
                tools=_UNSET,
                tool_calls="auto",
                selected_tools=_UNSET,
            ):
                pass

        reasoning = ConcreteReasoning(agent=mock_agent)
        result_plan = asyncio.run(
            reasoning.aexecute_tool_call("Execute the plan.", tools=["tool1"])
        )

        mock_agent.llm.agenerate.assert_awaited_once_with(
            prompt="Execute the plan.",
            tool_schema=[{"schema": "example"}],
            tool_choice="auto",
            system_prompt="You are an executor that executes the plan given to you in the prompt through tool calls. If the plan concludes that no action should be taken, do not call any tool.",
        )
        mock_agent.memory.aadd_to_memory.assert_awaited_once_with(
            type="plan_execution",
            content={"content": str(result_plan)},
        )
