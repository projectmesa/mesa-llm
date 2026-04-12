import asyncio
from unittest.mock import AsyncMock, Mock

from mesa_llm.reasoning.reasoning import (
    Observation,
    Plan,
    Reasoning,
)


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
        mock_agent.tool_manager.get_all_tools_schema.return_value = [
            {"schema": "example"}
        ]

        # 2. Instantiate a concrete implementation of Reasoning to test the base method
        class ConcreteReasoning(Reasoning):
            def plan(
                self,
                prompt=None,
                obs=None,
                ttl=1,
                selected_tools=None,
                tool_calls="auto",
            ):
                pass  # Not needed for this test

        reasoning = ConcreteReasoning(agent=mock_agent)

        # 3. Call the method we want to test
        chaining_message = "Execute the plan."
        result_plan = reasoning.execute_tool_call(
            chaining_message, selected_tools=["tool1"]
        )

        # 4. Assert the results
        # Assert that the LLM was called with the correct parameters
        mock_agent.llm.generate.assert_called_once_with(
            prompt=chaining_message,
            tool_schema=[{"schema": "example"}],
            tool_choice="auto",
        )
        # Assert that the tool manager was asked for the correct schema
        mock_agent.tool_manager.get_all_tools_schema.assert_called_once_with(
            selected_tools=["tool1"]
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

    def test_execute_tool_call_propagates_ttl(self):
        """Test that execute_tool_call propagates caller-provided TTL."""
        mock_agent = Mock()
        mock_agent.model.steps = 5
        mock_llm_response = Mock()
        mock_llm_response.choices = [Mock()]
        mock_llm_response.choices[0].message = "Final LLM message"
        mock_agent.llm.generate.return_value = mock_llm_response
        mock_agent.tool_manager.get_all_tools_schema.return_value = []

        class ConcreteReasoning(Reasoning):
            def plan(
                self,
                prompt=None,
                obs=None,
                ttl=1,
                selected_tools=None,
                tool_calls="auto",
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
        mock_agent.tool_manager.get_all_tools_schema.return_value = [
            {"schema": "example"}
        ]

        class ConcreteReasoning(Reasoning):
            def plan(
                self,
                prompt=None,
                obs=None,
                ttl=1,
                selected_tools=None,
                tool_calls="auto",
            ):
                pass

        reasoning = ConcreteReasoning(agent=mock_agent)
        reasoning.execute_tool_call(
            "Execute the plan.",
            selected_tools=["tool1"],
            tool_calls="required",
        )

        mock_agent.llm.generate.assert_called_once_with(
            prompt="Execute the plan.",
            tool_schema=[{"schema": "example"}],
            tool_choice="required",
        )

    def test_aexecute_tool_call_records_plan_execution(
        self, llm_response_factory, mock_agent
    ):
        """Test that aexecute_tool_call logs plan_execution to memory."""
        mock_agent.model.steps = 5
        mock_llm_response = llm_response_factory(content="Async final LLM message")
        mock_agent.llm.agenerate = AsyncMock(return_value=mock_llm_response)
        mock_agent.tool_manager.get_all_tools_schema.return_value = [
            {"schema": "example"}
        ]
        mock_agent.memory.aadd_to_memory = AsyncMock()

        class ConcreteReasoning(Reasoning):
            def plan(
                self,
                prompt=None,
                obs=None,
                ttl=1,
                selected_tools=None,
                tool_calls="auto",
            ):
                pass

        reasoning = ConcreteReasoning(agent=mock_agent)
        result_plan = asyncio.run(
            reasoning.aexecute_tool_call("Execute the plan.", selected_tools=["tool1"])
        )

        mock_agent.llm.agenerate.assert_awaited_once_with(
            prompt="Execute the plan.",
            tool_schema=[{"schema": "example"}],
            tool_choice="auto",
        )
        mock_agent.memory.aadd_to_memory.assert_awaited_once_with(
            type="plan_execution",
            content={"content": str(result_plan)},
        )
