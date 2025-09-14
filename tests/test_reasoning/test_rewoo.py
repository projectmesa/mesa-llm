# tests/test_reasoning/test_rewoo.py

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from mesa_llm.reasoning.reasoning import Observation, Plan
from mesa_llm.reasoning.rewoo import ReWOOReasoning


class TestReWOOReasoning:
    """Test the ReWOOReasoning class."""

    def test_rewoo_reasoning_initialization(self):
        """Test ReWOOReasoning initialization."""
        mock_agent = Mock()
        reasoning = ReWOOReasoning(mock_agent)

        assert reasoning.agent == mock_agent
        assert reasoning.remaining_tool_calls == 0
        assert reasoning.current_plan is None
        assert reasoning.current_obs is None

    def test_get_rewoo_system_prompt(self):
        """Test get_rewoo_system_prompt."""
        mock_agent = Mock()
        mock_agent.memory = Mock()
        mock_agent.memory.format_long_term.return_value = "Long term memory content"
        mock_agent.memory.format_short_term.return_value = "Short term memory content"

        reasoning = ReWOOReasoning(mock_agent)
        reasoning.current_obs = Observation(
            step=1, self_state={"test": "data"}, local_state={}
        )

        prompt = reasoning.get_rewoo_system_prompt(reasoning.current_obs)

        assert "Long term memory content" in prompt
        assert "Short term memory content" in prompt
        assert "Current Observation" in prompt
        assert "plan" in prompt
        assert "step_1" in prompt
        assert "contingency" in prompt

    def test_plan_with_remaining_tool_calls(self):
        """Test plan method when there are remaining tool calls."""
        mock_agent = Mock()
        mock_agent.generate_obs = Mock()

        reasoning = ReWOOReasoning(mock_agent)
        reasoning.remaining_tool_calls = 2
        reasoning.current_plan = Mock()

        # Create actual mock objects for tool calls
        mock_tool_1 = Mock()
        mock_tool_2 = Mock()
        mock_tool_3 = Mock()
        reasoning.current_plan.tool_calls = [
            mock_tool_1,
            mock_tool_2,
            mock_tool_3,
        ]  # 3 tool calls
        reasoning.current_obs = Observation(step=1, self_state={}, local_state={})

        result = reasoning.plan()

        assert isinstance(result, Plan)
        assert result.llm_plan.tool_calls == [mock_tool_2]  # Should get index 1 (3-2)
        assert reasoning.remaining_tool_calls == 1
        mock_agent.generate_obs.assert_not_called()

    def test_plan_new_plan_generation(self):
        """Test plan method when generating a new plan."""
        mock_agent = Mock()
        mock_agent.step_prompt = "Default step prompt"
        mock_agent.generate_obs.return_value = Observation(
            step=1, self_state={}, local_state={}
        )
        mock_agent.memory = Mock()
        mock_agent.memory.format_long_term.return_value = "Long term memory"
        mock_agent.memory.format_short_term.return_value = "Short term memory"
        mock_agent.memory.add_to_memory = Mock()
        mock_agent.llm = Mock()
        mock_agent.tool_manager = Mock()
        mock_agent.tool_manager.get_all_tools_schema.return_value = {}

        # Mock the LLM response for planning
        mock_plan_response = Mock()
        mock_plan_response.choices = [Mock()]
        mock_plan_response.choices[0].message.content = "Test plan content"

        # Mock the LLM response for execution
        mock_exec_response = Mock()
        mock_exec_response.choices = [Mock()]
        mock_exec_response.choices[0].message = Mock()
        mock_exec_response.choices[0].message.tool_calls = [
            Mock(),
            Mock(),
        ]  # 2 tool calls

        mock_agent.llm.generate.side_effect = [mock_plan_response, mock_exec_response]

        reasoning = ReWOOReasoning(mock_agent)
        reasoning.execute_tool_call = Mock(
            return_value=Plan(step=1, llm_plan=mock_exec_response.choices[0].message)
        )

        result = reasoning.plan()

        assert isinstance(result, Plan)
        assert reasoning.remaining_tool_calls == 2
        assert reasoning.current_plan == mock_exec_response.choices[0].message
        assert reasoning.current_obs is not None
        mock_agent.generate_obs.assert_called_once()

    def test_plan_with_custom_prompt(self):
        """Test plan method with custom prompt."""
        mock_agent = Mock()
        mock_agent.generate_obs.return_value = Observation(
            step=1, self_state={}, local_state={}
        )
        mock_agent.memory = Mock()
        mock_agent.memory.format_long_term.return_value = "Long term memory"
        mock_agent.memory.format_short_term.return_value = "Short term memory"
        mock_agent.memory.add_to_memory = Mock()
        mock_agent.llm = Mock()
        mock_agent.tool_manager = Mock()
        mock_agent.tool_manager.get_all_tools_schema.return_value = {}

        # Mock the LLM response for planning
        mock_plan_response = Mock()
        mock_plan_response.choices = [Mock()]
        mock_plan_response.choices[0].message.content = "Custom plan content"

        # Mock the LLM response for execution
        mock_exec_response = Mock()
        mock_exec_response.choices = [Mock()]
        mock_exec_response.choices[0].message = Mock()
        mock_exec_response.choices[0].message.tool_calls = [Mock()]

        mock_agent.llm.generate.side_effect = [mock_plan_response, mock_exec_response]

        reasoning = ReWOOReasoning(mock_agent)
        reasoning.execute_tool_call = Mock(
            return_value=Plan(step=1, llm_plan=mock_exec_response.choices[0].message)
        )

        result = reasoning.plan(prompt="Custom prompt")

        assert isinstance(result, Plan)
        assert reasoning.remaining_tool_calls == 1

    def test_plan_with_selected_tools(self):
        """Test plan method with selected tools."""
        mock_agent = Mock()
        mock_agent.step_prompt = "Default step prompt"
        mock_agent.generate_obs.return_value = Observation(
            step=1, self_state={}, local_state={}
        )
        mock_agent.memory = Mock()
        mock_agent.memory.format_long_term.return_value = "Long term memory"
        mock_agent.memory.format_short_term.return_value = "Short term memory"
        mock_agent.memory.add_to_memory = Mock()
        mock_agent.llm = Mock()
        mock_agent.tool_manager = Mock()
        mock_agent.tool_manager.get_all_tools_schema.return_value = {}

        # Mock the LLM response for planning
        mock_plan_response = Mock()
        mock_plan_response.choices = [Mock()]
        mock_plan_response.choices[0].message.content = "Test plan content"

        # Mock the LLM response for execution
        mock_exec_response = Mock()
        mock_exec_response.choices = [Mock()]
        mock_exec_response.choices[0].message = Mock()
        mock_exec_response.choices[0].message.tool_calls = [Mock()]

        mock_agent.llm.generate.side_effect = [mock_plan_response, mock_exec_response]

        reasoning = ReWOOReasoning(mock_agent)
        reasoning.execute_tool_call = Mock(
            return_value=Plan(step=1, llm_plan=mock_exec_response.choices[0].message)
        )

        selected_tools = ["tool1", "tool2"]
        result = reasoning.plan(selected_tools=selected_tools)

        assert isinstance(result, Plan)
        mock_agent.tool_manager.get_all_tools_schema.assert_called_with(selected_tools)

    def test_plan_no_prompt_error(self):
        """Test plan method raises error when no prompt is provided."""
        mock_agent = Mock()
        mock_agent.step_prompt = None
        mock_agent.generate_obs.return_value = Observation(
            step=1, self_state={}, local_state={}
        )
        mock_agent.memory = Mock()

        reasoning = ReWOOReasoning(mock_agent)

        with pytest.raises(
            ValueError, match=r"No prompt provided and agent.step_prompt is None"
        ):
            reasoning.plan()

    def test_plan_with_no_tool_calls(self):
        """Test plan method when execution returns no tool calls."""
        mock_agent = Mock()
        mock_agent.step_prompt = "Default step prompt"
        mock_agent.generate_obs.return_value = Observation(
            step=1, self_state={}, local_state={}
        )
        mock_agent.memory = Mock()
        mock_agent.memory.format_long_term.return_value = "Long term memory"
        mock_agent.memory.format_short_term.return_value = "Short term memory"
        mock_agent.memory.add_to_memory = Mock()
        mock_agent.llm = Mock()
        mock_agent.tool_manager = Mock()
        mock_agent.tool_manager.get_all_tools_schema.return_value = {}

        # Mock the LLM response for planning
        mock_plan_response = Mock()
        mock_plan_response.choices = [Mock()]
        mock_plan_response.choices[0].message.content = "Test plan content"

        # Mock the LLM response for execution with no tool_calls attribute
        mock_exec_response = Mock()
        mock_exec_response.choices = [Mock()]
        mock_exec_response.choices[0].message = Mock()
        # Don't set tool_calls attribute, but mock the execution result properly

        mock_agent.llm.generate.side_effect = [mock_plan_response, mock_exec_response]

        # Create a mock plan that doesn't have tool_calls attribute
        mock_plan_without_tool_calls = Mock(spec=[])  # spec=[] means no attributes
        reasoning = ReWOOReasoning(mock_agent)
        reasoning.execute_tool_call = Mock(
            return_value=Plan(step=1, llm_plan=mock_plan_without_tool_calls)
        )

        result = reasoning.plan()

        assert isinstance(result, Plan)
        assert reasoning.remaining_tool_calls == 0

    def test_aplan_with_remaining_tool_calls(self):
        """Test aplan method when there are remaining tool calls."""
        mock_agent = Mock()
        mock_agent.generate_obs = Mock()

        reasoning = ReWOOReasoning(mock_agent)
        reasoning.remaining_tool_calls = 1
        reasoning.current_plan = Mock()

        # Create actual mock objects for tool calls
        mock_tool_1 = Mock()
        mock_tool_2 = Mock()
        reasoning.current_plan.tool_calls = [mock_tool_1, mock_tool_2]  # 2 tool calls
        reasoning.current_obs = Observation(step=1, self_state={}, local_state={})

        result = asyncio.run(reasoning.aplan("test prompt"))

        assert isinstance(result, Plan)
        assert result.llm_plan.tool_calls == [mock_tool_2]  # Should get index 1 (2-1)
        assert reasoning.remaining_tool_calls == 0
        mock_agent.generate_obs.assert_not_called()

    def test_aplan_new_plan_generation(self):
        """Test aplan method when generating a new plan."""
        mock_agent = Mock()
        mock_agent.generate_obs.return_value = Observation(
            step=1, self_state={}, local_state={}
        )
        mock_agent.memory = Mock()
        mock_agent.memory.format_long_term.return_value = "Long term memory"
        mock_agent.memory.format_short_term.return_value = "Short term memory"
        mock_agent.memory.add_to_memory = Mock()
        mock_agent.llm = Mock()
        mock_agent.tool_manager = Mock()
        mock_agent.tool_manager.get_all_tools_schema.return_value = {}

        # Mock the async LLM response for planning
        mock_plan_response = Mock()
        mock_plan_response.choices = [Mock()]
        mock_plan_response.choices[0].message.content = "Async plan content"

        # Mock the async LLM response for execution
        mock_exec_response = Mock()
        mock_exec_response.choices = [Mock()]
        mock_exec_response.choices[0].message = Mock()
        mock_exec_response.choices[0].message.tool_calls = [
            Mock(),
            Mock(),
            Mock(),
        ]  # 3 tool calls

        mock_agent.llm.agenerate = AsyncMock(
            side_effect=[mock_plan_response, mock_exec_response]
        )

        reasoning = ReWOOReasoning(mock_agent)
        reasoning.aexecute_tool_call = AsyncMock(
            return_value=Plan(step=1, llm_plan=mock_exec_response.choices[0].message)
        )

        result = asyncio.run(reasoning.aplan("test prompt"))

        assert isinstance(result, Plan)
        assert reasoning.remaining_tool_calls == 3
        assert reasoning.current_plan == mock_exec_response.choices[0].message
        assert reasoning.current_obs is not None
        mock_agent.generate_obs.assert_called_once()

    def test_aplan_with_selected_tools(self):
        """Test aplan method with selected tools."""
        mock_agent = Mock()
        mock_agent.generate_obs.return_value = Observation(
            step=1, self_state={}, local_state={}
        )
        mock_agent.memory = Mock()
        mock_agent.memory.format_long_term.return_value = "Long term memory"
        mock_agent.memory.format_short_term.return_value = "Short term memory"
        mock_agent.memory.add_to_memory = Mock()
        mock_agent.llm = Mock()
        mock_agent.tool_manager = Mock()
        mock_agent.tool_manager.get_all_tools_schema.return_value = {}

        # Mock the async LLM response for planning
        mock_plan_response = Mock()
        mock_plan_response.choices = [Mock()]
        mock_plan_response.choices[0].message.content = "Async plan content"

        # Mock the async LLM response for execution
        mock_exec_response = Mock()
        mock_exec_response.choices = [Mock()]
        mock_exec_response.choices[0].message = Mock()
        mock_exec_response.choices[0].message.tool_calls = [Mock()]

        mock_agent.llm.agenerate = AsyncMock(
            side_effect=[mock_plan_response, mock_exec_response]
        )

        reasoning = ReWOOReasoning(mock_agent)
        reasoning.aexecute_tool_call = AsyncMock(
            return_value=Plan(step=1, llm_plan=mock_exec_response.choices[0].message)
        )

        selected_tools = ["tool1", "tool2"]
        result = asyncio.run(
            reasoning.aplan("test prompt", selected_tools=selected_tools)
        )

        assert isinstance(result, Plan)
        mock_agent.tool_manager.get_all_tools_schema.assert_called_with(selected_tools)

    def test_aplan_with_no_tool_calls(self):
        """Test aplan method when execution returns no tool calls."""
        mock_agent = Mock()
        mock_agent.generate_obs.return_value = Observation(
            step=1, self_state={}, local_state={}
        )
        mock_agent.memory = Mock()
        mock_agent.memory.format_long_term.return_value = "Long term memory"
        mock_agent.memory.format_short_term.return_value = "Short term memory"
        mock_agent.memory.add_to_memory = Mock()
        mock_agent.llm = Mock()
        mock_agent.tool_manager = Mock()
        mock_agent.tool_manager.get_all_tools_schema.return_value = {}

        # Mock the async LLM response for planning
        mock_plan_response = Mock()
        mock_plan_response.choices = [Mock()]
        mock_plan_response.choices[0].message.content = "Async plan content"

        # Mock the async LLM response for execution with no tool_calls attribute
        mock_exec_response = Mock()
        mock_exec_response.choices = [Mock()]
        mock_exec_response.choices[0].message = Mock()
        # Don't set tool_calls attribute

        mock_agent.llm.agenerate = AsyncMock(
            side_effect=[mock_plan_response, mock_exec_response]
        )

        # Create a mock plan that doesn't have tool_calls attribute
        mock_plan_without_tool_calls = Mock(spec=[])  # spec=[] means no attributes
        reasoning = ReWOOReasoning(mock_agent)
        reasoning.aexecute_tool_call = AsyncMock(
            return_value=Plan(step=1, llm_plan=mock_plan_without_tool_calls)
        )

        result = asyncio.run(reasoning.aplan("test prompt"))

        assert isinstance(result, Plan)
        assert reasoning.remaining_tool_calls == 0

    def test_remaining_tool_calls_decrement(self):
        """Test that remaining_tool_calls is properly decremented."""
        mock_agent = Mock()
        mock_agent.generate_obs = Mock()

        reasoning = ReWOOReasoning(mock_agent)

        # Create actual mock objects for tool calls
        mock_tool_1 = Mock(name="tool_1")
        mock_tool_2 = Mock(name="tool_2")
        mock_tool_3 = Mock(name="tool_3")

        reasoning.current_obs = Observation(step=1, self_state={}, local_state={})

        # Test the index calculation logic separately for each scenario

        # Scenario 1: 3 remaining out of 3 total -> index should be 0
        reasoning.remaining_tool_calls = 3
        reasoning.current_plan = Mock()
        reasoning.current_plan.tool_calls = [mock_tool_1, mock_tool_2, mock_tool_3]

        result1 = reasoning.plan()
        assert reasoning.remaining_tool_calls == 2
        assert result1.llm_plan.tool_calls == [mock_tool_1]  # index 0 (3-3=0)

        # Scenario 2: 2 remaining out of 3 total -> index should be 1
        reasoning.remaining_tool_calls = 2
        reasoning.current_plan = Mock()
        reasoning.current_plan.tool_calls = [mock_tool_1, mock_tool_2, mock_tool_3]

        result2 = reasoning.plan()
        assert reasoning.remaining_tool_calls == 1
        assert result2.llm_plan.tool_calls == [mock_tool_2]  # index 1 (3-2=1)

        # Scenario 3: 1 remaining out of 3 total -> index should be 2
        reasoning.remaining_tool_calls = 1
        reasoning.current_plan = Mock()
        reasoning.current_plan.tool_calls = [mock_tool_1, mock_tool_2, mock_tool_3]

        result3 = reasoning.plan()
        assert reasoning.remaining_tool_calls == 0
        assert result3.llm_plan.tool_calls == [mock_tool_3]  # index 2 (3-1=2)
