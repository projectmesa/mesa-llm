# tests/test_reasoning/test_react.py

import asyncio
import json
from unittest.mock import AsyncMock, Mock

import pytest

from mesa_llm.reasoning.react import ReActOutput, ReActReasoning
from mesa_llm.reasoning.reasoning import Observation, Plan


class TestReActOutput:
    """Test the ReActOutput model."""

    def test_react_output_creation(self):
        """Test creating a ReActOutput with valid data."""
        output = ReActOutput(
            reasoning="I need to move to a better position", action="move_north"
        )

        assert output.reasoning == "I need to move to a better position"
        assert output.action == "move_north"


class TestReActReasoning:
    """Test the ReActReasoning class."""

    def test_react_reasoning_initialization(self):
        """Test ReActReasoning initialization."""
        mock_agent = Mock()
        reasoning = ReActReasoning(mock_agent)

        assert reasoning.agent == mock_agent

    def test_get_react_system_prompt(self):
        """Test get_react_system_prompt method."""
        mock_agent = Mock()
        reasoning = ReActReasoning(mock_agent)

        prompt = reasoning.get_react_system_prompt()

        assert "reasoning:" in prompt
        assert "action:" in prompt

    def test_get_react_prompt_with_observation(self):
        """Test get_react_prompt with observation."""
        mock_agent = Mock()
        mock_agent.memory = Mock()
        mock_agent.memory.get_prompt_ready.return_value = ["memory1", "memory2"]
        mock_agent.memory.get_communication_history.return_value = "communication"

        reasoning = ReActReasoning(mock_agent)

        obs = Observation(step=1, self_state={}, local_state={})
        prompt_list = reasoning.get_react_prompt(obs)

        assert len(prompt_list) >= 2
        assert "current observation" in prompt_list[-1]
        assert "last communication" in prompt_list[-2]

    def test_get_react_prompt_without_observation(self):
        """Test get_react_prompt without observation."""
        mock_agent = Mock()
        mock_agent.memory = Mock()
        mock_agent.memory.get_prompt_ready.return_value = ["memory1"]
        mock_agent.memory.get_communication_history.return_value = ""

        reasoning = ReActReasoning(mock_agent)

        prompt_list = reasoning.get_react_prompt(None)

        assert len(prompt_list) >= 1
        assert "last communication" not in prompt_list[-1]

    def test_plan_with_prompt(self):
        """Test plan method with custom prompt."""
        mock_agent = Mock()
        mock_agent.memory = Mock()
        mock_agent.memory.get_prompt_ready.return_value = ["memory1"]
        mock_agent.memory.get_communication_history.return_value = ""
        mock_agent.memory.add_to_memory = Mock()
        mock_agent.llm = Mock()
        mock_agent.tool_manager = Mock()
        mock_agent.tool_manager.get_all_tools_schema.return_value = {}

        # Mock the LLM response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps(
            {"reasoning": "Custom reasoning", "action": "custom_action"}
        )
        mock_agent.llm.generate.return_value = mock_response

        # Mock execute_tool_call
        mock_plan = Plan(step=1, llm_plan=Mock())
        reasoning = ReActReasoning(mock_agent)
        reasoning.execute_tool_call = Mock(return_value=mock_plan)

        obs = Observation(step=1, self_state={}, local_state={})
        result = reasoning.plan(obs=obs, prompt="Custom prompt")

        assert result == mock_plan
        reasoning.execute_tool_call.assert_called_once_with("custom_action", None)

    def test_plan_with_selected_tools(self):
        """Test plan method with selected tools."""
        mock_agent = Mock()
        mock_agent.step_prompt = "Default step prompt"
        mock_agent.memory = Mock()
        mock_agent.memory.get_prompt_ready.return_value = ["memory1"]
        mock_agent.memory.get_communication_history.return_value = ""
        mock_agent.memory.add_to_memory = Mock()
        mock_agent.llm = Mock()
        mock_agent.tool_manager = Mock()
        mock_agent.tool_manager.get_all_tools_schema.return_value = {}

        # Mock the LLM response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps(
            {"reasoning": "Test reasoning", "action": "test_action"}
        )
        mock_agent.llm.generate.return_value = mock_response

        # Mock execute_tool_call
        mock_plan = Plan(step=1, llm_plan=Mock())
        reasoning = ReActReasoning(mock_agent)
        reasoning.execute_tool_call = Mock(return_value=mock_plan)

        obs = Observation(step=1, self_state={}, local_state={})
        selected_tools = ["tool1", "tool2"]
        result = reasoning.plan(obs=obs, selected_tools=selected_tools)

        assert result == mock_plan
        mock_agent.tool_manager.get_all_tools_schema.assert_called_with(selected_tools)
        reasoning.execute_tool_call.assert_called_once_with(
            "test_action", selected_tools
        )

    def test_plan_no_prompt_error(self):
        """Test plan method raises error when no prompt is provided."""
        mock_agent = Mock()
        mock_agent.step_prompt = None
        mock_agent.memory = Mock()
        mock_agent.memory.get_prompt_ready.return_value = ["memory1"]
        mock_agent.memory.get_communication_history.return_value = ""

        reasoning = ReActReasoning(mock_agent)
        obs = Observation(step=1, self_state={}, local_state={})

        with pytest.raises(
            ValueError, match=r"No prompt provided and agent.step_prompt is None"
        ):
            reasoning.plan(obs=obs)

    def test_aplan_async_version(self):
        """Test aplan async method."""
        mock_agent = Mock()
        mock_agent.step_prompt = "Default step prompt"
        mock_agent.memory = Mock()
        mock_agent.memory.get_prompt_ready.return_value = ["memory1"]
        mock_agent.memory.get_communication_history.return_value = ""
        mock_agent.memory.add_to_memory = Mock()
        mock_agent.llm = Mock()
        mock_agent.tool_manager = Mock()
        mock_agent.tool_manager.get_all_tools_schema.return_value = {}

        # Mock the async LLM response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps(
            {"reasoning": "Async reasoning", "action": "async_action"}
        )
        mock_agent.llm.agenerate = AsyncMock(return_value=mock_response)

        # Mock aexecute_tool_call
        mock_plan = Plan(step=1, llm_plan=Mock())
        reasoning = ReActReasoning(mock_agent)
        reasoning.aexecute_tool_call = AsyncMock(return_value=mock_plan)

        obs = Observation(step=1, self_state={}, local_state={})

        # Test async execution
        result = asyncio.run(reasoning.aplan(obs=obs))

        assert result == mock_plan
        mock_agent.llm.agenerate.assert_called_once()
        reasoning.aexecute_tool_call.assert_called_once_with("async_action", None)

    def test_aplan_no_prompt_error(self):
        """Test aplan method raises error when no prompt is provided."""
        mock_agent = Mock()
        mock_agent.step_prompt = None
        mock_agent.memory = Mock()
        mock_agent.memory.get_prompt_ready.return_value = ["memory1"]
        mock_agent.memory.get_communication_history.return_value = ""

        reasoning = ReActReasoning(mock_agent)
        obs = Observation(step=1, self_state={}, local_state={})

        with pytest.raises(
            ValueError, match=r"No prompt provided and agent.step_prompt is None"
        ):
            asyncio.run(reasoning.aplan(obs=obs))
