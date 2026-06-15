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

    def test_react_output_schema_includes_field_descriptions(self):
        """Structured output schema should keep the field guidance text."""
        schema = ReActOutput.model_json_schema()

        assert schema["properties"]["reasoning"]["description"] == (
            "Step-by-step reasoning about the situation based on memory and observation"
        )
        assert schema["properties"]["action"]["description"] == (
            "The specific action to take without using tools"
        )


class TestReActReasoning:
    """Test the ReActReasoning class."""

    def test_react_reasoning_initialization(self, mock_agent):
        """Test ReActReasoning initialization."""
        reasoning = ReActReasoning(mock_agent)

        assert reasoning.agent == mock_agent

    def test_get_react_system_prompt(self, mock_agent):
        """Test get_react_system_prompt method."""
        mock_agent.system_prompt = "Agent persona"
        reasoning = ReActReasoning(mock_agent)

        prompt = reasoning.get_react_system_prompt()

        assert "Agent Persona" in prompt
        assert "Agent persona" in prompt
        assert "reasoning:" in prompt
        assert "action:" in prompt

    def test_get_react_system_prompt_omits_empty_persona(self, mock_agent):
        """Empty agent persona should not add a persona section."""
        mock_agent.system_prompt = None
        reasoning = ReActReasoning(mock_agent)

        prompt = reasoning.get_react_system_prompt()

        assert "Agent Persona" not in prompt

    def test_get_react_prompt_with_observation(self, mock_agent):
        """Test get_react_prompt with observation."""
        mock_agent.memory = Mock()
        mock_agent.memory.get_prompt_ready.return_value = "memory1\n\nmemory2"
        mock_agent.memory.get_communication_history.return_value = "communication"

        reasoning = ReActReasoning(mock_agent)

        obs = Observation(step=1, self_state={}, local_state={})
        prompt_list = reasoning.get_react_prompt(obs)

        assert len(prompt_list) >= 2
        assert "current observation" in prompt_list[-1]
        assert "last communication" in prompt_list[-2]

    def test_get_react_prompt_without_observation(self, mock_agent):
        """Test get_react_prompt without observation."""
        mock_agent.memory = Mock()
        mock_agent.memory.get_prompt_ready.return_value = "memory1"
        mock_agent.memory.get_communication_history.return_value = ""

        reasoning = ReActReasoning(mock_agent)

        prompt_list = reasoning.get_react_prompt(None)

        assert len(prompt_list) >= 1
        assert "last communication" not in prompt_list[-1]

    def test_plan_with_prompt(self, llm_response_factory, mock_agent):
        """Test plan method with custom prompt."""
        mock_agent.memory = Mock()
        mock_agent.memory.get_prompt_ready.return_value = "memory1"
        mock_agent.memory.get_communication_history.return_value = ""
        mock_agent.memory.add_to_memory = Mock()
        mock_agent.llm = Mock()
        mock_agent.tool_manager = Mock()
        mock_agent.tool_manager.get_tools_schema.return_value = {}

        mock_agent.llm.generate.return_value = llm_response_factory(
            content=json.dumps(
                {"reasoning": "Custom reasoning", "action": "custom_action"}
            )
        )

        # Mock execute_tool_call
        mock_plan = Plan(step=1, llm_plan=Mock())
        reasoning = ReActReasoning(mock_agent)
        reasoning.execute_tool_call = Mock(return_value=mock_plan)

        obs = Observation(step=1, self_state={}, local_state={})
        result = reasoning.plan(obs=obs, prompt="Custom prompt")

        assert result == mock_plan
        reasoning.execute_tool_call.assert_called_once_with(
            "custom_action",
            ttl=1,
            tool_calls="auto",
        )

    def test_plan_with_tools(self, llm_response_factory, mock_agent):
        """Test plan method with explicit tools."""
        mock_agent.step_prompt = "Default step prompt"
        mock_agent.memory = Mock()
        mock_agent.memory.get_prompt_ready.return_value = "memory1"
        mock_agent.memory.get_communication_history.return_value = ""
        mock_agent.memory.add_to_memory = Mock()
        mock_agent.llm = Mock()
        mock_agent.tool_manager = Mock()
        mock_agent.tool_manager.get_tools_schema.return_value = {}

        mock_agent.llm.generate.return_value = llm_response_factory(
            content=json.dumps({"reasoning": "Test reasoning", "action": "test_action"})
        )

        # Mock execute_tool_call
        mock_plan = Plan(step=1, llm_plan=Mock())
        reasoning = ReActReasoning(mock_agent)
        reasoning.execute_tool_call = Mock(return_value=mock_plan)

        obs = Observation(step=1, self_state={}, local_state={})
        tools = ["tool1", "tool2"]
        result = reasoning.plan(obs=obs, ttl=3, tools=tools)

        assert result == mock_plan
        mock_agent.tool_manager.get_tools_schema.assert_called_with(tools=tools)
        reasoning.execute_tool_call.assert_called_once_with(
            "test_action",
            tools=tools,
            ttl=3,
            tool_calls="auto",
        )

    def test_plan_with_custom_tool_calls(self, llm_response_factory, mock_agent):
        """Test plan method forwards a custom execution tool choice."""
        mock_agent.step_prompt = "Default step prompt"
        mock_agent.memory = Mock()
        mock_agent.memory.get_prompt_ready.return_value = "memory1"
        mock_agent.memory.get_communication_history.return_value = ""
        mock_agent.memory.add_to_memory = Mock()
        mock_agent.llm = Mock()
        mock_agent.tool_manager = Mock()
        mock_agent.tool_manager.get_tools_schema.return_value = {}

        mock_agent.llm.generate.return_value = llm_response_factory(
            content=json.dumps({"reasoning": "Test reasoning", "action": "test_action"})
        )

        mock_plan = Plan(step=1, llm_plan=Mock())
        reasoning = ReActReasoning(mock_agent)
        reasoning.execute_tool_call = Mock(return_value=mock_plan)

        obs = Observation(step=1, self_state={}, local_state={})
        result = reasoning.plan(obs=obs, tool_calls="required")

        assert result == mock_plan
        reasoning.execute_tool_call.assert_called_once_with(
            "test_action",
            ttl=1,
            tool_calls="required",
        )

    def test_plan_no_prompt_error(self, mock_agent):
        """Test plan method raises error when no prompt is provided."""
        mock_agent.step_prompt = None
        mock_agent.memory = Mock()
        mock_agent.memory.get_prompt_ready.return_value = "memory1"
        mock_agent.memory.get_communication_history.return_value = ""

        reasoning = ReActReasoning(mock_agent)
        obs = Observation(step=1, self_state={}, local_state={})

        with pytest.raises(
            ValueError, match=r"No prompt provided and agent.step_prompt is None"
        ):
            reasoning.plan(obs=obs)

    def test_aplan_async_version(self, llm_response_factory, mock_agent):
        """Test aplan async method."""
        mock_agent.step_prompt = "Default step prompt"
        mock_agent.memory = Mock()
        mock_agent.memory.get_prompt_ready.return_value = "memory1"
        mock_agent.memory.get_communication_history.return_value = ""
        mock_agent.memory.add_to_memory = Mock()
        mock_agent.memory.aadd_to_memory = AsyncMock()
        mock_agent.llm = Mock()
        mock_agent.tool_manager = Mock()
        mock_agent.tool_manager.get_tools_schema.return_value = {}

        mock_agent.llm.agenerate = AsyncMock(
            return_value=llm_response_factory(
                content=json.dumps(
                    {"reasoning": "Async reasoning", "action": "async_action"}
                )
            )
        )

        # Mock aexecute_tool_call
        mock_plan = Plan(step=1, llm_plan=Mock())
        reasoning = ReActReasoning(mock_agent)
        reasoning.aexecute_tool_call = AsyncMock(return_value=mock_plan)

        obs = Observation(step=1, self_state={}, local_state={})

        # Test async execution
        result = asyncio.run(reasoning.aplan(obs=obs, ttl=4))

        assert result == mock_plan
        mock_agent.llm.agenerate.assert_called_once()
        reasoning.aexecute_tool_call.assert_called_once_with(
            "async_action",
            ttl=4,
            tool_calls="auto",
        )

    def test_aplan_no_prompt_error(self, mock_agent):
        """Test aplan method raises error when no prompt is provided."""
        mock_agent.step_prompt = None
        mock_agent.memory = Mock()
        mock_agent.memory.get_prompt_ready.return_value = "memory1"
        mock_agent.memory.get_communication_history.return_value = ""

        reasoning = ReActReasoning(mock_agent)
        obs = Observation(step=1, self_state={}, local_state={})

        with pytest.raises(
            ValueError, match=r"No prompt provided and agent.step_prompt is None"
        ):
            asyncio.run(reasoning.aplan(obs=obs))

    def test_plan_uses_scoped_system_prompt(self, llm_response_factory, mock_agent):
        """ReAct plan should pass system prompt per call and not mutate llm state."""
        mock_agent.step_prompt = "Default step prompt"
        mock_agent.llm.system_prompt = "base-system-prompt"
        mock_agent.memory = Mock()
        mock_agent.memory.get_prompt_ready.return_value = "memory1"
        mock_agent.memory.get_communication_history.return_value = ""
        mock_agent.memory.add_to_memory = Mock()
        mock_agent.tool_manager = Mock()
        mock_agent.tool_manager.get_tools_schema.return_value = {}

        mock_agent.llm.generate.return_value = llm_response_factory(
            content=json.dumps({"reasoning": "Test reasoning", "action": "test_action"})
        )

        reasoning = ReActReasoning(mock_agent)
        reasoning.execute_tool_call = Mock(return_value=Plan(step=1, llm_plan=Mock()))

        obs = Observation(step=1, self_state={}, local_state={})
        expected_prompt = reasoning.get_react_system_prompt()
        reasoning.plan(obs=obs)

        assert mock_agent.llm.system_prompt == "base-system-prompt"
        assert (
            mock_agent.llm.generate.call_args.kwargs["system_prompt"] == expected_prompt
        )
