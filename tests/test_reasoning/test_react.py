# tests/test_reasoning/test_react.py

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from mesa_llm.reasoning.react import ReActReasoning
from mesa_llm.reasoning.reasoning import Observation, Plan


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
        mock_agent.tool_manager.get_all_tools_schema.return_value = {}

        mock_agent.llm.generate.return_value = llm_response_factory(
            content="Custom reasoning"
        )

        reasoning = ReActReasoning(mock_agent)

        obs = Observation(step=1, self_state={}, local_state={})
        result = reasoning.plan(obs=obs, prompt="Custom prompt")

        assert result.step == mock_agent.model.steps
        assert result.llm_plan.content == "Custom reasoning"
        assert result.ttl == 1

    def test_plan_with_selected_tools(self, llm_response_factory, mock_agent):
        """Test plan method with selected tools."""
        mock_agent.step_prompt = "Default step prompt"
        mock_agent.memory = Mock()
        mock_agent.memory.get_prompt_ready.return_value = "memory1"
        mock_agent.memory.get_communication_history.return_value = ""
        mock_agent.memory.add_to_memory = Mock()
        mock_agent.llm = Mock()
        mock_agent.tool_manager = Mock()
        mock_agent.tool_manager.get_all_tools_schema.return_value = {}

        mock_agent.llm.generate.return_value = llm_response_factory(
            content="Test reasoning"
        )

        reasoning = ReActReasoning(mock_agent)

        obs = Observation(step=1, self_state={}, local_state={})
        selected_tools = ["tool1", "tool2"]
        result = reasoning.plan(obs=obs, ttl=3, selected_tools=selected_tools)

        assert result.step == mock_agent.model.steps
        assert result.llm_plan.content == "Test reasoning"
        assert result.ttl == 3
        mock_agent.tool_manager.get_all_tools_schema.assert_called_with(selected_tools)

    def test_plan_with_custom_tool_calls(self, llm_response_factory, mock_agent):
        """Test plan method forwards a custom execution tool choice."""
        mock_agent.step_prompt = "Default step prompt"
        mock_agent.memory = Mock()
        mock_agent.memory.get_prompt_ready.return_value = "memory1"
        mock_agent.memory.get_communication_history.return_value = ""
        mock_agent.memory.add_to_memory = Mock()
        mock_agent.llm = Mock()
        mock_agent.tool_manager = Mock()
        mock_agent.tool_manager.get_all_tools_schema.return_value = {}

        mock_agent.llm.generate.return_value = llm_response_factory(
            content="Test reasoning"
        )

        reasoning = ReActReasoning(mock_agent)

        obs = Observation(step=1, self_state={}, local_state={})
        result = reasoning.plan(obs=obs, tool_calls="required")

        assert result.step == mock_agent.model.steps
        assert result.llm_plan.content == "Test reasoning"
        assert result.ttl == 1
        assert mock_agent.llm.generate.call_args.kwargs["tool_choice"] == "required"

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
        mock_agent.tool_manager.get_all_tools_schema.return_value = {}

        mock_agent.llm.agenerate = AsyncMock(
            return_value=llm_response_factory(
                content="Async reasoning"
            )
        )

        reasoning = ReActReasoning(mock_agent)

        obs = Observation(step=1, self_state={}, local_state={})

        # Test async execution
        result = asyncio.run(reasoning.aplan(obs=obs, ttl=4))

        assert result.step == mock_agent.model.steps
        assert result.llm_plan.content == "Async reasoning"
        assert result.ttl == 4
        mock_agent.llm.agenerate.assert_called_once()
        assert mock_agent.llm.agenerate.call_args.kwargs["tool_choice"] == "auto"

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
        mock_agent.tool_manager.get_all_tools_schema.return_value = {}

        mock_agent.llm.generate.return_value = llm_response_factory(
            content="Test reasoning"
        )

        reasoning = ReActReasoning(mock_agent)

        obs = Observation(step=1, self_state={}, local_state={})
        expected_prompt = reasoning.get_react_system_prompt()
        reasoning.plan(obs=obs)

        assert mock_agent.llm.system_prompt == "base-system-prompt"
        assert (
            mock_agent.llm.generate.call_args.kwargs["system_prompt"] == expected_prompt
        )
