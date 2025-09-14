# tests/test_reasoning/test_cot.py

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest
from mesa.model import Model
from mesa.space import MultiGrid

from mesa_llm.llm_agent import LLMAgent
from mesa_llm.reasoning.cot import CoTReasoning
from mesa_llm.reasoning.reasoning import Observation, Plan


class TestCoTReasoning:
    def test_cot_reasoning_initialization(self):
        """Test CoTReasoning initialization."""
        mock_agent = Mock()
        reasoning = CoTReasoning(mock_agent)

        assert reasoning.agent == mock_agent

    def test_get_cot_system_prompt_with_memory(self):
        """Test get_cot_system_prompt with memory methods available."""
        mock_agent = Mock()
        mock_agent.memory = Mock()
        mock_agent.memory.format_long_term.return_value = "Long term memory content"
        mock_agent.memory.format_short_term.return_value = "Short term memory content"

        reasoning = CoTReasoning(mock_agent)

        obs = Observation(step=1, self_state={"test": "data"}, local_state={})
        prompt = reasoning.get_cot_system_prompt(obs)

        assert "Long term memory content" in prompt
        assert "Short term memory content" in prompt
        assert "Current Observation" in prompt
        assert "Thought 1:" in prompt
        assert "Action:" in prompt

    def test_plan_returns_proper_plan(self, monkeypatch):
        """
        Test CoTReasoning.plan without triggering _step_display_data and without calling the real LLM.
        """

        # Dummy model to initialize LLMAgent
        class DummyModel(Model):
            def __init__(self):
                super().__init__(seed=45)
                self.grid = MultiGrid(3, 3, torus=False)

        # Monkeypatch a dummy API key so ModuleLLM does not fail
        monkeypatch.setenv("GEMINI_API_KEY", "dummy")

        # Create an LLMAgent with CoTReasoning
        model = DummyModel()
        agent = LLMAgent(
            model=model,
            reasoning=CoTReasoning,
            step_prompt="you are an agent in a simulation",
        )
        mock_memory = Mock()
        agent.memory = mock_memory

        # Remove the attribute so `hasattr(..., "_step_display_data")` returns False
        if hasattr(agent.reasoning.agent, "_step_display_data"):
            delattr(agent.reasoning.agent, "_step_display_data")

        # Prepare mocked llm.generate() responses
        class MockResp:
            def __init__(self, content):
                self.choices = [
                    type("obj", (), {"message": type("mobj", (), {"content": content})})
                ]

        responses = iter([MockResp("mock plan content"), MockResp("mock execution")])

        def fake_generate(*args, **kwargs):
            return next(responses)

        # Patch llm.generate
        monkeypatch.setattr(agent.llm, "generate", fake_generate)

        # Create an observation (step 0 -> plan.step should be 1)
        obs = Observation(step=0, self_state={}, local_state={})

        plan = agent.reasoning.plan(obs)

        # Assertions
        assert isinstance(plan, Plan)
        assert plan.step == 1
        assert plan.llm_plan.content == "mock execution"
        # and our memory.add_to_memory should at least have been called once with type="observation"
        mock_memory.add_to_memory.assert_any_call(
            type="Observation",
            content=str(obs),
        )

    def test_plan_with_selected_tools(self):
        """Test plan method with selected tools."""
        mock_agent = Mock()
        mock_agent.step_prompt = "You are an agent in a simulatio"
        mock_agent.memory = Mock()
        mock_agent.memory.format_long_term.return_value = "Long term memory"
        mock_agent.memory.format_short_term.return_value = "Short term memory"
        mock_agent.memory.add_to_memory = Mock()
        mock_agent.llm = Mock()
        mock_agent.tool_manager = Mock()
        mock_agent.tool_manager.get_all_tools_schema.return_value = {}
        mock_agent._step_display_data = {}  # Use real dict instead of Mock
        # Mock the LLM response for planning
        mock_plan_response = Mock()
        mock_plan_response.choices = [Mock()]
        mock_plan_response.choices[
            0
        ].message.content = "Thought 1: Test reasoning\nAction: test_action"

        # Mock the LLM response for execution
        mock_exec_response = Mock()
        mock_exec_response.choices = [Mock()]
        mock_exec_response.choices[0].message = Mock()

        mock_agent.llm.generate.side_effect = [mock_plan_response, mock_exec_response]

        reasoning = CoTReasoning(mock_agent)

        obs = Observation(step=1, self_state={}, local_state={})
        selected_tools = ["tool1", "tool2"]
        result = reasoning.plan(obs=obs, selected_tools=selected_tools)

        assert isinstance(result, Plan)
        # Check that tool schema was called with selected tools
        assert mock_agent.tool_manager.get_all_tools_schema.call_count == 2

    def test_plan_no_prompt_error(self):
        """Test plan method raises error when no prompt is provided."""
        mock_agent = Mock()
        mock_agent.step_prompt = None
        mock_agent.memory = Mock()

        reasoning = CoTReasoning(mock_agent)
        obs = Observation(step=1, self_state={}, local_state={})

        with pytest.raises(
            ValueError, match=r"No prompt provided and agent.step_prompt is None"
        ):
            reasoning.plan(obs=obs)

    def test_aplan_async_version(self):
        """Test aplan async method."""
        mock_agent = Mock()
        mock_agent.memory = Mock()
        mock_agent.memory.format_long_term.return_value = "Long term memory"
        mock_agent.memory.format_short_term.return_value = "Short term memory"
        mock_agent.memory.add_to_memory = Mock()
        mock_agent.llm = Mock()
        mock_agent.tool_manager = Mock()
        mock_agent.tool_manager.get_all_tools_schema.return_value = {}
        mock_agent._step_display_data = {}  # Use real dict instead of Mock

        # Mock the async LLM response for planning
        mock_plan_response = Mock()
        mock_plan_response.choices = [Mock()]
        mock_plan_response.choices[
            0
        ].message.content = "Thought 1: Async reasoning\nAction: async_action"

        # Mock the async LLM response for execution
        mock_exec_response = Mock()
        mock_exec_response.choices = [Mock()]
        mock_exec_response.choices[0].message = Mock()

        mock_agent.llm.agenerate = AsyncMock(
            side_effect=[mock_plan_response, mock_exec_response]
        )

        reasoning = CoTReasoning(mock_agent)

        obs = Observation(step=1, self_state={}, local_state={})

        result = asyncio.run(reasoning.aplan(prompt="Async prompt", obs=obs))

        assert isinstance(result, Plan)
        assert result.step == 2
        assert mock_agent.llm.agenerate.call_count == 2
