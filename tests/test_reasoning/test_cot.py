# tests/test_reasoning/test_cot.py

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest
from mesa.model import Model
from mesa.space import MultiGrid

from mesa_llm.llm_agent import LLMAgent
from mesa_llm.reasoning.cot import CoTReasoning
from mesa_llm.reasoning.reasoning import Observation, Plan
from mesa_llm.tools.tool_decorator import tool
from mesa_llm.tools.tool_manager import ToolManager


class TestCoTReasoning:
    def test_cot_reasoning_initialization(self, mock_agent):
        """Test CoTReasoning initialization."""
        reasoning = CoTReasoning(mock_agent)

        assert reasoning.agent == mock_agent

    def test_get_cot_system_prompt_with_memory(self, mock_agent):
        """Test get_cot_system_prompt with memory methods available."""
        mock_agent.system_prompt = "Agent persona"
        mock_agent.memory = Mock()
        mock_agent.memory.format_long_term.return_value = "Long term memory content"
        mock_agent.memory.format_short_term.return_value = "Short term memory content"

        reasoning = CoTReasoning(mock_agent)

        obs = Observation(step=1, self_state={"test": "data"}, local_state={})
        prompt = reasoning.get_cot_system_prompt(obs)

        assert "Long term memory content" in prompt
        assert "Short term memory content" in prompt
        assert "Agent Persona" in prompt
        assert "Agent persona" in prompt
        assert "Current Observation" in prompt
        assert "Thought 1:" in prompt
        assert "Action:" in prompt

    def test_get_cot_system_prompt_omits_empty_persona(self, mock_agent):
        """Empty agent persona should not add a persona section."""
        mock_agent.system_prompt = None
        mock_agent.memory = Mock()
        mock_agent.memory.format_long_term.return_value = "Long term memory content"
        mock_agent.memory.format_short_term.return_value = "Short term memory content"

        reasoning = CoTReasoning(mock_agent)
        obs = Observation(step=1, self_state={"test": "data"}, local_state={})

        prompt = reasoning.get_cot_system_prompt(obs)

        assert "Agent Persona" not in prompt

    def test_plan_returns_proper_plan(self, monkeypatch, llm_response_factory):
        """
        Test CoTReasoning.plan without triggering _step_display_data and without calling the real LLM.
        """

        # Dummy model to initialize LLMAgent
        class DummyModel(Model):
            def __init__(self):
                super().__init__(rng=45)
                self.grid = MultiGrid(3, 3, torus=False)

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

        responses = iter(
            [
                llm_response_factory(content="mock plan content"),
                llm_response_factory(content="mock execution"),
            ]
        )

        def fake_generate(*args, **kwargs):
            return next(responses)

        # Patch llm.generate
        monkeypatch.setattr(agent.llm, "generate", fake_generate)

        # Create an observation. Plan.step reflects the current model step.
        obs = Observation(step=0, self_state={}, local_state={})

        plan = agent.reasoning.plan(obs=obs)

        # Assertions
        assert isinstance(plan, Plan)
        assert plan.step == 0
        assert plan.llm_plan.content == "mock execution"
        assert plan.ttl == 1
        assert not any(
            call.kwargs.get("type") == "observation"
            for call in mock_memory.add_to_memory.call_args_list
        )

    def test_plan_does_not_write_observation_entries(
        self, llm_response_factory, mock_agent
    ):
        """CoT should not persist observations during planning."""
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
        mock_agent.tool_manager.get_tools_schema.return_value = {}
        mock_agent._step_display_data = {}

        mock_plan_response = llm_response_factory(
            content="Thought 1: reasoning\nAction: act"
        )
        mock_exec_response = llm_response_factory(content="executor response")
        mock_agent.llm.generate.side_effect = [mock_plan_response, mock_exec_response]

        reasoning = CoTReasoning(mock_agent)
        reasoning.plan(obs=Observation(step=1, self_state={}, local_state={}))

        calls = mock_agent.memory.add_to_memory.call_args_list
        assert not any(call.kwargs.get("type") == "observation" for call in calls)

    def test_plan_with_tools(self, llm_response_factory, mock_agent):
        """Test plan method with explicit tools."""
        mock_agent.step_prompt = "You are an agent in a simulatio"
        mock_agent.memory = Mock()
        mock_agent.memory.format_long_term.return_value = "Long term memory"
        mock_agent.memory.format_short_term.return_value = "Short term memory"
        mock_agent.memory.add_to_memory = Mock()
        mock_agent.llm = Mock()
        mock_agent.tool_manager = Mock()
        mock_agent.tool_manager.get_tools_schema.return_value = {}
        mock_agent._step_display_data = {}  # Use real dict instead of Mock
        mock_plan_response = llm_response_factory(
            content="Thought 1: Test reasoning\nAction: test_action"
        )
        mock_exec_response = llm_response_factory(content="executor response")

        mock_agent.llm.generate.side_effect = [mock_plan_response, mock_exec_response]

        reasoning = CoTReasoning(mock_agent)

        obs = Observation(step=1, self_state={}, local_state={})
        tools = ["tool1", "tool2"]
        result = reasoning.plan(obs=obs, ttl=3, tools=tools)

        assert isinstance(result, Plan)
        assert result.ttl == 3
        # Check that tool schema was called with explicit tools.
        assert mock_agent.tool_manager.get_tools_schema.call_count == 2
        mock_agent.tool_manager.get_tools_schema.assert_any_call(tools=tools)
        assert mock_agent.llm.generate.call_args_list[1].kwargs["tool_choice"] == "auto"

    def test_plan_tools_sentinel_semantics(self, llm_response_factory, mock_agent):
        """Omitted tools inherit; explicit per-call tools override."""

        @tool
        def inherited_cot_tool(agent, x: int) -> int:
            """Inherited CoT tool.
            Args:
                agent: The agent making the request (provided automatically)
                x: Input.
            Returns:
                Output.
            """
            return x

        @tool
        def override_cot_tool(agent, y: int) -> int:
            """Override CoT tool.
            Args:
                agent: The agent making the request (provided automatically)
                y: Input.
            Returns:
                Output.
            """
            return y

        mock_agent.step_prompt = "You are an agent in a simulation"
        mock_agent.memory = Mock()
        mock_agent.memory.format_long_term.return_value = "Long term memory"
        mock_agent.memory.format_short_term.return_value = "Short term memory"
        mock_agent.memory.add_to_memory = Mock()
        mock_agent.llm = Mock()
        mock_agent._tool_manager = ToolManager(
            tools=[inherited_cot_tool, override_cot_tool]
        )
        mock_agent._step_display_data = {}

        reasoning = CoTReasoning(mock_agent)
        obs = Observation(step=1, self_state={}, local_state={})

        def schema_names_for_plan(**kwargs):
            mock_plan_response = llm_response_factory(
                content="Thought 1: Test reasoning\nAction: test_action"
            )
            mock_exec_response = llm_response_factory(content="executor response")
            mock_agent.llm.generate.reset_mock()
            mock_agent.llm.generate.side_effect = [
                mock_plan_response,
                mock_exec_response,
            ]

            reasoning.plan(obs=obs, **kwargs)

            return [
                [schema["function"]["name"] for schema in call.kwargs["tool_schema"]]
                for call in mock_agent.llm.generate.call_args_list
            ]

        assert schema_names_for_plan() == [
            ["inherited_cot_tool", "override_cot_tool"],
            ["inherited_cot_tool", "override_cot_tool"],
        ]
        assert schema_names_for_plan(tools=None) == [[], []]
        assert schema_names_for_plan(tools=[]) == [[], []]
        assert schema_names_for_plan(tools=[override_cot_tool]) == [
            ["override_cot_tool"],
            ["override_cot_tool"],
        ]

    def test_plan_with_custom_tool_calls(self, llm_response_factory, mock_agent):
        """Test plan method forwards a custom execution tool choice."""
        mock_agent.step_prompt = "You are an agent in a simulation"
        mock_agent.memory = Mock()
        mock_agent.memory.format_long_term.return_value = "Long term memory"
        mock_agent.memory.format_short_term.return_value = "Short term memory"
        mock_agent.memory.add_to_memory = Mock()
        mock_agent.llm = Mock()
        mock_agent.tool_manager = Mock()
        mock_agent.tool_manager.get_tools_schema.return_value = {}
        mock_agent._step_display_data = {}
        mock_plan_response = llm_response_factory(
            content="Thought 1: Test reasoning\nAction: test_action"
        )
        mock_exec_response = llm_response_factory(content="executor response")
        mock_agent.llm.generate.side_effect = [mock_plan_response, mock_exec_response]

        reasoning = CoTReasoning(mock_agent)
        obs = Observation(step=1, self_state={}, local_state={})
        reasoning.plan(obs=obs, tool_calls="required")

        assert mock_agent.llm.generate.call_args_list[1].kwargs["tool_choice"] == (
            "required"
        )

    def test_plan_no_prompt_error(self, mock_agent):
        """Test plan method raises error when no prompt is provided."""
        mock_agent.step_prompt = None
        mock_agent.memory = Mock()

        reasoning = CoTReasoning(mock_agent)
        obs = Observation(step=1, self_state={}, local_state={})

        with pytest.raises(
            ValueError, match=r"No prompt provided and agent.step_prompt is None"
        ):
            reasoning.plan(obs=obs)

    def test_aplan_uses_step_prompt_when_no_prompt_given(
        self, llm_response_factory, mock_agent
    ):
        """Test aplan falls back to agent.step_prompt like sync plan does."""
        mock_agent.step_prompt = "Default step prompt"
        mock_agent.memory = Mock()
        mock_agent.memory.format_long_term.return_value = "Long term memory"
        mock_agent.memory.format_short_term.return_value = "Short term memory"
        mock_agent.memory.aadd_to_memory = AsyncMock()
        mock_agent.llm = Mock()
        mock_agent.tool_manager = Mock()
        mock_agent.tool_manager.get_tools_schema.return_value = {}
        mock_agent._step_display_data = {}

        mock_plan_response = llm_response_factory(
            content="Thought 1: reasoning\nAction: act"
        )
        mock_exec_response = llm_response_factory(content="executor response")
        mock_agent.llm.agenerate = AsyncMock(
            side_effect=[mock_plan_response, mock_exec_response]
        )

        reasoning = CoTReasoning(mock_agent)
        obs = Observation(step=1, self_state={}, local_state={})

        # Call without prompt — should use agent.step_prompt
        result = asyncio.run(reasoning.aplan(obs=obs))
        assert isinstance(result, Plan)

    def test_aplan_async_version(self, llm_response_factory, mock_agent):
        """Test aplan async method."""
        mock_agent.memory.format_long_term.return_value = "Long term memory"
        mock_agent.memory.format_short_term.return_value = "Short term memory"
        mock_agent.memory.add_to_memory = Mock()
        mock_agent.memory.aadd_to_memory = AsyncMock()
        mock_agent.llm = Mock()
        mock_agent.tool_manager = Mock()
        mock_agent.tool_manager.get_tools_schema.return_value = {}
        mock_agent._step_display_data = {}  # Use real dict instead of Mock

        mock_plan_response = llm_response_factory(
            content="Thought 1: Async reasoning\nAction: async_action"
        )
        mock_exec_response = llm_response_factory(content="async executor response")

        mock_agent.llm.agenerate = AsyncMock(
            side_effect=[mock_plan_response, mock_exec_response]
        )

        reasoning = CoTReasoning(mock_agent)

        obs = Observation(step=1, self_state={}, local_state={})

        result = asyncio.run(reasoning.aplan(prompt="Async prompt", obs=obs, ttl=4))

        assert isinstance(result, Plan)
        assert result.step == 1
        assert result.ttl == 4
        assert mock_agent.llm.agenerate.call_count == 2
        assert mock_agent.llm.agenerate.call_args_list[1].kwargs["tool_choice"] == (
            "auto"
        )

    def test_plan_uses_scoped_system_prompts(self, llm_response_factory, mock_agent):
        """CoT plan should pass prompts per call and avoid mutating llm.system_prompt."""
        mock_agent.step_prompt = "Default step prompt"
        mock_agent.llm.system_prompt = "base-system-prompt"
        mock_agent.memory = Mock()
        mock_agent.memory.format_long_term.return_value = "Long term memory"
        mock_agent.memory.format_short_term.return_value = "Short term memory"
        mock_agent.memory.add_to_memory = Mock()
        mock_agent.tool_manager = Mock()
        mock_agent.tool_manager.get_tools_schema.return_value = {}
        mock_agent._step_display_data = {}

        mock_plan_response = llm_response_factory(
            content="Thought 1: plan\nAction: move"
        )
        mock_exec_response = llm_response_factory(content="executor response")
        mock_agent.llm.generate = Mock(
            side_effect=[mock_plan_response, mock_exec_response]
        )

        reasoning = CoTReasoning(mock_agent)
        obs = Observation(step=1, self_state={}, local_state={})
        expected_plan_prompt = reasoning.get_cot_system_prompt(obs)

        reasoning.plan(obs=obs)

        assert mock_agent.llm.system_prompt == "base-system-prompt"
        assert (
            mock_agent.llm.generate.call_args_list[0].kwargs["system_prompt"]
            == expected_plan_prompt
        )
        assert (
            mock_agent.llm.generate.call_args_list[1].kwargs["system_prompt"]
            == "You are an executor that executes the plan given to you in the prompt through tool calls. "
            "If the plan concludes that no action should be taken, do not call any tool."
        )
