"""Integration tests: Memory backend x Reasoning strategy matrix.

Verifies that each memory implementation works correctly with each
reasoning strategy, testing the full flow:
    memory.get_prompt_ready() -> reasoning prompt construction ->
    reasoning.plan() / aplan() -> memory.add_to_memory()

These tests use real memory instances (not mocks) combined with
mocked LLM responses to isolate integration issues without
requiring API keys.
"""

import asyncio
import json
from collections import deque
from unittest.mock import AsyncMock, Mock

from mesa_llm.memory.episodic_memory import EpisodicMemory
from mesa_llm.memory.st_lt_memory import STLTMemory
from mesa_llm.memory.st_memory import ShortTermMemory
from mesa_llm.reasoning.cot import CoTReasoning
from mesa_llm.reasoning.react import ReActReasoning
from mesa_llm.reasoning.reasoning import Observation, Plan
from mesa_llm.reasoning.rewoo import ReWOOReasoning

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def make_mock_agent(memory_instance, *, step_prompt="You are an agent in a simulation"):
    """Build a mock agent wired to a *real* memory instance."""
    agent = Mock()
    agent.__class__.__name__ = "TestAgent"
    agent.unique_id = 1
    agent.model = Mock()
    agent.model.steps = 1
    agent.step_prompt = step_prompt
    agent.llm = Mock()
    agent.tool_manager = Mock()
    agent.tool_manager.get_tools_schema.return_value = {}
    agent._step_display_data = {}

    # Wire memory
    agent.memory = memory_instance
    memory_instance.agent = agent
    memory_instance.display = False  # suppress rich output in tests

    return agent


def make_llm_response(content="mock plan content", tool_calls=None):
    """Create a minimal mock LLM response."""
    rsp = Mock()
    rsp.choices = [Mock()]
    rsp.choices[0].message = Mock()
    rsp.choices[0].message.content = content
    rsp.choices[0].message.tool_calls = tool_calls
    return rsp


def make_react_response():
    """Create a mock LLM response in ReActOutput JSON format."""
    content = json.dumps({"reasoning": "test reasoning", "action": "test action"})
    return make_llm_response(content)


def seed_memory(memory, agent, n=2):
    """Add n dummy entries to memory so get_prompt_ready has content."""
    for i in range(n):
        memory.add_to_memory(type="observation", content={"info": f"step {i}"})
        memory.process_step(pre_step=True)
        agent.model.steps = i + 1
        memory.process_step(pre_step=False)


def make_short_term_memory():
    """Create a ShortTermMemory without triggering ModuleLLM init."""
    temp_agent = Mock()
    temp_agent.step_prompt = "test"
    memory = ShortTermMemory.__new__(ShortTermMemory)
    memory.n = 5
    memory.short_term_memory = deque()
    memory.step_content = {}
    memory.additive_event_types = {"message", "action"}
    memory.display = False
    memory.agent = temp_agent
    return memory


# ===================================================================
# CoT x Memory backends
# ===================================================================


class TestCoTWithShortTermMemory:
    """CoT reasoning with ShortTermMemory (no LLM consolidation needed)."""

    def _setup(self):
        memory = make_short_term_memory()
        agent = make_mock_agent(memory)
        return agent, memory, CoTReasoning(agent)

    def test_memory_prompt_used_in_reasoning(self):
        """CoT system prompt includes ShortTermMemory content."""
        agent, memory, reasoning = self._setup()
        seed_memory(memory, agent)

        obs = Observation(step=2, self_state={"health": 100}, local_state={})
        prompt = reasoning.get_cot_system_prompt(obs)

        # ShortTermMemory has format_short_term but NOT format_long_term
        assert "Short-Term Memory" in prompt or "Short" in prompt

    def test_plan_records_to_memory(self):
        """CoT plan() adds plan and plan_execution to memory."""
        agent, memory, reasoning = self._setup()

        plan_content = "Thought 1: reasoning\nAction: move north"
        rsp_plan = make_llm_response(plan_content)
        rsp_exec = make_llm_response("executing")
        agent.llm.generate = Mock(side_effect=[rsp_plan, rsp_exec])

        obs = Observation(step=0, self_state={}, local_state={})
        plan = reasoning.plan(obs=obs)

        assert isinstance(plan, Plan)
        assert plan.step == 1
        assert plan.llm_plan is rsp_exec.choices[0].message
        assert memory.step_content["plan"]["content"] == plan_content
        assert memory.step_content["plan_execution"]["content"] == str(plan)
        assert agent._step_display_data["plan_content"] == plan_content
        assert agent.llm.generate.call_count == 2
        assert "observation" not in memory.step_content

    def test_async_plan_works(self):
        """aplan() completes without error using ShortTermMemory."""
        agent, memory, reasoning = self._setup()

        plan_content = "Thought 1: async reasoning\nAction: act"
        rsp_plan = make_llm_response(plan_content)
        rsp_exec = make_llm_response("executing")
        agent.llm.agenerate = AsyncMock(side_effect=[rsp_plan, rsp_exec])
        agent.memory.aadd_to_memory = AsyncMock(side_effect=memory.add_to_memory)

        obs = Observation(step=0, self_state={}, local_state={})
        plan = asyncio.run(reasoning.aplan(obs=obs))

        assert isinstance(plan, Plan)
        assert plan.step == 1
        assert plan.llm_plan is rsp_exec.choices[0].message
        assert memory.step_content["plan"]["content"] == plan_content
        assert memory.step_content["plan_execution"]["content"] == str(plan)
        assert agent._step_display_data["plan_content"] == plan_content
        assert agent.llm.agenerate.await_count == 2
        assert "observation" not in memory.step_content


class TestCoTWithSTLTMemory:
    """CoT reasoning with STLTMemory (default memory, has both ST + LT)."""

    def _setup(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "dummy")
        agent = Mock()
        agent.__class__.__name__ = "TestAgent"
        agent.unique_id = 1
        agent.model = Mock()
        agent.model.steps = 1
        agent.step_prompt = "You are an agent"
        agent.llm = Mock()
        agent.tool_manager = Mock()
        agent.tool_manager.get_tools_schema.return_value = {}
        agent._step_display_data = {}

        memory = STLTMemory(
            agent=agent,
            short_term_capacity=5,
            consolidation_capacity=2,
            llm_model="openai/gpt-4o-mini",
            display=False,
        )
        agent.memory = memory
        return agent, memory, CoTReasoning(agent)

    def test_memory_prompt_used_in_reasoning(self, monkeypatch):
        """CoT system prompt includes both short-term and long-term content."""
        _agent, memory, reasoning = self._setup(monkeypatch)
        memory.long_term_memory = "Previously the agent explored the north."
        obs = Observation(step=2, self_state={}, local_state={})
        prompt = reasoning.get_cot_system_prompt(obs)

        assert "Previously the agent explored the north" in prompt

    def test_plan_records_to_memory(self, monkeypatch):
        """CoT plan() stores content into STLTMemory."""
        agent, memory, reasoning = self._setup(monkeypatch)

        plan_content = "Thought 1: reasoning\nAction: act"
        rsp_plan = make_llm_response(plan_content)
        rsp_exec = make_llm_response("executing")
        agent.llm.generate = Mock(side_effect=[rsp_plan, rsp_exec])

        obs = Observation(step=0, self_state={}, local_state={})
        plan = reasoning.plan(obs=obs)

        assert isinstance(plan, Plan)
        assert plan.step == 1
        assert memory.step_content["plan"]["content"] == plan_content
        assert memory.step_content["plan_execution"]["content"] == str(plan)
        assert agent.llm.generate.call_count == 2
        assert "observation" not in memory.step_content

    def test_async_plan_works(self, monkeypatch):
        """aplan() completes with STLTMemory."""
        agent, memory, reasoning = self._setup(monkeypatch)

        plan_content = "Thought 1: async\nAction: act"
        rsp_plan = make_llm_response(plan_content)
        rsp_exec = make_llm_response("executing")
        agent.llm.agenerate = AsyncMock(side_effect=[rsp_plan, rsp_exec])
        agent.memory.aadd_to_memory = AsyncMock(side_effect=memory.add_to_memory)

        obs = Observation(step=0, self_state={}, local_state={})
        plan = asyncio.run(reasoning.aplan(obs=obs))

        assert isinstance(plan, Plan)
        assert plan.step == 1
        assert memory.step_content["plan"]["content"] == plan_content
        assert memory.step_content["plan_execution"]["content"] == str(plan)
        assert agent.llm.agenerate.await_count == 2
        assert "observation" not in memory.step_content


class TestCoTWithEpisodicMemory:
    """CoT reasoning with EpisodicMemory (importance-graded entries)."""

    def _setup(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "dummy")
        agent = Mock()
        agent.__class__.__name__ = "TestAgent"
        agent.unique_id = 1
        agent.model = Mock()
        agent.model.steps = 1
        agent.step_prompt = "You are an agent"
        agent.llm = Mock()
        agent.tool_manager = Mock()
        agent.tool_manager.get_tools_schema.return_value = {}
        agent._step_display_data = {}

        memory = EpisodicMemory(
            agent=agent,
            llm_model="openai/gpt-4o-mini",
            display=False,
        )
        # Keep EpisodicMemory behavior but avoid extra grading LLM mocks in each test.
        memory.grade_event_importance = Mock(return_value=3)
        memory.agrade_event_importance = AsyncMock(return_value=3)
        agent.memory = memory
        return agent, memory, CoTReasoning(agent)

    def test_memory_prompt_used_in_reasoning(self, monkeypatch):
        """CoT system prompt works with EpisodicMemory (no format methods)."""
        _agent, _memory, reasoning = self._setup(monkeypatch)
        obs = Observation(step=1, self_state={}, local_state={})
        # Should not crash - CoT uses hasattr checks for format methods
        prompt = reasoning.get_cot_system_prompt(obs)
        assert "Current Observation" in prompt

    def test_plan_records_to_memory(self, monkeypatch):
        """CoT plan() writes graded entries into EpisodicMemory."""
        agent, memory, reasoning = self._setup(monkeypatch)

        plan_content = "Thought 1: reason\nAction: act"
        rsp_plan = make_llm_response(plan_content)
        rsp_exec = make_llm_response("done")

        agent.llm.generate = Mock(side_effect=[rsp_plan, rsp_exec])

        obs = Observation(step=0, self_state={}, local_state={})
        plan = reasoning.plan(obs=obs)

        assert isinstance(plan, Plan)
        entries = list(memory.memory_entries)
        assert len(entries) == 2
        assert entries[0].content["plan"]["content"] == plan_content
        assert entries[1].content["plan_execution"]["content"] == str(plan)
        assert entries[0].content["plan"]["importance"] == 3
        assert entries[1].content["plan_execution"]["importance"] == 3
        assert memory.grade_event_importance.call_count == 2

    def test_async_plan_works(self, monkeypatch):
        """aplan() completes with EpisodicMemory."""
        agent, memory, reasoning = self._setup(monkeypatch)

        plan_content = "Thought 1: async\nAction: act"
        rsp_plan = make_llm_response(plan_content)
        rsp_exec = make_llm_response("done")
        agent.llm.agenerate = AsyncMock(side_effect=[rsp_plan, rsp_exec])

        obs = Observation(step=0, self_state={}, local_state={})
        plan = asyncio.run(reasoning.aplan(obs=obs))

        assert isinstance(plan, Plan)
        entries = list(memory.memory_entries)
        assert len(entries) == 2
        assert entries[0].content["plan"]["content"] == plan_content
        assert entries[1].content["plan_execution"]["content"] == str(plan)
        assert entries[0].content["plan"]["importance"] == 3
        assert entries[1].content["plan_execution"]["importance"] == 3
        assert memory.agrade_event_importance.await_count == 2


# ===================================================================
# ReAct x Memory backends
# ===================================================================


class TestReActWithSTLTMemory:
    """ReAct with STLTMemory - the default and expected-to-work combination."""

    def _setup(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "dummy")
        agent = Mock()
        agent.__class__.__name__ = "TestAgent"
        agent.unique_id = 1
        agent.model = Mock()
        agent.model.steps = 1
        agent.step_prompt = "You are an agent"
        agent.llm = Mock()
        agent.tool_manager = Mock()
        agent.tool_manager.get_tools_schema.return_value = {}

        memory = STLTMemory(
            agent=agent,
            short_term_capacity=5,
            consolidation_capacity=2,
            llm_model="openai/gpt-4o-mini",
            display=False,
        )
        agent.memory = memory
        return agent, memory, ReActReasoning(agent)

    def test_get_prompt_ready_returns_str(self, monkeypatch):
        """STLTMemory.get_prompt_ready() returns a string memory snapshot."""
        _agent, memory, _reasoning = self._setup(monkeypatch)
        result = memory.get_prompt_ready()
        assert isinstance(result, str)

    def test_memory_prompt_used_in_reasoning(self, monkeypatch):
        """get_prompt_ready() string is wrapped and used in ReAct prompts."""
        _agent, memory, reasoning = self._setup(monkeypatch)
        memory.long_term_memory = "Agent explored north previously."

        obs = Observation(step=1, self_state={}, local_state={})
        prompt_list = reasoning.get_react_prompt(obs)

        assert isinstance(prompt_list, list)
        assert len(prompt_list) == 2
        assert "Long term memory" in prompt_list[0]
        assert "current observation" in prompt_list[1]
        assert str(obs) in prompt_list[1]

    def test_plan_records_to_memory(self, monkeypatch):
        """ReAct plan() stores formatted response to memory."""
        agent, memory, reasoning = self._setup(monkeypatch)

        rsp_react = make_react_response()
        rsp_exec = make_llm_response("executing")
        agent.llm.generate = Mock(side_effect=[rsp_react, rsp_exec])

        obs = Observation(step=0, self_state={}, local_state={})
        plan = reasoning.plan(obs=obs)

        assert isinstance(plan, Plan)
        assert memory.step_content["plan"]["reasoning"] == "test reasoning"
        assert memory.step_content["plan"]["action"] == "test action"
        assert memory.step_content["plan_execution"]["content"] == str(plan)
        assert agent.llm.generate.call_count == 2

    def test_async_plan_works(self, monkeypatch):
        """aplan() completes with STLTMemory."""
        agent, memory, reasoning = self._setup(monkeypatch)

        rsp_react = make_react_response()
        rsp_exec = make_llm_response("executing")
        agent.llm.agenerate = AsyncMock(side_effect=[rsp_react, rsp_exec])
        agent.memory.aadd_to_memory = AsyncMock(side_effect=memory.add_to_memory)

        obs = Observation(step=0, self_state={}, local_state={})
        plan = asyncio.run(reasoning.aplan(obs=obs))

        assert isinstance(plan, Plan)
        assert memory.step_content["plan"]["reasoning"] == "test reasoning"
        assert memory.step_content["plan"]["action"] == "test action"
        assert memory.step_content["plan_execution"]["content"] == str(plan)
        assert agent.llm.agenerate.await_count == 2


class TestReActWithShortTermMemory:
    """ReAct with ShortTermMemory."""

    def _setup(self):
        memory = make_short_term_memory()
        agent = make_mock_agent(memory)
        return agent, memory, ReActReasoning(agent)

    def test_get_prompt_ready_returns_str(self):
        """ShortTermMemory.get_prompt_ready() returns str, not list."""
        _agent, memory, _reasoning = self._setup()
        result = memory.get_prompt_ready()
        assert isinstance(result, str)

    def test_react_prompt_construction_handles_str_memory(self):
        """ReAct wraps string memory into a prompt list and appends context."""
        _agent, _memory, reasoning = self._setup()
        obs = Observation(step=1, self_state={}, local_state={})

        prompt_list = reasoning.get_react_prompt(obs)
        assert isinstance(prompt_list, list)
        assert len(prompt_list) == 2
        assert isinstance(prompt_list[0], str)
        assert "current observation" in prompt_list[-1]
        assert str(obs) in prompt_list[-1]


class TestReActWithEpisodicMemory:
    """ReAct with EpisodicMemory."""

    def _setup(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "dummy")
        agent = Mock()
        agent.__class__.__name__ = "TestAgent"
        agent.unique_id = 1
        agent.model = Mock()
        agent.model.steps = 1
        agent.step_prompt = "You are an agent"
        agent.llm = Mock()
        agent.tool_manager = Mock()
        agent.tool_manager.get_tools_schema.return_value = {}

        memory = EpisodicMemory(
            agent=agent,
            llm_model="openai/gpt-4o-mini",
            display=False,
        )
        agent.memory = memory
        return agent, memory, ReActReasoning(agent)

    def test_get_prompt_ready_returns_str(self, monkeypatch):
        """EpisodicMemory.get_prompt_ready() returns str, not list."""
        _agent, memory, _reasoning = self._setup(monkeypatch)
        result = memory.get_prompt_ready()
        assert isinstance(result, str)

    def test_react_prompt_construction_handles_str_memory(self, monkeypatch):
        """ReAct wraps EpisodicMemory string context into a prompt list."""
        _agent, _memory, reasoning = self._setup(monkeypatch)
        obs = Observation(step=1, self_state={}, local_state={})

        prompt_list = reasoning.get_react_prompt(obs)
        assert isinstance(prompt_list, list)
        assert len(prompt_list) == 2
        assert isinstance(prompt_list[0], str)
        assert "current observation" in prompt_list[-1]
        assert str(obs) in prompt_list[-1]


# ===================================================================
# ReWOO x Memory backends
# ===================================================================


class TestReWOOWithShortTermMemory:
    """ReWOO with ShortTermMemory."""

    def _setup(self):
        memory = make_short_term_memory()
        agent = make_mock_agent(memory)
        default_obs = Observation(step=1, self_state={}, local_state={})
        agent.generate_obs = Mock(return_value=default_obs)
        agent.agenerate_obs = AsyncMock(return_value=default_obs)
        return agent, memory, ReWOOReasoning(agent)

    def test_memory_prompt_used_in_reasoning(self):
        """ReWOO system prompt includes ShortTermMemory content."""
        agent, memory, reasoning = self._setup()
        seed_memory(memory, agent)

        reasoning.current_obs = Observation(step=2, self_state={}, local_state={})
        prompt = reasoning.get_rewoo_system_prompt(reasoning.current_obs)

        assert "Short-Term Memory" in prompt or "Short" in prompt

    def test_plan_records_to_memory(self):
        """ReWOO plan() stores plan content to memory."""
        agent, memory, reasoning = self._setup()

        plan_content = "multi-step plan content"
        rsp_plan = make_llm_response(plan_content)
        rsp_exec = make_llm_response("executing step 1", tool_calls=[])
        agent.llm.generate = Mock(side_effect=[rsp_plan, rsp_exec])

        plan = reasoning.plan()
        assert isinstance(plan, Plan)
        assert memory.step_content["plan"]["content"] == plan_content
        assert memory.step_content["plan_execution"]["content"] == str(plan)
        assert agent.llm.generate.call_count == 2
        agent.generate_obs.assert_called_once()

    def test_async_plan_works(self):
        """aplan() completes with ShortTermMemory."""
        agent, memory, reasoning = self._setup()

        plan_content = "async plan content"
        rsp_plan = make_llm_response(plan_content)
        rsp_exec = make_llm_response("executing", tool_calls=[])
        agent.llm.agenerate = AsyncMock(side_effect=[rsp_plan, rsp_exec])

        plan = asyncio.run(reasoning.aplan())
        assert isinstance(plan, Plan)
        assert reasoning.current_obs == agent.agenerate_obs.return_value
        assert reasoning.remaining_tool_calls == 0
        assert memory.step_content["plan"]["content"] == plan_content
        assert memory.step_content["plan_execution"]["content"] == str(plan)
        assert agent.llm.agenerate.await_count == 2
        agent.agenerate_obs.assert_awaited_once()


class TestReWOOWithSTLTMemory:
    """ReWOO with STLTMemory."""

    def _setup(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "dummy")
        agent = Mock()
        agent.__class__.__name__ = "TestAgent"
        agent.unique_id = 1
        agent.model = Mock()
        agent.model.steps = 1
        agent.step_prompt = "You are an agent"
        agent.llm = Mock()
        agent.tool_manager = Mock()
        agent.tool_manager.get_tools_schema.return_value = {}
        agent._step_display_data = {}
        default_obs = Observation(step=1, self_state={}, local_state={})
        agent.generate_obs = Mock(return_value=default_obs)
        agent.agenerate_obs = AsyncMock(return_value=default_obs)

        memory = STLTMemory(
            agent=agent,
            short_term_capacity=5,
            consolidation_capacity=2,
            llm_model="openai/gpt-4o-mini",
            display=False,
        )
        agent.memory = memory
        return agent, memory, ReWOOReasoning(agent)

    def test_memory_prompt_used_in_reasoning(self, monkeypatch):
        """ReWOO system prompt includes both ST and LT content."""
        _agent, memory, reasoning = self._setup(monkeypatch)
        memory.long_term_memory = "Agent was exploring east."
        reasoning.current_obs = Observation(step=1, self_state={}, local_state={})

        prompt = reasoning.get_rewoo_system_prompt(reasoning.current_obs)
        assert "Agent was exploring east" in prompt

    def test_plan_records_to_memory(self, monkeypatch):
        """ReWOO plan() stores to STLTMemory."""
        agent, memory, reasoning = self._setup(monkeypatch)

        plan_content = "rewoo plan"
        rsp_plan = make_llm_response(plan_content)
        rsp_exec = make_llm_response("exec", tool_calls=[])
        agent.llm.generate = Mock(side_effect=[rsp_plan, rsp_exec])

        plan = reasoning.plan()
        assert isinstance(plan, Plan)
        assert memory.step_content["plan"]["content"] == plan_content
        assert memory.step_content["plan_execution"]["content"] == str(plan)
        assert agent.llm.generate.call_count == 2
        agent.generate_obs.assert_called_once()

    def test_async_plan_works(self, monkeypatch):
        """aplan() completes with STLTMemory."""
        agent, memory, reasoning = self._setup(monkeypatch)

        plan_content = "async rewoo plan"
        rsp_plan = make_llm_response(plan_content)
        rsp_exec = make_llm_response("exec", tool_calls=[])
        agent.llm.agenerate = AsyncMock(side_effect=[rsp_plan, rsp_exec])

        plan = asyncio.run(reasoning.aplan())
        assert isinstance(plan, Plan)
        assert memory.step_content["plan"]["content"] == plan_content
        assert memory.step_content["plan_execution"]["content"] == str(plan)
        assert agent.llm.agenerate.await_count == 2
        agent.agenerate_obs.assert_awaited_once()


class TestReWOOWithEpisodicMemory:
    """ReWOO with EpisodicMemory."""

    def _setup(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "dummy")
        agent = Mock()
        agent.__class__.__name__ = "TestAgent"
        agent.unique_id = 1
        agent.model = Mock()
        agent.model.steps = 1
        agent.step_prompt = "You are an agent"
        agent.llm = Mock()
        agent.tool_manager = Mock()
        agent.tool_manager.get_tools_schema.return_value = {}
        agent._step_display_data = {}
        default_obs = Observation(step=1, self_state={}, local_state={})
        agent.generate_obs = Mock(return_value=default_obs)
        agent.agenerate_obs = AsyncMock(return_value=default_obs)

        memory = EpisodicMemory(
            agent=agent,
            llm_model="openai/gpt-4o-mini",
            display=False,
        )
        # Keep EpisodicMemory behavior but avoid extra grading LLM mocks in each test.
        memory.grade_event_importance = Mock(return_value=3)
        memory.agrade_event_importance = AsyncMock(return_value=3)
        agent.memory = memory
        return agent, memory, ReWOOReasoning(agent)

    def test_memory_prompt_used_in_reasoning(self, monkeypatch):
        """ReWOO system prompt works with EpisodicMemory (no format methods)."""
        _agent, _memory, reasoning = self._setup(monkeypatch)
        reasoning.current_obs = Observation(step=1, self_state={}, local_state={})

        prompt = reasoning.get_rewoo_system_prompt(reasoning.current_obs)
        assert "Current Observation" in prompt

    def test_plan_records_to_memory(self, monkeypatch):
        """ReWOO plan() writes graded entries into EpisodicMemory."""
        agent, memory, reasoning = self._setup(monkeypatch)

        plan_content = "episodic rewoo plan"
        rsp_plan = make_llm_response(plan_content)
        rsp_exec = make_llm_response("exec", tool_calls=[])
        agent.llm.generate = Mock(side_effect=[rsp_plan, rsp_exec])

        plan = reasoning.plan()
        assert isinstance(plan, Plan)
        entries = list(memory.memory_entries)
        assert len(entries) == 2
        assert entries[0].content["plan"]["content"] == plan_content
        assert entries[1].content["plan_execution"]["content"] == str(plan)
        assert entries[0].content["plan"]["importance"] == 3
        assert entries[1].content["plan_execution"]["importance"] == 3
        assert memory.grade_event_importance.call_count == 2

    def test_async_plan_works(self, monkeypatch):
        """aplan() completes with EpisodicMemory."""
        agent, memory, reasoning = self._setup(monkeypatch)

        plan_content = "async episodic plan"
        rsp_plan = make_llm_response(plan_content)
        rsp_exec = make_llm_response("exec", tool_calls=[])
        agent.llm.agenerate = AsyncMock(side_effect=[rsp_plan, rsp_exec])

        plan = asyncio.run(reasoning.aplan())
        assert isinstance(plan, Plan)
        entries = list(memory.memory_entries)
        assert len(entries) == 2
        assert entries[0].content["plan"]["content"] == plan_content
        assert entries[1].content["plan_execution"]["content"] == str(plan)
        assert entries[0].content["plan"]["importance"] == 3
        assert entries[1].content["plan_execution"]["importance"] == 3
        assert memory.agrade_event_importance.await_count == 2
        assert agent.llm.agenerate.await_count == 2
        agent.agenerate_obs.assert_awaited_once()
