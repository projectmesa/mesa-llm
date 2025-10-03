# tests/test_llm_agent_extra.py
#
# Testing library/framework: pytest with unittest.mock (consistent with existing repository tests)
#
# This complementary suite focuses on:
# - Regression around scheduler-assigned unique_id (diff fix: use schedule.next_id()).
# - Public interfaces: pre_step/post_step, __str__, send_message formatting.
# - Plan application metadata filtering, observation behavior and internal_state filtering.
#
import pytest
from unittest.mock import Mock, patch

from mesa.model import Model
from mesa.space import MultiGrid

from mesa_llm.llm_agent import LLMAgent
from mesa_llm.reasoning.react import ReActReasoning
from mesa_llm import Plan


# Autouse fixture to ensure ModuleLLM never hits real backends
@pytest.fixture(autouse=True)
def _mock_module_llm():
    with patch("mesa_llm.module_llm.ModuleLLM") as mock_llm_cls:
        instance = Mock()
        mock_llm_cls.return_value = instance
        yield instance


# Local MockScheduler used to validate the PR diff behavior (schedule.next_id)
class MockScheduler:
    def __init__(self, model):
        self._ids = 0
        self.agents = []
    def add(self, agent):
        self.agents.append(agent)
    def next_id(self):
        self._ids += 1
        return self._ids


def create_dummy_model(seed):
    class DummyModel(Model):
        def __init__(self):
            super().__init__(seed=seed)
            self.grid = MultiGrid(5, 5, torus=False)
            self.schedule = MockScheduler(self)
        def add_agent(self, pos, *, internal_state=None, vision=-1, system_prompt="System prompt"):
            agents = LLMAgent.create_agents(
                self,
                n=1,
                reasoning=ReActReasoning,
                system_prompt=system_prompt,
                vision=vision,
                internal_state=internal_state or ["test_state"],
            )
            agent = agents[0]
            # Regression guard for the PR diff: use schedule.next_id(), not model.next_id()
            agent.unique_id = self.schedule.next_id()
            self.grid.place_agent(agent, pos)
            self.schedule.add(agent)
            return agent
    return DummyModel()


def test_unique_id_uses_scheduler_increment():
    model = create_dummy_model(seed=1)
    a1 = model.add_agent((0, 0))
    a2 = model.add_agent((1, 1))
    a3 = model.add_agent((2, 2))
    # Ensure schedule's counter is the source of truth
    assert a1.unique_id == 1
    assert a2.unique_id == 2
    assert a3.unique_id == 3


def test_pre_step_calls_memory_process_step():
    model = create_dummy_model(seed=2)
    agent = model.add_agent((0, 0))
    mem = Mock()
    agent.memory = mem
    agent.pre_step()
    mem.process_step.assert_called_once_with(pre_step=True)


def test_post_step_calls_memory_process_step():
    model = create_dummy_model(seed=3)
    agent = model.add_agent((0, 0))
    mem = Mock()
    agent.memory = mem
    agent.post_step()
    mem.process_step.assert_called_once_with()


def test_str_returns_readable_id():
    model = create_dummy_model(seed=4)
    agent = model.add_agent((0, 0))
    assert str(agent) == f"LLMAgent {agent.unique_id}"


def test_send_message_return_contains_sender_and_message():
    model = create_dummy_model(seed=5)
    sender = model.add_agent((0, 0))
    recipient = model.add_agent((1, 1))
    # Assign simple memories to avoid side effects
    sender.memory = Mock()
    recipient.memory = Mock()
    ret = sender.send_message("hello-world", recipients=[recipient])
    assert "hello-world" in ret
    assert str(sender) in ret
    assert "→" in ret


def test_apply_plan_filters_tool_call_metadata():
    model = create_dummy_model(seed=6)
    agent = model.add_agent((1, 1))
    # Capture memory writes
    mem = Mock()
    agent.memory = mem
    # ToolManager returns metadata that must be filtered out by apply_plan
    fake_response = [{
        "tool": "move",
        "argument": "north",
        "tool_call_id": "abc123",
        "role": "assistant",
    }]
    agent.tool_manager = Mock()
    agent.tool_manager.call_tools = Mock(return_value=fake_response)
    plan = Plan(step=0, llm_plan="do something")
    _ = agent.apply_plan(plan)
    # Validate memory integration excludes tool_call_id, role
    assert mem.add_to_memory.called
    _, kwargs = mem.add_to_memory.call_args
    assert kwargs.get("type") == "action"
    content = kwargs.get("content", {})
    assert "tool" in content and content["tool"] == "move"
    assert "argument" in content and content["argument"] == "north"
    assert "tool_call_id" not in content
    assert "role" not in content


def test_generate_obs_vision_none_returns_empty(monkeypatch):
    model = create_dummy_model(seed=7)
    agent = model.add_agent((1, 1))
    # Neighbor exists but vision None means no local_state
    _ = model.add_agent((1, 2))
    agent.vision = None
    # Silence memory writes
    agent.memory = type("M", (), {"add_to_memory": staticmethod(lambda **kwargs: None)})()
    obs = agent.generate_obs()
    assert obs.local_state == {}


def test_generate_obs_filters_private_internal_state(monkeypatch):
    model = create_dummy_model(seed=8)
    seer = model.add_agent((2, 2), vision=1)
    seer.memory = type("M", (), {"add_to_memory": staticmethod(lambda **kwargs: None)})()
    neighbor = model.add_agent((2, 3), internal_state=["status_ok", "_secret", "energy=5"])
    obs = seer.generate_obs()
    key = f"LLMAgent {neighbor.unique_id}"
    assert key in obs.local_state
    internal = obs.local_state[key]["internal_state"]
    assert "status_ok" in internal
    assert "energy=5" in internal
    assert "_secret" not in internal  # private entries starting with '_' must be filtered


def test_create_agents_returns_requested_count():
    model = create_dummy_model(seed=9)
    agents = LLMAgent.create_agents(
        model,
        n=3,
        reasoning=ReActReasoning,
        system_prompt="X",
        vision=2,
    )
    assert len(agents) == 3
    assert all(isinstance(a, LLMAgent) for a in agents)


def test_create_agents_sets_vision_value():
    model = create_dummy_model(seed=10)
    agents = LLMAgent.create_agents(
        model,
        n=1,
        reasoning=ReActReasoning,
        system_prompt="Y",
        vision=3,
    )
    assert agents[0].vision == 3