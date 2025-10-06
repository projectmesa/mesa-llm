# tests/test_llm_agent.py


from mesa.model import Model
from mesa.space import MultiGrid

from mesa_llm import Plan
from mesa_llm.llm_agent import LLMAgent
from mesa_llm.memory.st_memory import ShortTermMemory
from mesa_llm.reasoning.react import ReActReasoning


# Create a Mock Scheduler to bypass any potential installation/import issues
class MockScheduler:
    def __init__(self, model):
        self._ids = 0
        self.agents = []

    def add(self, agent):
        self.agents.append(agent)

    def next_id(self):
        self._ids += 1
        return self._ids


# Helper function to create a standardized DummyModel to avoid repetition
def create_dummy_model(seed):
    class DummyModel(Model):
        def __init__(self):
            super().__init__(seed=seed)
            self.grid = MultiGrid(5, 5, torus=False)
            self.schedule = MockScheduler(self)  # Use the mock scheduler

        def add_agent(self, pos):
            agents = LLMAgent.create_agents(
                self,
                n=1,
                reasoning=ReActReasoning,
                system_prompt="System prompt",
                vision=-1,
                internal_state=["test_state"],
            )
            agent = agents[0]
            # THE FIX: Call next_id() on the schedule, not the model
            agent.unique_id = self.schedule.next_id()
            self.grid.place_agent(agent, pos)
            self.schedule.add(agent)
            return agent

    return DummyModel()


def test_apply_plan_adds_to_memory(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")
    model = create_dummy_model(seed=42)
    agent = model.add_agent((1, 1))
    agent.memory = ShortTermMemory(agent=agent, n=5, display=True)

    fake_response = [{"tool": "foo", "argument": "bar"}]
    monkeypatch.setattr(
        agent.tool_manager, "call_tools", lambda agent, llm_response: fake_response
    )

    plan = Plan(step=0, llm_plan="do something")
    resp = agent.apply_plan(plan)

    assert resp == fake_response
    assert {"tool": "foo", "argument": "bar"} in agent.memory.step_content.values()


def test_generate_obs_with_one_neighbor(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")
    model = create_dummy_model(seed=45)
    agent = model.add_agent((1, 1))
    agent.memory = ShortTermMemory(agent=agent, n=5)
    neighbor = model.add_agent((1, 2))
    monkeypatch.setattr(agent.memory, "add_to_memory", lambda *args, **kwargs: None)

    obs = agent.generate_obs()

    assert obs.self_state["agent_unique_id"] == agent.unique_id
    assert len(obs.local_state) == 1
    key = next(iter(obs.local_state.keys()))
    assert key == f"LLMAgent {neighbor.unique_id}"
    assert obs.local_state[key]["position"] == (1, 2)


def test_send_message_updates_both_agents_memory(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")
    model = create_dummy_model(seed=45)
    sender = model.add_agent((0, 0))
    sender.memory = ShortTermMemory(agent=sender, n=5)
    recipient = model.add_agent((1, 1))
    recipient.memory = ShortTermMemory(agent=recipient, n=5)

    call_counter = {"count": 0}

    def fake_add_to_memory(*args, **kwargs):
        call_counter["count"] += 1

    monkeypatch.setattr(sender.memory, "add_to_memory", fake_add_to_memory)
    monkeypatch.setattr(recipient.memory, "add_to_memory", fake_add_to_memory)

    sender.send_message("hello", recipients=[recipient])
    assert call_counter["count"] == 2


def test_generate_obs_zero_vision(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")
    model = create_dummy_model(seed=45)
    agent = model.add_agent((1, 1))
    agent.memory = ShortTermMemory(agent=agent, n=5)
    monkeypatch.setattr(agent.memory, "add_to_memory", lambda *args, **kwargs: None)
    _ = model.add_agent((1, 2))

    agent.vision = 0
    obs = agent.generate_obs()
    assert obs.local_state == {}


def test_generate_obs_limited_vision(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")
    model = create_dummy_model(seed=45)
    agent = model.add_agent((2, 2))
    agent.memory = ShortTermMemory(agent=agent, n=5)
    monkeypatch.setattr(agent.memory, "add_to_memory", lambda *args, **kwargs: None)

    neighbor = model.add_agent((2, 3))
    far_agent = model.add_agent((4, 4))

    agent.vision = 1
    obs = agent.generate_obs()

    assert len(obs.local_state) == 1
    assert f"LLMAgent {neighbor.unique_id}" in obs.local_state
    assert f"LLMAgent {far_agent.unique_id}" not in obs.local_state
