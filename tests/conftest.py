import os
from unittest.mock import Mock, patch

import pytest
from litellm import Choices, Message, ModelResponse
from mesa.model import Model
from mesa.space import MultiGrid

from mesa_llm.llm_agent import LLMAgent
from mesa_llm.memory.st_memory import ShortTermMemory
from mesa_llm.reasoning.react import ReActReasoning

# Module-level constants
DEFAULT_AGENT_CONFIG = {
    "reasoning": ReActReasoning,
    "system_prompt": "Test",
    "internal_state": ["test"],
}


class MockCell:
    """Mock cell with coordinate attribute"""

    def __init__(self, coordinate):
        self.coordinate = coordinate


@pytest.fixture(autouse=True)
def mock_environment():
    """Ensure tests don't depend on real environment variables"""
    with patch.dict(
        os.environ,
        {
            "GEMINI_API_KEY": "dummy",
            "PROVIDER_API_KEY": "test_key",
            "OPENAI_API_KEY": "test_openai_key",
        },
        clear=True,
    ):
        yield


@pytest.fixture
def mock_agent():
    """Create a mock agent for testing"""
    agent = Mock()
    agent.__class__.__name__ = "TestAgent"
    agent.unique_id = 123
    agent.__str__ = Mock(return_value="TestAgent(123)")
    agent.model = Mock()
    agent.model.steps = 1
    agent.model.events = []
    agent.step_prompt = "Test step prompt"
    agent.llm = Mock()
    return agent


@pytest.fixture
def mock_llm():
    """Create a mock LLM for testing"""
    with patch("mesa_llm.module_llm.ModuleLLM") as mock_llm_class:
        mock_llm_instance = Mock()
        mock_llm_class.return_value = mock_llm_instance
        yield mock_llm_instance


def build_llm_response(
    content: str = "ok",
    *,
    role: str = "assistant",
    tool_calls: list | None = None,
    response_id: str = "mock-response-id",
    model: str = "openai/mock-model",
    created: int = 0,
) -> ModelResponse:
    """Create a typed LiteLLM response object for tests."""
    return ModelResponse(
        id=response_id,
        created=created,
        model=model,
        object="chat.completion",
        choices=[
            Choices(
                message=Message(
                    role=role,
                    content=content,
                    tool_calls=tool_calls,
                )
            )
        ],
    )


@pytest.fixture
def llm_response_factory():
    """Fixture wrapper around typed LiteLLM response builder."""
    return build_llm_response


@pytest.fixture
def basic_model():
    """Create basic model without grid"""
    return Model(seed=42)


@pytest.fixture
def grid_model():
    """Create model with MultiGrid"""

    class GridModel(Model):
        def __init__(self):
            super().__init__(seed=42)
            self.grid = MultiGrid(10, 10, torus=False)

    return GridModel()


@pytest.fixture
def basic_agent(basic_model):
    """Create single agent with memory"""
    agents = LLMAgent.create_agents(basic_model, n=1, vision=0, **DEFAULT_AGENT_CONFIG)
    agent = agents.to_list()[0]
    agent.memory = ShortTermMemory(agent=agent, n=5, display=True)
    return agent


@pytest.fixture
def disable_memory(monkeypatch):
    """Helper to disable memory add_to_memory method"""

    def _disable(agent):
        monkeypatch.setattr(agent.memory, "add_to_memory", lambda *args, **kwargs: None)

    return _disable
