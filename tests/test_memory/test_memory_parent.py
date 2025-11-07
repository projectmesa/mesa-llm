from typing import TYPE_CHECKING
from unittest.mock import Mock

from memory_utils import mock_agent

from mesa_llm.memory.memory import Memory, MemoryEntry
from mesa_llm.module_llm import ModuleLLM

if TYPE_CHECKING:
    from mesa_llm.llm_agent import LLMAgent


class TestMemoryEntry:
    """Test the MemoryEntry dataclass"""

    def test_memory_entry_creation(self):
        """Test MemoryEntry creation and basic functionality"""

        mock_agent = Mock()
        content = {"observation": "Test content", "metadata": "value"}
        entry = MemoryEntry(content=content, step=1, agent=mock_agent)

        assert entry.content == content
        assert entry.step == 1
        assert entry.agent == mock_agent

    def test_memory_entry_str(self):
        """Test MemoryEntry string representation"""

        mock_agent = Mock()
        content = {"observation": "Test content", "type": "observation"}
        entry = MemoryEntry(content=content, step=1, agent=mock_agent)

        str_repr = str(entry)
        assert "Test content" in str_repr
        assert "observation" in str_repr


class MemoryMock(Memory):
    def __init__(
        self, agent: "LLMAgent", llm_model: str | None = None, display: bool = True
    ):
        super().__init__(agent, llm_model, display)

    def get_prompt_ready(self) -> str:
        return ""

    def get_communication_history(self) -> str:
        return ""

    def process_step(self, pre_step: bool = False):
        """
        Mock implementation of process_step for testing purposes.
        Since this is a test mock, we can use a simple pass implementation.
        """


class TestMemoryParent:
    """Test the Memory class"""

    def test_memory_init(self):
        """Test the init of Memory class"""
        mock_agent = Mock()
        memory = MemoryMock(agent=mock_agent, llm_model="provider/test_model")

        # Parameters init
        assert memory.display
        assert memory.step_content == {}
        assert memory.last_observation == {}

        # llm init with ModuleLLM
        assert isinstance(memory.llm, ModuleLLM)
        assert memory.llm.llm_model == "provider/test_model"

        memory = MemoryMock(agent=mock_agent)
        assert not hasattr(memory, "llm")

    def test_add_to_memory(self):
        memory = MemoryMock(agent=mock_agent)
        # Test basic addition with observation
        memory.add_to_memory("observation", {"step": 1, "content": "Test content"})

        # Test with planning
        memory.add_to_memory("planning", {"plan": "Test plan", "importance": "high"})

        # Test with action
        memory.add_to_memory("action", {"action": "Test action"})

        # Should be non-empty step_content after adding to memory
        assert memory.step_content != {}
        assert "observation" in memory.step_content
