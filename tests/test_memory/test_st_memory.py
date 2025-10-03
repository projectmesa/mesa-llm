# tests/test_memory/test_st_memory.py

from collections import deque

from mesa_llm.memory.memory import MemoryEntry
from mesa_llm.memory.st_memory import ShortTermMemory


class TestShortTermMemory:
    """Tests for the ShortTermMemory class."""

    def test_initialization(self, mock_agent):
        """Test that the memory initializes correctly."""
        memory = ShortTermMemory(agent=mock_agent, n=7, display=False)
        assert memory.n == 7
        assert isinstance(memory.short_term_memory, deque)
        assert (
            memory.short_term_memory.maxlen is None
        )  # Deque for STMemory is not bounded

    def test_process_step_logic(self, mock_agent):
        """Test the two-part process_step logic for pre- and post-step."""
        mock_agent.model.steps = 1
        memory = ShortTermMemory(agent=mock_agent, n=5, display=False)

        # 1. Simulate pre_step: content is added with step=None
        memory.step_content = {"observation": "seeing a cat"}
        memory.process_step(pre_step=True)

        assert len(memory.short_term_memory) == 1
        first_entry = memory.short_term_memory[0]
        assert first_entry.step is None
        assert first_entry.content == {"observation": "seeing a cat"}
        assert memory.step_content == {}  # step_content should be cleared

        # 2. Simulate post_step: the previous entry is updated with the real step number
        memory.step_content = {"action": "pet the cat"}
        memory.process_step(pre_step=False)

        assert len(memory.short_term_memory) == 1
        updated_entry = memory.short_term_memory[0]
        assert updated_entry.step == 1  # Step number is now set
        assert updated_entry.content == {
            "observation": "seeing a cat",
            "action": "pet the cat",
        }
        assert memory.step_content == {}  # step_content should be cleared again

    def test_format_short_term_empty(self, mock_agent):
        """Test that formatting an empty memory returns the correct string."""
        memory = ShortTermMemory(agent=mock_agent)
        assert memory.format_short_term() == "No recent memory."

    def test_get_communication_history(self, mock_agent):
        """Test that communication history is correctly extracted."""
        memory = ShortTermMemory(agent=mock_agent)

        # Manually add some entries
        msg_entry_content = {"message": "Hello there!"}
        action_entry_content = {"action": "move"}

        memory.short_term_memory.append(
            MemoryEntry(content=msg_entry_content, step=1, agent=mock_agent)
        )
        memory.short_term_memory.append(
            MemoryEntry(content=action_entry_content, step=1, agent=mock_agent)
        )

        history = memory.get_communication_history()
        assert "step 1: Hello there!" in history
        assert "action" not in history
