from unittest.mock import patch

import pytest

from mesa_llm.memory.memory import MemoryEntry
from mesa_llm.memory.st_memory import ShortTermMemory


class TestShortTermMemory:
    """Test the ShortTermMemory Functionality"""

    def test_memory_initilaization(self, mock_agent):
        """
        Tests initialization of ShortTermMemory.
        the assertion statements present are used to check if the ShortTermMemory instance is created properly, it checks for the following
            - if agent = mock_agent
            - if memory.n = 5 , i.e., since its default value is 5
            - also checks if initially the length of the memory created is 0
        """
        memory = ShortTermMemory(agent=mock_agent, display=False)

        assert memory.agent == mock_agent
        assert memory.n == 5
        assert len(memory.short_term_memory) == 0

    def test_memory_initialization_rejects_non_positive_capacity(self, mock_agent):
        """
        n must be a positive integer capacity.
        """
        with pytest.raises(ValueError, match="n must be >= 1"):
            ShortTermMemory(agent=mock_agent, n=0, display=False)

        with pytest.raises(ValueError, match="n must be >= 1"):
            ShortTermMemory(agent=mock_agent, n=-1, display=False)

    def test_process_step_creates_current_step_entry(self, mock_agent):
        """
        Function to check if the process_step() creates a temp memory
        """
        memory = ShortTermMemory(agent=mock_agent, display=False)

        memory.step_content = {"observation": "Test observation"}

        # Process pre_step = True
        memory.process_step(pre_step=True)

        assert len(memory.short_term_memory) == 0
        assert memory._current_step_entry is not None
        assert memory._current_step_entry.step is None
        assert memory.step_content == {}

    def test_process_step_full_lifecycle(self, mock_agent):
        """
        Function that checks the entire lifecycle of process_step
        """
        memory = ShortTermMemory(agent=mock_agent, display=False)

        # pre-step simulation
        memory.step_content = {"observation": "before"}
        memory.process_step(pre_step=True)

        # post-step simulation [here we set the agents step count and call process_step()]
        mock_agent.model.steps = 7
        memory.step_content = {"action": "after"}
        memory.process_step(pre_step=False)

        assert len(memory.short_term_memory) == 1
        new_entry = memory.short_term_memory[0]

        # here we assert the memory entries step count it must match the earlier agents step value
        assert new_entry.step == 7
        assert new_entry.content["observation"] == "before"
        assert new_entry.content["action"] == "after"
        assert memory._current_step_entry is None

    def test_display_called_when_enabled(self, mock_agent):
        """
        Tests whether the process step function displays content properly when enabled
        """
        memory = ShortTermMemory(agent=mock_agent, display=True)

        memory.step_content = {"x": 1}

        with patch.object(MemoryEntry, "display") as mock_display:
            memory.process_step(pre_step=True)
            memory.step_content = {"y": 2}
            memory.process_step(pre_step=False)

            mock_display.assert_called_once()

    def test_format_short_term_with_entries(self, mock_agent):
        """
        Tests whether the short term memory was formatted correctly
        """
        memory = ShortTermMemory(agent=mock_agent, display=False)

        # create a fake memory entry
        new_entry = MemoryEntry(
            agent=mock_agent,
            content={"observation": "hello"},
            step=3,
        )

        memory.short_term_memory.append(new_entry)

        output = memory.format_short_term()

        assert "Step 3" in output
        assert "observation" in output
        assert "hello" in output

    def test_short_term_memory_capacity(self, mock_agent):
        """
        This function checks whether the capacity limit of the short term memory is implemented correctly
        """
        memory = ShortTermMemory(agent=mock_agent, n=2, display=False)

        with patch.object(MemoryEntry, "display"):
            for step in range(3):
                # simulate one full step lifecycle
                mock_agent.model.steps = step

                # pre-step
                memory.step_content = {"obs": step}
                memory.process_step(pre_step=True)

                # post-step
                memory.step_content = {"action": step}
                memory.process_step(pre_step=False)

        assert len(memory.short_term_memory) <= 2

        steps_in_memory = [entry.step for entry in memory.short_term_memory]
        assert steps_in_memory[-1] == 2

    def test_pre_step_does_not_evict_when_memory_is_full(self, mock_agent):
        """
        Ensure pre-step staging does not consume deque capacity before finalization.
        """
        memory = ShortTermMemory(agent=mock_agent, n=3, display=False)
        memory.short_term_memory.extend(
            [
                MemoryEntry(agent=mock_agent, content={"obs": 1}, step=1),
                MemoryEntry(agent=mock_agent, content={"obs": 2}, step=2),
                MemoryEntry(agent=mock_agent, content={"obs": 3}, step=3),
            ]
        )

        # pre-step should stage data without evicting finalized entries
        memory.step_content = {"observation": "before_step_4"}
        memory.process_step(pre_step=True)
        assert [entry.step for entry in memory.short_term_memory] == [1, 2, 3]
        assert memory._current_step_entry is not None

        # eviction happens only when final step entry is committed
        mock_agent.model.steps = 4
        memory.step_content = {"action": "after_step_4"}
        memory.process_step(pre_step=False)
        assert [entry.step for entry in memory.short_term_memory] == [2, 3, 4]

    def test_get_communication_history_nested_dict(self, mock_agent):
        """
        Regression test: get_communication_history must produce readable text when
        the message entry is a nested dict (produced by speak_to).

        speak_to calls:
            add_to_memory(type="message", content={"message": <text>, "sender": <id>, ...})

        Memory.add_to_memory stores this under step_content["message"], so:
            entry.content = {"message": {"message": <text>, "sender": <id>, "recipients": [...]}}

        The fixed code must render "Agent <id> says: <text>", not a raw dict.
        """
        memory = ShortTermMemory(agent=mock_agent, display=False)

        entry = MemoryEntry(
            content={
                "message": {"message": "status update", "sender": 12, "recipients": [3]}
            },
            step=7,
            agent=mock_agent,
        )
        memory.short_term_memory.append(entry)

        history = memory.get_communication_history()
        assert history == "Step 7: Agent 12 says: status update\n\n"

    def test_get_communication_history_skips_non_message_entries(self, mock_agent):
        """Entries without a 'message' key are excluded from communication history."""
        memory = ShortTermMemory(agent=mock_agent, display=False)

        entry_obs = MemoryEntry(
            content={"observation": "watching"},
            step=1,
            agent=mock_agent,
        )
        entry_msg = MemoryEntry(
            content={
                "message": {"message": "over here", "sender": 2, "recipients": []}
            },
            step=2,
            agent=mock_agent,
        )
        memory.short_term_memory.extend([entry_obs, entry_msg])

        history = memory.get_communication_history()

        assert "over here" in history
        assert "watching" not in history

    def test_get_communication_history_returns_empty_string_when_no_messages(
        self, mock_agent
    ):
        """Returns empty string when no entries contain a 'message' key."""
        memory = ShortTermMemory(agent=mock_agent, display=False)

        entry = MemoryEntry(
            content={"observation": "nothing happening"},
            step=1,
            agent=mock_agent,
        )
        memory.short_term_memory.append(entry)

        assert memory.get_communication_history() == ""
