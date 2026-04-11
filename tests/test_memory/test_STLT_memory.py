from collections import deque
from unittest.mock import AsyncMock, patch

import pytest

from mesa_llm.memory.memory import MemoryEntry
from mesa_llm.memory.st_lt_memory import STLTMemory


class TestSTLTMemory:
    """Test the Memory class core functionality"""

    def test_memory_initialization(self, mock_agent):
        """Test Memory class initialization with defaults and custom values"""
        memory = STLTMemory(
            agent=mock_agent,
            short_term_capacity=3,
            consolidation_capacity=1,
            llm_model="provider/test_model",
        )

        assert memory.agent == mock_agent
        assert memory.capacity == 3
        assert memory.consolidation_capacity == 1
        assert isinstance(memory.short_term_memory, deque)
        assert memory.long_term_memory == ""
        assert memory.llm.system_prompt is not None

    def test_add_to_memory(self, mock_agent):
        """Test adding memories to short-term memory"""
        memory = STLTMemory(agent=mock_agent, llm_model="provider/test_model")

        # Test basic addition with observation
        memory.add_to_memory("observation", {"step": 1, "content": "Test content"})

        # Test with planning
        memory.add_to_memory("planning", {"plan": "Test plan", "importance": "high"})

        # Test with action
        memory.add_to_memory("action", {"action": "Test action"})

        # Should be empty step_content initially
        assert memory.step_content != {}

    def test_process_step(self, mock_agent):
        """Test process_step functionality"""
        memory = STLTMemory(agent=mock_agent, llm_model="provider/test_model")

        # Add some content
        memory.add_to_memory("observation", {"content": "Test observation"})
        memory.add_to_memory("plan", {"content": "Test plan"})

        # Process the step
        with patch("rich.console.Console"):
            memory.process_step(pre_step=True)
            assert len(memory.short_term_memory) == 1

            # Process post-step
            memory.process_step(pre_step=False)

    def test_memory_consolidation(self, mock_agent, mock_llm, llm_response_factory):
        """Test memory consolidation when capacity is exceeded"""
        mock_llm.generate.return_value = llm_response_factory(
            "Consolidated memory summary"
        )

        memory = STLTMemory(
            agent=mock_agent,
            short_term_capacity=2,
            consolidation_capacity=1,
            llm_model="provider/test_model",
        )

        memory.llm = mock_llm

        # Add memories to trigger consolidation
        with patch("rich.console.Console"):
            for i in range(5):
                memory.add_to_memory("observation", {"content": f"content_{i}"})
                memory.process_step(pre_step=True)
                memory.process_step(pre_step=False)

        # Should have consolidated some memories
        assert (
            len(memory.short_term_memory)
            <= memory.capacity + memory.consolidation_capacity
        )

    def test_format_memories(self, mock_agent):
        """Test formatting of short-term and long-term memory"""
        memory = STLTMemory(agent=mock_agent, llm_model="provider/test_model")

        # Test empty short-term memory
        assert memory.format_short_term() == "No recent memory."

        # Test with entries
        memory.short_term_memory.append(
            MemoryEntry(content={"observation": "Test obs"}, step=1, agent=mock_agent)
        )
        memory.short_term_memory.append(
            MemoryEntry(content={"planning": "Test plan"}, step=2, agent=mock_agent)
        )

        result = memory.format_short_term()
        assert "Step 1:" in result
        assert "Test obs" in result
        assert "Step 2:" in result
        assert "Test plan" in result

        # Test long-term memory formatting
        memory.long_term_memory = "Long-term summary"
        assert memory.format_long_term() == "Long-term summary"

    def test_update_long_term_memory(self, mock_agent, mock_llm, llm_response_factory):
        """Check that after consolidation, long_term_memory holds the actual
        text from the LLM response, not some object."""
        mock_llm.generate.return_value = llm_response_factory(
            "Updated long-term memory"
        )

        memory = STLTMemory(agent=mock_agent, llm_model="provider/test_model")
        memory.llm = mock_llm
        memory.long_term_memory = "Previous memory"

        evicted = [
            MemoryEntry(
                content={"observation": "old content"}, step=0, agent=mock_agent
            )
        ]
        memory._update_long_term_memory(evicted)

        call_args = mock_llm.generate.call_args[0][0]
        assert "old content" in call_args
        assert "Previous memory" in call_args

        # Must be a plain string, not a ModelResponse object
        assert isinstance(memory.long_term_memory, str)
        assert memory.long_term_memory == "Updated long-term memory"

    def test_long_term_memory_stores_string_not_response_object(
        self, mock_agent, mock_llm, llm_response_factory
    ):
        """Make sure long_term_memory is always a plain string.
        Before this fix, it was storing the whole LLM response object instead
        of just the text — which broke any prompt that used the memory.
        """
        mock_llm.generate.return_value = llm_response_factory(
            "This is the summary text"
        )

        memory = STLTMemory(agent=mock_agent, llm_model="provider/test_model")
        memory.llm = mock_llm

        evicted = [MemoryEntry(content={"data": "evicted"}, step=0, agent=mock_agent)]
        memory._update_long_term_memory(evicted)

        assert isinstance(memory.long_term_memory, str), (
            "long_term_memory must be a string, not a ModelResponse object"
        )
        assert memory.long_term_memory == "This is the summary text"

    def test_consolidation_receives_evicted_entries(
        self, mock_agent, mock_llm, llm_response_factory
    ):
        """Regression test for #107: evicted entries must be passed to the
        LLM for summarization, not the remaining short-term memories."""
        mock_llm.generate.return_value = llm_response_factory("Consolidated summary")

        memory = STLTMemory(
            agent=mock_agent,
            short_term_capacity=2,
            consolidation_capacity=2,
            llm_model="provider/test_model",
        )
        memory.llm = mock_llm

        # Fill up: 2 (capacity) + 2 (consolidation) + 1 to trigger
        with patch("rich.console.Console"):
            for i in range(5):
                memory.add_to_memory("observation", {"content": f"step_{i}"})
                memory.process_step(pre_step=True)
                mock_agent.model.steps = i + 1
                memory.process_step(pre_step=False)

        # The LLM should have been called with the evicted entries
        assert mock_llm.generate.called
        prompt = mock_llm.generate.call_args[0][0]
        # The prompt must contain the evicted memories, not just the
        # remaining ones
        assert "consolidate" in prompt.lower() or "removed" in prompt.lower()

    @pytest.mark.asyncio
    async def test_aupdate_long_term_memory(
        self, mock_agent, mock_llm, llm_response_factory
    ):
        """Cover the async consolidation path (_aupdate_long_term_memory)."""
        mock_llm.agenerate = AsyncMock(
            return_value=llm_response_factory("Async consolidated summary")
        )

        memory = STLTMemory(agent=mock_agent, llm_model="provider/test_model")
        memory.llm = mock_llm
        memory.long_term_memory = "Old summary"

        evicted = [
            MemoryEntry(
                content={"observation": "evicted data"}, step=0, agent=mock_agent
            )
        ]
        await memory._aupdate_long_term_memory(evicted)

        mock_llm.agenerate.assert_called_once()
        prompt = mock_llm.agenerate.call_args[0][0]
        assert "evicted data" in prompt
        assert "Old summary" in prompt
        assert isinstance(memory.long_term_memory, str)
        assert memory.long_term_memory == "Async consolidated summary"

    def test_post_step_without_pending_pre_step_is_noop(self, mock_agent):
        """Calling process_step(pre_step=False) when no pre-step entry is
        pending must return early without side effects."""
        memory = STLTMemory(agent=mock_agent, llm_model="provider/test_model")
        # Inject a finalized entry (step is not None)
        memory.short_term_memory.append(
            MemoryEntry(content={"data": "done"}, step=1, agent=mock_agent)
        )
        with patch("rich.console.Console"):
            memory.process_step(pre_step=False)
        # Memory should be unchanged
        assert len(memory.short_term_memory) == 1
        assert memory.short_term_memory[0].step == 1

    def test_overflow_without_consolidation_discards_oldest(self, mock_agent):
        """When consolidation_capacity=0, exceeding capacity should simply
        discard the oldest entry without calling the LLM."""
        memory = STLTMemory(
            agent=mock_agent,
            short_term_capacity=2,
            consolidation_capacity=0,
            llm_model="provider/test_model",
        )
        with patch("rich.console.Console"):
            for i in range(4):
                memory.add_to_memory("observation", {"content": f"step_{i}"})
                memory.process_step(pre_step=True)
                mock_agent.model.steps = i + 1
                memory.process_step(pre_step=False)
        # Capacity is 2, no consolidation, so only 2 entries should remain
        assert len(memory.short_term_memory) <= 2

    def test_consolidation_stops_when_deque_exhausted(
        self, mock_agent, mock_llm, llm_response_factory
    ):
        """The guard `if self.short_term_memory` inside the eviction loop
        prevents popping from an empty deque.  Exercise it by forcing
        consolidation_capacity to exceed the actual deque size."""
        mock_llm.generate.return_value = llm_response_factory("Summary")

        memory = STLTMemory(
            agent=mock_agent,
            short_term_capacity=1,
            consolidation_capacity=5,
            llm_model="provider/test_model",
        )
        memory.llm = mock_llm

        # Seed the deque with one finalized entry and one pending entry
        memory.short_term_memory.append(
            MemoryEntry(content={"data": "old"}, step=0, agent=mock_agent)
        )
        memory.short_term_memory.append(
            MemoryEntry(content={"data": "pending"}, step=None, agent=mock_agent)
        )
        # Force the consolidation condition to trigger despite the small
        # deque (2 entries).  After process_step merges the pending entry
        # the deque still has 2, but the loop tries to pop 5 — the guard
        # stops it after 2.
        memory.capacity = -100

        with patch("rich.console.Console"):
            memory.process_step(pre_step=False)

        # Consolidation was called with the 2 entries that were available
        assert mock_llm.generate.called
        assert len(memory.short_term_memory) == 0

    def test_observation_overwrites_by_default(self, mock_agent):
        """Observations should overwrite by default unless configured additive."""
        memory = STLTMemory(agent=mock_agent, llm_model="provider/test_model")

        obs1 = {"position": (0, 0), "health": 100}
        obs2 = {"position": (1, 1), "health": 90}

        memory.add_to_memory("observation", obs1)
        memory.add_to_memory("observation", obs2)

        assert memory.step_content["observation"] == obs2

    def test_get_prompt_ready_returns_str(self, mock_agent):
        """Test that get_prompt_ready returns a str, not a list (issue #116)."""
        memory = STLTMemory(agent=mock_agent, llm_model="provider/test_model")

        memory.short_term_memory.append(
            MemoryEntry(content={"observation": "Test obs"}, step=1, agent=mock_agent)
        )
        memory.long_term_memory = "Long-term summary"

        result = memory.get_prompt_ready()

        assert isinstance(result, str), (
            f"get_prompt_ready() must return str, got {type(result).__name__}"
        )
        assert "Short term memory:" in result
        assert "Long term memory:" in result
        assert "Test obs" in result
        assert "Long-term summary" in result

    def test_get_prompt_ready_returns_str_when_empty(self, mock_agent):
        """Test that get_prompt_ready returns str even with empty memory."""
        memory = STLTMemory(agent=mock_agent, llm_model="provider/test_model")

        result = memory.get_prompt_ready()

        assert isinstance(result, str), (
            f"get_prompt_ready() must return str, got {type(result).__name__}"
        )
        assert "Short term memory:" in result
        assert "Long term memory:" in result

    def test_get_communication_history_nested_dict(self, mock_agent):
        """
        Regression test: get_communication_history must produce readable text when
        the message entry is a nested dict (produced by speak_to).

        speak_to calls:
            add_to_memory(type="message", content={"message": <text>, "sender": <id>, ...})

        Memory.add_to_memory stores the content dict under step_content["message"], so:
            entry.content = {"message": {"message": <text>, "sender": <id>, "recipients": [...]}}

        The fixed code must render "Agent <id> says: <text>", not a raw dict.
        """
        memory = STLTMemory(agent=mock_agent, llm_model="provider/test_model")

        entry = MemoryEntry(
            content={
                "message": {
                    "message": "regroup at base",
                    "sender": 3,
                    "recipients": [1, 2],
                }
            },
            step=10,
            agent=mock_agent,
        )
        memory.short_term_memory.append(entry)

        history = memory.get_communication_history()
        assert history == "Step 10: Agent 3 says: regroup at base\n\n"

    def test_get_communication_history_skips_non_message_entries(self, mock_agent):
        """Entries without a top-level 'message' key are excluded from communication history."""
        memory = STLTMemory(agent=mock_agent, llm_model="provider/test_model")

        entry_obs = MemoryEntry(
            content={"observation": {"position": (0, 0)}},
            step=1,
            agent=mock_agent,
        )
        entry_msg = MemoryEntry(
            content={
                "message": {"message": "all clear", "sender": 9, "recipients": []}
            },
            step=2,
            agent=mock_agent,
        )
        memory.short_term_memory.extend([entry_obs, entry_msg])

        history = memory.get_communication_history()

        assert "all clear" in history
        assert "position" not in history

    def test_get_communication_history_returns_empty_string_when_no_messages(
        self, mock_agent
    ):
        """Returns an empty string when short-term memory has no message entries."""
        memory = STLTMemory(agent=mock_agent, llm_model="provider/test_model")

        entry = MemoryEntry(
            content={"observation": {"data": "nothing to say"}},
            step=1,
            agent=mock_agent,
        )
        memory.short_term_memory.append(entry)

        assert memory.get_communication_history() == ""
