from collections import deque
from unittest.mock import patch

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

        memory._update_long_term_memory()

        call_args = mock_llm.generate.call_args[0][0]
        assert "Short term memory:" in call_args
        assert "Long term memory:" in call_args
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

        memory._update_long_term_memory()

        assert isinstance(memory.long_term_memory, str), (
            "long_term_memory must be a string, not a ModelResponse object"
        )
        assert memory.long_term_memory == "This is the summary text"

    def test_observation_tracking(self, mock_agent):
        """Test that observations are properly tracked and only changes stored"""
        memory = STLTMemory(agent=mock_agent, llm_model="provider/test_model")

        # First observation
        obs1 = {"position": (0, 0), "health": 100}
        memory.add_to_memory("observation", obs1)

        # Same observation (should not add much to step_content)
        memory.add_to_memory("observation", obs1)

        # Changed observation
        obs2 = {"position": (1, 1), "health": 90}
        memory.add_to_memory("observation", obs2)

        # Verify last observation is tracked
        assert memory.last_observation == obs2

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

    def test_consolidation_includes_evicted_entry(
        self, mock_agent, mock_llm, llm_response_factory
    ):
        """Test that the entry being evicted from short-term memory is included
        in the consolidation prompt so its content is preserved in long-term
        memory.

        Before the fix, popleft() removed the oldest entry *before*
        consolidation ran, so the evicted entry was permanently lost.
        """
        mock_llm.generate.return_value = llm_response_factory(
            "Consolidated summary"
        )

        memory = STLTMemory(
            agent=mock_agent,
            short_term_capacity=2,
            consolidation_capacity=1,
            llm_model="provider/test_model",
        )
        memory.llm = mock_llm

        with patch("rich.console.Console"):
            # Fill memory to just below consolidation threshold
            for i in range(3):
                memory.add_to_memory(
                    "observation", {"content": f"event_{i}"}
                )
                memory.process_step(pre_step=True)
                memory.process_step(pre_step=False)

            # At this point, capacity(2) + consolidation_capacity(1) = 3
            # We have 3 entries, so no consolidation yet.
            assert not mock_llm.generate.called

            # Add one more entry to trigger consolidation
            memory.add_to_memory(
                "observation", {"content": "event_3"}
            )
            memory.process_step(pre_step=True)
            memory.process_step(pre_step=False)

            # Consolidation should have been triggered
            assert mock_llm.generate.called

            # The consolidation prompt must contain the oldest entry
            # (event_0) that is about to be evicted
            prompt_sent = mock_llm.generate.call_args[0][0]
            assert "event_0" in prompt_sent, (
                "The evicted entry's content must appear in the consolidation "
                "prompt so the LLM can preserve it in long-term memory"
            )
