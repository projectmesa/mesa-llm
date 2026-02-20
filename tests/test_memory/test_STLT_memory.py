from collections import deque
from unittest.mock import Mock, patch

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

    def test_memory_consolidation(self, mock_agent, mock_llm):
        """Test memory consolidation when capacity is exceeded"""
        mock_llm.generate.return_value = "Consolidated memory summary"

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

    def test_update_long_term_memory(self, mock_agent, mock_llm):
        """Test long-term memory update process"""
        mock_llm.generate.return_value = "Updated long-term memory"

        memory = STLTMemory(agent=mock_agent, llm_model="provider/test_model")
        # Replace the real LLM with our mock
        memory.llm = mock_llm
        memory.long_term_memory = "Previous memory"

        memory._update_long_term_memory()

        # Verify LLM was called with correct prompt structure
        call_args = mock_llm.generate.call_args[0][0]
        assert "Short term memory:" in call_args
        assert "Long term memory:" in call_args
        assert "Previous memory" in call_args

        assert memory.long_term_memory == "Updated long-term memory"

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

    def test_batch_size_fix(self, mock_agent):
        """
        A minimal test to check whether stlt memory pops out the consolidation batches properly.
        """
        new_entry = STLTMemory(
            agent=mock_agent,
            short_term_capacity=2,
            consolidation_capacity=3,
            llm_model="provider/test_model",
        )
        new_entry.llm.generate = Mock(return_value="summary")
        new_entry.agent.model.steps = 0

        # Populate with 5 sample values
        for i in range(5):
            new_entry.short_term_memory.append(
                MemoryEntry(agent=new_entry.agent, content={"v": i}, step=i)
            )

        # Add the 6th item via process_step to trigger the logic
        # New state: size will be 6. (6 > 5) triggers consolidation.

        new_entry.step_content = {"v": 5}

        new_entry.process_step(pre_step=True)
        new_entry.agent.model.steps = 5
        new_entry.process_step(pre_step=False)

        # We started with 6 items total
        # If the fix works: 6 - 3 (Batch) = 3 items should remain.
        # If the bug exists: 6 - 1 (Old behavior) = 5 items will remain.

        actual_count = len(new_entry.short_term_memory)
        print(actual_count)
        assert actual_count == 3, (
            f"FAILED: Expected 3 items left, but found {actual_count}."
            f"The fix loop did not remove the full batch of 3!"
        )
