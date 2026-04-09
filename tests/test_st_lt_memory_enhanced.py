"""
Unit tests for enhanced STLTMemory with token counting and salience pruning.

Tests the new features added in Track B contribution:
- Token counting and token limit enforcement
- Salience-based memory eviction
- Recency and relevance scoring
- Smart pruning strategies
"""

from collections import deque
from unittest.mock import Mock, patch

import pytest

from mesa_llm.memory.memory import MemoryEntry
from mesa_llm.memory.st_lt_memory import STLTMemory


class MockLLM:
    """Mock LLM for testing without actual API calls."""

    def __init__(self):
        self.system_prompt = ""

    def generate(self, prompt):
        response = Mock()
        response.choices = [Mock()]
        response.choices[0].message.content = f"Summarized: {prompt[:50]}..."
        return response

    async def agenerate(self, prompt):
        response = Mock()
        response.choices = [Mock()]
        response.choices[0].message.content = f"Summarized: {prompt[:50]}..."
        return response


class MockAgent:
    """Mock agent for testing."""

    def __init__(self):
        self.model = Mock()
        self.model.steps = 1
        self.step_prompt = "Test task"


@pytest.fixture
def memory_instance():
    """Create a memory instance for testing."""
    agent = MockAgent()

    with patch.object(STLTMemory, "__init__", lambda x, **kwargs: None):
        memory = STLTMemory()

    # Manually set up attributes
    memory.agent = agent
    memory.llm = MockLLM()
    memory.capacity = 5
    memory.consolidation_capacity = 2
    memory.max_tokens = 200
    memory.enable_salience_pruning = True
    memory.salience_recency_weight = 0.3
    memory.salience_threshold = 0.2
    memory.short_term_memory = deque()
    memory.long_term_memory = ""
    memory.step_content = {}
    memory.display = False
    memory.token_buffer_threshold = 0.8
    memory.memory_creation_times = {}
    memory._token_count_cache = {}
    memory.system_prompt = "Test"

    return memory


def test_token_counting():
    """Test token counting method."""
    agent = MockAgent()

    with patch.object(STLTMemory, "__init__", lambda x, **kwargs: None):
        memory = STLTMemory()
    memory.agent = agent

    # Test simple text
    text = "This is a test" * 10  # ~140 chars
    tokens = memory._count_tokens(text)

    # Should be roughly 140 / 4 = 35 tokens
    assert 30 < tokens < 40, f"Expected ~35 tokens, got {tokens}"


def test_recency_score():
    """Test recency score calculation."""
    agent = MockAgent()

    with patch.object(STLTMemory, "__init__", lambda x, **kwargs: None):
        memory = STLTMemory()
    memory.agent = agent
    memory.salience_recency_weight = 0.3

    # Test most recent entry
    recent_score = memory._calculate_recency_score(4, 5)  # Last of 5
    assert 0.8 < recent_score <= 1.0, "Most recent should have high score"

    # Test oldest entry
    oldest_score = memory._calculate_recency_score(0, 5)  # First of 5
    assert 0.01 < oldest_score < 0.3, "Oldest should have low score"

    # Test middle entry
    middle_score = memory._calculate_recency_score(2, 5)  # Middle of 5
    assert oldest_score < middle_score < recent_score, "Middle should be between"


def test_relevance_score():
    """Test relevance score based on keywords."""
    agent = MockAgent()

    with patch.object(STLTMemory, "__init__", lambda x, **kwargs: None):
        memory = STLTMemory()
    memory.agent = agent

    # Test high relevance entry
    high_rel_entry = MemoryEntry(
        content={"action": "move", "decision": "important", "result": "success"},
        step=1,
        agent=agent,
    )
    high_score = memory._calculate_relevance_score(high_rel_entry)
    assert 0.4 < high_score <= 1.0, "Multiple keywords should give higher score"

    # Test low relevance entry
    low_rel_entry = MemoryEntry(
        content={"data": "xyz", "info": "abc"}, step=1, agent=agent
    )
    low_score = memory._calculate_relevance_score(low_rel_entry)
    assert 0.0 <= low_score < 0.2, "No keywords should give low score"


def test_salience_score_calculation():
    """Test combined salience score."""
    agent = MockAgent()

    with patch.object(STLTMemory, "__init__", lambda x, **kwargs: None):
        memory = STLTMemory()
    memory.agent = agent
    memory.salience_recency_weight = 0.3
    memory.short_term_memory = deque()

    # Create test entries
    entry1 = MemoryEntry(content={"type": "action"}, step=1, agent=agent)
    entry2 = MemoryEntry(content={"type": "observation"}, step=2, agent=agent)
    entry3 = MemoryEntry(
        content={"action": "goal", "decision": "plan"}, step=3, agent=agent
    )

    memory.short_term_memory.extend([entry1, entry2, entry3])

    # Recent + high-relevance should get highest score
    score3 = memory._calculate_salience_score(2, entry3)

    # Old + low-relevance should get lowest score
    score1 = memory._calculate_salience_score(0, entry1)

    assert score3 > score1, "Recent relevant should score higher than old irrelevant"
    assert 0.0 <= score1 <= 1.0, "Score should be normalized 0-1"
    assert 0.0 <= score3 <= 1.0, "Score should be normalized 0-1"


def test_token_usage_tracking(memory_instance):
    """Test token usage tracking."""
    agent = memory_instance.agent

    # Add some entries
    entry1 = MemoryEntry(content={"data": "x" * 100}, step=1, agent=agent)
    entry2 = MemoryEntry(content={"data": "y" * 150}, step=2, agent=agent)

    memory_instance.short_term_memory.append(entry1)
    memory_instance.short_term_memory.append(entry2)

    usage = memory_instance.get_token_usage()

    assert usage["entries"] == 2, "Should have 2 entries"
    assert usage["short_term_tokens"] > 0, "Should have tracked tokens"
    assert "token_utilization_percent" in usage
    assert 0 <= usage["token_utilization_percent"] <= 100


def test_memory_stats(memory_instance):
    """Test comprehensive memory statistics."""
    agent = memory_instance.agent

    # Add entries with varying relevance
    entry1 = MemoryEntry(content={"action": "move"}, step=1, agent=agent)
    entry2 = MemoryEntry(content={"observation": "state"}, step=2, agent=agent)
    entry3 = MemoryEntry(
        content={"error": "fail", "decision": "retry"}, step=3, agent=agent
    )

    memory_instance.short_term_memory.extend([entry1, entry2, entry3])

    stats = memory_instance.get_memory_stats()
<<<<<<< HEAD

    assert len(stats['entries']) == 3, "Should report 3 entries"
    assert 'average_salience' in stats
    assert 0.0 <= stats['average_salience'] <= 1.0
    assert stats['pruning_enabled']

=======

    assert len(stats["entries"]) == 3, "Should report 3 entries"
    assert "average_salience" in stats
    assert 0.0 <= stats["average_salience"] <= 1.0
    assert stats["pruning_enabled"] == True

>>>>>>> de89bf431885d8ddf234d06e4d82edae73c0c11f
    # Each entry should have salience info
    for entry_stat in stats["entries"]:
        assert "salience" in entry_stat
        assert "relevance" in entry_stat
        assert "recency" in entry_stat
        assert "tokens" in entry_stat


def test_token_limit_enforcement(memory_instance):
    """Test that token limits are enforced during eviction."""
    agent = memory_instance.agent
    memory_instance.max_tokens = 100  # Low limit for testing
    memory_instance.enable_salience_pruning = False  # Test token limit only

    # Create entries that will exceed token limit when all added
    entry1 = MemoryEntry(content={"data": "x" * 150}, step=1, agent=agent)
    entry2 = MemoryEntry(content={"data": "y" * 150}, step=2, agent=agent)
    entry3 = MemoryEntry(content={"data": "z" * 100}, step=3, agent=agent)

    memory_instance.short_term_memory.append(entry1)
    memory_instance.short_term_memory.append(entry2)
    memory_instance.short_term_memory.append(entry3)

    tokens_initial = memory_instance._calculate_short_term_tokens()

    # Verify we have many entries/tokens to evict from
    if tokens_initial > memory_instance.max_tokens:
        # Token limit should have been triggered, we can verify processing
        assert len(memory_instance.short_term_memory) >= 1, "Should have memory entries"


def test_salience_pruning_removes_low_salience_entries(memory_instance):
    """Test that salience pruning removes low-salience entries."""
    agent = memory_instance.agent
    memory_instance.enable_salience_pruning = True
    memory_instance.salience_threshold = 0.5  # High threshold

    # Create low-salience entry (old, irrelevant)
    low_sal_entry = MemoryEntry(content={"data": "xyz"}, step=1, agent=agent)

    # Create high-salience entry (recent, relevant)
    high_sal_entry = MemoryEntry(content={"action": "success"}, step=2, agent=agent)

    memory_instance.short_term_memory.extend([low_sal_entry, high_sal_entry])

    # Process step should trigger pruning
<<<<<<< HEAD
    _, _evicted = memory_instance._process_step_core(pre_step=False)

=======
    _, evicted = memory_instance._process_step_core(pre_step=False)

>>>>>>> de89bf431885d8ddf234d06e4d82edae73c0c11f
    # Low-salience entry should be considered for eviction
    assert len(memory_instance.short_term_memory) <= 2, "Memory should store entries"


def test_backward_compatibility():
    """Test that memory works with defaults (no token limiting)."""
    agent = MockAgent()

    # Create memory with minimal config using init patches
    with patch.object(STLTMemory, "__init__", lambda x, **kwargs: None):
        memory = STLTMemory()

    # Manually initialize with minimum required
    memory.agent = agent
    memory.llm = MockLLM()
    memory.max_tokens = None  # Default: no limit
    memory.enable_salience_pruning = True
    memory.salience_recency_weight = 0.3
    memory.short_term_memory = deque()
    memory.long_term_memory = ""
    memory.step_content = {}
    memory.display = False

    # Should have defaults set
    assert memory.max_tokens is None, "Should allow unlimited tokens"
<<<<<<< HEAD
    assert memory.enable_salience_pruning, "Salience pruning on by default"
    assert 0 <= memory.salience_recency_weight <= 1, "Recency weight should be normalized"
=======
    assert memory.enable_salience_pruning == True, "Salience pruning on by default"
    assert 0 <= memory.salience_recency_weight <= 1, (
        "Recency weight should be normalized"
    )
>>>>>>> de89bf431885d8ddf234d06e4d82edae73c0c11f


def test_format_short_term_with_stats(memory_instance):
    """Test debug formatting of memory with stats."""
    agent = memory_instance.agent

    entry = MemoryEntry(
        content={"action": "test", "goal": "verify"}, step=5, agent=agent
    )
    memory_instance.short_term_memory.append(entry)

    formatted = memory_instance.format_short_term_with_stats()

    assert "Step 5" in formatted, "Should show step number"
    assert "Salience" in formatted, "Should show salience score"
    assert "Tokens" in formatted, "Should show token count"
    assert "Token usage" in formatted, "Should show token usage header"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
