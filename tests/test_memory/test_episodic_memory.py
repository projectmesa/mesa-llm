import json
from collections import deque
from unittest.mock import AsyncMock, MagicMock

import pytest

from mesa_llm.memory.episodic_memory import (
    EpisodicMemory,
    cos_sim,
    normalize_dict_values,
)
from mesa_llm.memory.memory import MemoryEntry


def test_normalize_dict_floats_logic():
    """
    Function to check whether the values are normalised properly.
        - Hardcoded dict values are used currently to ensure that the normalization logic works.
        - Checks both cases, ie when the range = 0 and when its not 0.
    """
    d = {0: 10, 1: 20, 2: 30}
    norm = normalize_dict_values(d, 0, 1)
    assert norm[0] == 0.0
    assert norm[1] == 0.5
    assert norm[2] == 1.0

    # Checks normalized value when range is 0
    d_tie = {0: 5, 1: 5}
    norm_tie = normalize_dict_values(d_tie, 0, 1)
    assert norm_tie[0] == 0.5
    assert norm_tie[1] == 0.5


def test_normalize_dict_floats_logic_when_empty():
    """
    Function to check whether normalize_dict_values correctly returns an empty dict
    """
    norm = normalize_dict_values({}, 0, 1)
    assert norm == {}


def test_cos_sim_normal_math():
    """Proves normal cosine similarity function works gracefully on raw floats."""

    assert cos_sim([1.0, 0.0], [0.0, 1.0]) == 0.0
    assert abs(cos_sim([1.0, 1.0], [1.0, 1.0]) - 1.0) < 0.0001
    assert abs(cos_sim([1.0, 1.0], [-1.0, -1.0]) - (-1.0)) < 0.0001


def test_cos_sim_zero_division():
    """Tests an edge case wherein if division by 0 occurs it should safely return 0.0"""
    a = [0.0, 0.0, 0.0]
    b = [1.0, 1.0, 1.0]

    assert cos_sim(a, b) == 0.0
    assert cos_sim(a, a) == 0.0


class TestEpisodicMemory:
    """Core functionality test"""

    def test_memory_init(self, episodic_mock_agent):
        """Test EpisodicMemory class initialization with defaults and custom values"""
        memory = EpisodicMemory(
            agent=episodic_mock_agent,
            max_capacity=10,
            considered_entries=5,
            llm_model="provider/test_model",
        )

        assert memory.agent == episodic_mock_agent
        assert memory.max_capacity == 10
        assert memory.considered_entries == 5
        assert isinstance(memory.memory_entries, deque)
        assert memory.memory_entries.maxlen == 10
        assert memory.system_prompt is not None
        """FYI: The above line may not always work; use the one below if needed."""
        # assert isinstance(memory.system_prompt,str), memory.system_prompt.strip() != ""

    def test_add_memory_entry(self, episodic_mock_agent):
        """Test adding memories to Episodic memory"""
        memory = EpisodicMemory(
            agent=episodic_mock_agent, llm_model="provider/test_model"
        )

        mock_response = MagicMock()
        mock_response.choices[0].message.content = json.dumps({"grade": 3})
        memory.llm.generate = MagicMock(return_value=mock_response)

        # Test basic addition with observation
        memory.add_to_memory("observation", {"step": 1, "content": "Test content"})

        # Test with planning
        memory.add_to_memory("planning", {"plan": "Test plan", "importance": "high"})

        # Test with action
        memory.add_to_memory("action", {"action": "Test action"})

        # EpisodicMemory should not rely on transient step buffers.
        assert memory.step_content == {}
        assert len(memory.memory_entries) == 3, (
            "add_to_memory graded the event but never created a MemoryEntry"
        )

    def test_finalize_entry_consistency(self, mock_agent):
        """Minimal tests for the helper function _finalize_entry().
        - This test ensures that:
        - A `MemoryEntry` object is created and stored in episodic memory.
        - The stored entry contains the correct importance score.
        - The entry is stamped with the current agent step.
        """
        memory = EpisodicMemory(agent=mock_agent, llm_model="provider/test_model")
        graded_content = {"data": "test", "importance": 4}

        memory._finalize_entry("observation", graded_content)

        assert memory.memory_entries[0].content["observation"]["importance"] == 4
        assert memory.step_content == {}
        assert isinstance(memory.memory_entries[0], MemoryEntry)
        assert memory.memory_entries[0].step == mock_agent.model.steps

    def test_grade_event_importance(self, episodic_mock_agent, llm_response_factory):
        """Test grading event importance"""
        memory = EpisodicMemory(
            agent=episodic_mock_agent, llm_model="provider/test_model"
        )

        # 1. Set up a specific grade for this test
        memory.llm.generate = MagicMock(
            return_value=llm_response_factory(content=json.dumps({"grade": 5}))
        )

        # 2. Call the method
        grade = memory.grade_event_importance("observation", {"data": "critical info"})

        # 3. Assert the grade is correct
        assert grade == 5

        # 4. Assert the LLM was called correctly
        memory.llm.generate.assert_called_once()

        # Check that the system prompt was set on the llm object
        assert memory.llm.system_prompt == memory.system_prompt

    def test_retrieve_top_k_entries(self, episodic_mock_agent):
        """Test the sorting logic for retrieving entries (importance - recency_penalty)."""
        memory = EpisodicMemory(
            agent=episodic_mock_agent, llm_model="provider/test_model"
        )
        # Set current step
        episodic_mock_agent.model.steps = 100

        # Very important but old
        entry_a = MemoryEntry(
            content={"importance": 5, "id": "A"}, step=98, agent=episodic_mock_agent
        )
        # score = 1 - (100 - 99) = 0
        entry_b = MemoryEntry(
            content={"importance": 1, "id": "B"}, step=99, agent=episodic_mock_agent
        )
        # score = 4 - (100 - 90) = -6
        entry_c = MemoryEntry(
            content={"importance": 4, "id": "C"}, step=90, agent=episodic_mock_agent
        )
        # score = 4 - (100 - 95) = -1
        entry_d = MemoryEntry(
            content={"importance": 4, "id": "D"}, step=95, agent=episodic_mock_agent
        )

        memory.memory_entries.extend([entry_a, entry_b, entry_c, entry_d])

        top_entries = memory.retrieve_top_k_entries(1)

        # The highly important memory should win
        assert len(top_entries) == 1
        assert top_entries[0] == entry_a

    @pytest.mark.asyncio
    async def test_async_add_memory_entry(
        self, episodic_mock_agent, llm_response_factory
    ):
        """
        The aadd_to_memory function assigns an 'importance' value to the content and then calls the add_to_memory function

        The test function does the following
            - mocks the llm to produece a pre-determined grading.
            - then calls the aad_to_memory function
            - checks that entries are persisted directly into memory_entries.
        """
        memory = EpisodicMemory(
            agent=episodic_mock_agent, llm_model="provider/test_model"
        )

        memory.llm.agenerate = AsyncMock(
            return_value=llm_response_factory(content=json.dumps({"grade": 3}))
        )

        # adds content into the memory using the async counter part of add_to_memory function
        await memory.aadd_to_memory("observation", {"content": "Test content"})
        await memory.aadd_to_memory("planning", {"plan": "Test plan"})
        await memory.aadd_to_memory("action", {"action": "Test action"})

        new_entry = memory.memory_entries[0]

        for new_entry in memory.memory_entries:
            event_type = next(iter(new_entry.content.keys()))
            assert new_entry.content[event_type]["importance"] == 3
        assert memory.step_content == {}
        assert len(memory.memory_entries) == 3, (
            "aadd_to_memory graded the event but never created a MemoryEntry"
        )

    def test_build_grade_prompt_no_previous_entries(self, episodic_mock_agent):
        """
        The _build_grade_prompt function inserts 'No previous memory entries this message if there are no entries passed to it.

        This test function checks to see if this fall-back indeed works correctly
            - No memory entries are added before the _build_grade_prompt function call
            - So when the memory is empty we expect to see 'No previous memory entries' in the returned prompt.
        """
        memory = EpisodicMemory(
            agent=episodic_mock_agent, llm_model="provider/test_model"
        )

        prompt = memory._build_grade_prompt("observation", {"data": "test"})

        # checks if the fallback condition actaually works
        assert "No previous memory entries" in prompt
        assert "observation" in prompt

    def test_get_communication_history(self, episodic_mock_agent):
        """
        Return a formatted string of all messages stored in memory.

        This function:
        - Looks through all memory entries
        - Selects only entries that contain a "message" field
        - Formats each message as: "step <step_number>: <message>"
        - Combines them into one single string

        Returns:
            str: A string containing all communication messages
                from memory, separated by new lines.
        """
        memory = EpisodicMemory(
            agent=episodic_mock_agent, llm_model="provider/test_model"
        )

        entry_with_message = MemoryEntry(
            content={"importance": 3, "message": "Hello"},
            step=1,
            agent=episodic_mock_agent,
        )

        entry_without_message = MemoryEntry(
            content={"importance": 2, "data": "No message here"},
            step=2,
            agent=episodic_mock_agent,
        )

        memory.memory_entries.append(entry_with_message)
        memory.memory_entries.append(entry_without_message)

        history = memory.get_communication_history()

        # assertion checks must return true
        assert "Hello" in history
        assert "step 1" in history
        assert (
            "No message here" not in history
        )  # step 2  does not have message field thus it must not be present in the returned string

    def test_retrieve_empty_memory(self, mock_agent):
        """
        Function to verify empty list is returned when retrieval of memory is empty
        """
        memory = EpisodicMemory(agent=mock_agent, llm_model="provider/test_model")

        result = memory.retrieve_top_k_entries(3)

        assert result == []

    def test_extract_importance_flat(self, mock_agent):
        """Function to return importance when stored at top level"""
        memory = EpisodicMemory(agent=mock_agent, llm_model="provider/test_model")
        entry = MemoryEntry(
            content={"importance": 5, "message": "hello"},
            step=1,
            agent=mock_agent,
        )
        result = memory._extract_importance(entry)
        assert result == 5

    def test_extract_importance_nested(self, mock_agent):
        """Should return importance when nested inside another dict"""
        memory = EpisodicMemory(agent=mock_agent, llm_model="provider/test_model")

        entry = MemoryEntry(
            content={"message": {"importance": 4, "text": "nested"}},
            step=1,
            agent=mock_agent,
        )
        result = memory._extract_importance(entry)

        assert result == 4

    def test_extract_importance_missing(self, mock_agent):
        """Should fallback to 1 when importance is absent"""
        memory = EpisodicMemory(agent=mock_agent, llm_model="provider/test_model")

        entry = MemoryEntry(
            content={"message": {"text": "no importance"}},
            step=1,
            agent=mock_agent,
        )
        result = memory._extract_importance(entry)
        assert result == 1

    def test_get_embedding_exceptions(self, episodic_mock_agent, monkeypatch):
        """
        Simulates a failure in the `litellm.embedding`, the embedding function is
        replaced with a mock that raises an exception.

        Expected behavior:
        - `_get_embedding` should catch the exception internally.
        - The method should return `None` instead of propagating the error.
        - This ensures the system does not crash when the embedding service fails.
        """
        memory = EpisodicMemory(
            agent=episodic_mock_agent,
            llm_model="provider/test_model",
            embedding_model="test",
        )

        mock_litellm = MagicMock(side_effect=Exception("API Server Down"))
        monkeypatch.setattr(
            "mesa_llm.memory.episodic_memory.litellm.embedding", mock_litellm
        )

        result = memory._get_embedding("test string")
        assert result is None

    @pytest.mark.asyncio
    async def test_aget_embedding_exceptions(self, episodic_mock_agent, monkeypatch):
        """
        Async version of test_get_embedding
        """
        memory = EpisodicMemory(
            agent=episodic_mock_agent,
            llm_model="provider/test_model",
            embedding_model="test",
        )
        mock_alitellm = AsyncMock(side_effect=Exception("Async API Server Down"))
        monkeypatch.setattr(
            "mesa_llm.memory.episodic_memory.litellm.aembedding", mock_alitellm
        )

        result = await memory._aget_embedding("test string")
        assert result is None

    @pytest.mark.asyncio
    async def test_retrieve_top_k_entries_with_relevance(self, episodic_mock_agent):
        """
        Verify that providing `qry_str` triggers cosine-similarity based relevance scoring
        and overrides the default recency/importance ranking.

        The test manually injects embeddings for memory entries to avoid API calls and
        mocks `_get_embedding` to control the query embedding.
        """

        memory = EpisodicMemory(
            agent=episodic_mock_agent,
            llm_model="provider/test_model",
            embedding_model="test_embed",
        )

        episodic_mock_agent.model.steps = 100

        entry_a = MemoryEntry(
            content={"importance": 3, "data": "apple"},
            step=100,
            agent=episodic_mock_agent,
        )
        entry_b = MemoryEntry(
            content={"importance": 3, "data": "banana"},
            step=100,
            agent=episodic_mock_agent,
        )
        entry_c = MemoryEntry(
            content={"importance": 3, "data": "cherry"},
            step=100,
            agent=episodic_mock_agent,
        )

        memory.memory_entries.extend([entry_a, entry_b, entry_c])

        memory._embeddings.extend(
            [
                [1.0, 0.0, 0.0],  # apple
                [0.0, 1.0, 0.0],  # banana
                [0.0, 0.0, 1.0],  # cherry
            ]
        )

        memory._get_embedding = MagicMock(return_value=[0.0, 1.0, 0.0])

        top_entries = memory.retrieve_top_k_entries(k=1, qry_str="I want a banana")

        assert len(top_entries) == 1
        assert top_entries[0].content["data"] == "banana"
        memory._embeddings[1] = None

        memory._get_embedding = MagicMock(return_value=[1.0, 0.0, 0.0])
        top_entries = memory.retrieve_top_k_entries(k=1, qry_str="I want an apple")

        assert len(top_entries) == 1
        assert top_entries[0].content["data"] == "apple"

    def test_safely_parse_grade_fallbacks(self, episodic_mock_agent):
        """Tests that simulate parsing raw strings and totally broken output"""
        memory = EpisodicMemory(
            agent=episodic_mock_agent, llm_model="provider/test_model"
        )

        assert memory._safely_parse_grade("  4  ") == 4
        assert memory._safely_parse_grade("I don't know the grade") == 3

    def test_finalize_entry_embedding_exception(self, mock_agent):
        """Verifies that `_finalize_entry` handles embedding failures gracefully.
        Simulates a failure in the synchronous embedding generation
        by mocking `_get_embedding` to raise an exception.

        Expected behavior:
        - The memory entry should still be created and stored.
        - The embedding computation failure should not interrupt execution.
        - A `None` placeholder should be stored in `_embeddings` for the failed embedding.
        """
        memory = EpisodicMemory(
            agent=mock_agent, llm_model="provider/test_model", embedding_model="test"
        )

        memory._get_embedding = MagicMock(side_effect=Exception("Sync Embed Broken"))

        content = {"message": "hello", "importance": 3}
        memory._finalize_entry("observation", content)

        assert len(memory.memory_entries) == 1
        assert memory.memory_entries[0].content["observation"]["message"] == "hello"
        assert len(memory._embeddings) == 1
        assert memory._embeddings[0] is None

    @pytest.mark.asyncio
    async def test_afinalize_entry_embedding_exception(self, mock_agent):
        """Verifies that `_afinalize_entry` safely handles failures in asynchronous
        embedding generation.

        Expected behavior:
        - The memory entry should still be added to `memory_entries`.
        - The embedding error should be caught internally.
        - A `None` value should be inserted into `_embeddings` to represent the failed embedding.
        """
        memory = EpisodicMemory(
            agent=mock_agent, llm_model="provider/test_model", embedding_model="test"
        )

        memory._aget_embedding = AsyncMock(side_effect=Exception("Async Embed Broken"))

        content = {"message": "hello", "importance": 3}
        await memory._afinalize_entry("observation", content)

        assert len(memory.memory_entries) == 1
        assert len(memory._embeddings) == 1
        assert memory._embeddings[0] is None

    @pytest.mark.asyncio
    async def test_process_step_no_ops(self, episodic_mock_agent):
        """Ensures process hooks safely return on Episodic"""
        memory = EpisodicMemory(
            agent=episodic_mock_agent, llm_model="provider/test_model"
        )

        assert memory.process_step() is None
        assert await memory.aprocess_step() is None

    def test_get_embedding_success(self, mock_agent, monkeypatch):
        """Verify that `_get_embedding` successfully retrieves and returns an embedding
        vector when the synchronous embedding API call succeeds.

        Expected behavior:
        - `_get_embedding` calls the async embedding API.
        - The embedding vector is extracted from the response.
        """
        memory = EpisodicMemory(
            agent=mock_agent, llm_model="provider/test_model", embedding_model="test"
        )

        mock_response = MagicMock()
        mock_response.data = [{"embedding": [0.5, 0.5]}]
        mock_litellm = MagicMock(return_value=mock_response)

        monkeypatch.setattr(
            "mesa_llm.memory.episodic_memory.litellm.embedding", mock_litellm
        )

        result = memory._get_embedding("test string")
        assert result == [0.5, 0.5]

    @pytest.mark.asyncio
    async def test_aget_embedding_success(self, mock_agent, monkeypatch):
        """
        Verify that `_aget_embedding` successfully retrieves and returns an embedding
        vector when the asynchronous embedding API call succeeds.

        Expected behavior:
        - `_aget_embedding` calls the async embedding API.
        - The embedding vector is extracted from the response.
        """
        memory = EpisodicMemory(
            agent=mock_agent, llm_model="provider/test_model", embedding_model="test"
        )

        mock_response = MagicMock()
        mock_response.data = [{"embedding": [0.8, 0.8]}]
        mock_alitellm = AsyncMock(return_value=mock_response)

        monkeypatch.setattr(
            "mesa_llm.memory.episodic_memory.litellm.aembedding", mock_alitellm
        )

        result = await memory._aget_embedding("test string")
        assert result == [0.8, 0.8]

    def test_get_communication_history_dict(self, mock_agent):
        """
        Verify that `get_communication_history` correctly formats communication
        entries when the stored message content is a dictionary.

        Expected behavior:
        - The message dictionary is unpacked correctly.
        - The output history string follows the format:
        """
        memory = EpisodicMemory(agent=mock_agent, llm_model="provider/test_model")
        entry = MemoryEntry(
            content={
                "message": {"sender": "Alice", "message": "Hello World"},
                "importance": 3,
            },
            step=5,
            agent=mock_agent,
        )
        memory.memory_entries.append(entry)

        history = memory.get_communication_history()
        assert "step 5 - Alice: Hello World" in history
