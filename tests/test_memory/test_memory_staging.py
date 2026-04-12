"""Tests for memory staging area additive event handling (issue #137).

Verifies that concurrent events of the same type within a single step
are accumulated rather than overwritten.
"""

from unittest.mock import AsyncMock, Mock

import pytest

from mesa_llm.memory.episodic_memory import EpisodicMemory
from mesa_llm.memory.lt_memory import LongTermMemory
from mesa_llm.memory.memory import Memory, MemoryEntry
from mesa_llm.memory.st_lt_memory import STLTMemory
from mesa_llm.memory.st_memory import ShortTermMemory


# ---------------------------------------------------------------------------
# Concrete Memory subclass for unit-testing the base class behaviour
# ---------------------------------------------------------------------------
class ConcreteMemory(Memory):
    def get_prompt_ready(self) -> str:
        return ""

    def get_communication_history(self) -> str:
        return ""

    def process_step(self, pre_step: bool = False):
        pass


@pytest.fixture
def agent():
    a = Mock()
    a.__class__.__name__ = "TestAgent"
    a.unique_id = 1
    a.model = Mock()
    a.model.steps = 1
    a.step_prompt = None
    return a


def _make_buffered_memory(kind, agent, llm_response_factory):
    if kind == "short_term":
        return ShortTermMemory(agent=agent, n=5, display=False)

    if kind == "stlt":
        return STLTMemory(
            agent=agent, llm_model="gemini/gemini-2.0-flash", display=False
        )

    if kind == "long_term":
        mem = LongTermMemory(
            agent=agent, llm_model="gemini/gemini-2.0-flash", display=False
        )
        response = llm_response_factory("summary")
        mem.llm.generate = Mock(return_value=response)
        mem.llm.agenerate = AsyncMock(return_value=response)
        return mem

    raise ValueError(f"Unknown memory kind: {kind}")


def _get_finalized_entry(memory):
    if hasattr(memory, "short_term_memory"):
        return list(memory.short_term_memory)[-1]
    return memory.buffer


ADDITIVE_EVENT_CASES = [
    (
        "message",
        {"message": "before", "sender": "A1", "recipients": ["A3"]},
        {"message": "after", "sender": "A2", "recipients": ["A3"]},
    ),
    (
        "action",
        {"tool_calls": [{"name": "wait", "response": "ok"}]},
        {"tool_calls": [{"name": "move", "response": "done"}]},
    ),
]


# ===================================================================
# Tests for Memory.add_to_memory additive behaviour
# ===================================================================
class TestAdditiveMemory:
    """Verify that additive event types accumulate instead of overwriting."""

    def test_single_message_stored_as_list(self, agent):
        mem = ConcreteMemory(agent=agent)
        mem.add_to_memory("message", {"sender": "A1", "msg": "Hello"})
        assert isinstance(mem.step_content["message"], list)
        assert len(mem.step_content["message"]) == 1
        assert mem.step_content["message"][0]["sender"] == "A1"

    def test_multiple_messages_all_preserved(self, agent):
        """Core regression test for issue #137."""
        mem = ConcreteMemory(agent=agent)
        mem.add_to_memory("message", {"sender": "A1", "msg": "Attack!"})
        mem.add_to_memory("message", {"sender": "A2", "msg": "Defend!"})
        mem.add_to_memory("message", {"sender": "A3", "msg": "Retreat!"})

        msgs = mem.step_content["message"]
        assert isinstance(msgs, list)
        assert len(msgs) == 3
        senders = {m["sender"] for m in msgs}
        assert senders == {"A1", "A2", "A3"}

    def test_multiple_actions_all_preserved(self, agent):
        mem = ConcreteMemory(agent=agent)
        mem.add_to_memory("action", {"name": "move", "response": "ok"})
        mem.add_to_memory("action", {"name": "speak", "response": "done"})

        actions = mem.step_content["action"]
        assert isinstance(actions, list)
        assert len(actions) == 2

    def test_observation_still_overwrites(self, agent):
        """Types outside additive_event_types should keep overwrite semantics."""
        mem = ConcreteMemory(agent=agent)
        mem.add_to_memory("observation", {"pos": (0, 0)})
        mem.add_to_memory("observation", {"pos": (1, 1)})

        obs = mem.step_content["observation"]
        assert isinstance(obs, dict)
        assert obs == {"pos": (1, 1)}

    def test_non_additive_types_overwrite(self, agent):
        """Types not in additive_event_types should still overwrite."""
        mem = ConcreteMemory(agent=agent)
        mem.add_to_memory("Plan", {"content": "plan A"})
        mem.add_to_memory("Plan", {"content": "plan B"})

        assert mem.step_content["Plan"] == {"content": "plan B"}

    def test_custom_additive_event_types_can_include_observation(self, agent):
        mem = ConcreteMemory(agent=agent, additive_event_types=["observation"])
        first = {"pos": (0, 0)}
        second = {"pos": (1, 1)}

        mem.add_to_memory("observation", first)
        mem.add_to_memory("observation", second)

        assert mem.step_content["observation"] == [first, second]

    def test_mixed_types_in_same_step(self, agent):
        """Different types coexist correctly in step_content."""
        mem = ConcreteMemory(agent=agent)
        mem.add_to_memory("observation", {"pos": (0, 0)})
        mem.add_to_memory("message", {"sender": "A1", "msg": "hi"})
        mem.add_to_memory("message", {"sender": "A2", "msg": "hey"})
        mem.add_to_memory("Plan", {"content": "do something"})

        assert isinstance(mem.step_content["observation"], dict)
        assert isinstance(mem.step_content["message"], list)
        assert len(mem.step_content["message"]) == 2
        assert isinstance(mem.step_content["Plan"], dict)

    @pytest.mark.parametrize("memory_kind", ["short_term", "stlt", "long_term"])
    @pytest.mark.parametrize(("event_type", "before", "after"), ADDITIVE_EVENT_CASES)
    def test_buffered_memories_preserve_additive_events_across_step_boundary(
        self, agent, llm_response_factory, memory_kind, event_type, before, after
    ):
        """Additive events from both halves of a step must survive finalization."""
        mem = _make_buffered_memory(memory_kind, agent, llm_response_factory)

        mem.add_to_memory(event_type, before)
        mem.process_step(pre_step=True)

        agent.model.steps = 2
        mem.add_to_memory(event_type, after)
        mem.process_step(pre_step=False)

        finalized_entry = _get_finalized_entry(mem)
        assert finalized_entry.content[event_type] == [before, after]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("memory_kind", ["short_term", "stlt", "long_term"])
    async def test_async_buffered_memories_preserve_messages_across_step_boundary(
        self, agent, llm_response_factory, memory_kind
    ):
        """Async finalization should preserve pre-step and post-step messages."""
        before = {"message": "before", "sender": "A1", "recipients": ["A3"]}
        after = {"message": "after", "sender": "A2", "recipients": ["A3"]}
        mem = _make_buffered_memory(memory_kind, agent, llm_response_factory)

        await mem.aadd_to_memory("message", before)
        await mem.aprocess_step(pre_step=True)

        agent.model.steps = 2
        await mem.aadd_to_memory("message", after)
        await mem.aprocess_step(pre_step=False)

        finalized_entry = _get_finalized_entry(mem)
        assert finalized_entry.content["message"] == [before, after]


# ===================================================================
# Tests for MemoryEntry.__str__ with list values
# ===================================================================
class TestMemoryEntryDisplay:
    """Ensure MemoryEntry formats list-valued content correctly."""

    def test_str_with_list_content(self, agent):
        content = {
            "message": [
                {"sender": "A1", "msg": "Attack!"},
                {"sender": "A2", "msg": "Defend!"},
            ]
        }
        entry = MemoryEntry(content=content, step=1, agent=agent)
        result = str(entry)
        assert "A1" in result
        assert "A2" in result
        assert "Attack!" in result
        assert "Defend!" in result


# ===================================================================
# Tests for ShortTermMemory.get_communication_history with lists
# ===================================================================
class TestShortTermCommunicationHistory:
    """Ensure communication history handles list-valued messages."""

    def test_communication_history_with_multiple_messages(self, agent):
        mem = ShortTermMemory(agent=agent, n=5, display=False)

        # Simulate a step with multiple messages
        mem.add_to_memory(
            "message", {"message": "Attack!", "sender": "A1", "recipients": ["A3"]}
        )
        mem.add_to_memory(
            "message", {"message": "Defend!", "sender": "A2", "recipients": ["A3"]}
        )

        # Process pre-step then post-step to finalize
        mem.process_step(pre_step=True)
        agent.model.steps = 2
        mem.process_step(pre_step=False)

        history = mem.get_communication_history()
        assert "Attack!" in history
        assert "Defend!" in history

    def test_communication_history_with_no_messages(self, agent):
        mem = ShortTermMemory(agent=agent, n=5, display=False)
        assert mem.get_communication_history() == ""

    def test_communication_history_skips_non_message_entries(self, agent):
        """Entries without a 'message' key must be skipped."""
        mem = ShortTermMemory(agent=agent, n=5, display=False)
        mem.short_term_memory.append(
            MemoryEntry(agent=agent, content={"observation": {"pos": (1, 1)}}, step=1)
        )
        mem.short_term_memory.append(
            MemoryEntry(
                agent=agent,
                content={"message": {"sender": "A1", "msg": "hi"}},
                step=2,
            )
        )
        history = mem.get_communication_history()
        assert "hi" in history
        assert "observation" not in history

    def test_communication_history_with_legacy_single_message(self, agent):
        """Cover the non-list branch for backward compat with legacy data."""
        mem = ShortTermMemory(agent=agent, n=5, display=False)
        # Directly inject a legacy single-dict message entry
        entry = MemoryEntry(
            agent=agent,
            content={"message": {"sender": "A1", "msg": "legacy"}},
            step=1,
        )
        mem.short_term_memory.append(entry)
        history = mem.get_communication_history()
        assert "legacy" in history


# ===================================================================
# Tests for STLTMemory.get_communication_history with lists
# ===================================================================
class TestSTLTCommunicationHistory:
    """Ensure STLTMemory communication history handles list-valued messages."""

    def test_stlt_communication_history_with_multiple_messages(self, agent):
        mem = STLTMemory(
            agent=agent, llm_model="gemini/gemini-2.0-flash", display=False
        )
        # Inject entries with list-valued messages
        entry = MemoryEntry(
            agent=agent,
            content={
                "message": [
                    {"message": "Hello!", "sender": "A1", "recipients": ["A3"]},
                    {"message": "World!", "sender": "A2", "recipients": ["A3"]},
                ]
            },
            step=1,
        )
        mem.short_term_memory.append(entry)
        history = mem.get_communication_history()
        assert "Hello!" in history
        assert "World!" in history

    def test_stlt_communication_history_with_legacy_single_message(self, agent):
        """Cover the non-list branch."""
        mem = STLTMemory(
            agent=agent, llm_model="gemini/gemini-2.0-flash", display=False
        )
        entry = MemoryEntry(
            agent=agent,
            content={"message": {"sender": "A1", "msg": "legacy"}},
            step=1,
        )
        mem.short_term_memory.append(entry)
        history = mem.get_communication_history()
        assert "legacy" in history

    def test_stlt_communication_history_skips_non_message_entries(self, agent):
        """Entries without a 'message' key must be skipped."""
        mem = STLTMemory(
            agent=agent, llm_model="gemini/gemini-2.0-flash", display=False
        )
        mem.short_term_memory.append(
            MemoryEntry(agent=agent, content={"observation": {"pos": (1, 1)}}, step=1)
        )
        mem.short_term_memory.append(
            MemoryEntry(
                agent=agent,
                content={"message": {"sender": "A1", "msg": "hi"}},
                step=2,
            )
        )
        history = mem.get_communication_history()
        assert "hi" in history
        assert "observation" not in history

    def test_stlt_communication_history_no_messages(self, agent):
        mem = STLTMemory(
            agent=agent, llm_model="gemini/gemini-2.0-flash", display=False
        )
        assert mem.get_communication_history() == ""


# ===================================================================
# Tests for EpisodicMemory.get_communication_history with lists
# ===================================================================
class TestEpisodicCommunicationHistory:
    """Ensure EpisodicMemory communication history handles list-valued messages."""

    def test_episodic_communication_history_with_list_messages(self, agent):
        """Cover the list branch in EpisodicMemory.get_communication_history."""
        mem = EpisodicMemory(
            agent=agent, llm_model="gemini/gemini-2.0-flash", display=False
        )
        entry = MemoryEntry(
            agent=agent,
            content={
                "message": [
                    {"sender": "A1", "msg": "Attack!"},
                    {"sender": "A2", "msg": "Defend!"},
                ]
            },
            step=1,
        )
        mem.memory_entries.append(entry)
        history = mem.get_communication_history()
        assert "Attack!" in history
        assert "Defend!" in history

    def test_episodic_communication_history_with_scalar_message(self, agent):
        """Cover the non-list (scalar) branch in EpisodicMemory.get_communication_history."""
        mem = EpisodicMemory(
            agent=agent, llm_model="gemini/gemini-2.0-flash", display=False
        )
        entry = MemoryEntry(
            agent=agent,
            content={"message": {"sender": "A1", "msg": "solo"}},
            step=1,
        )
        mem.memory_entries.append(entry)
        history = mem.get_communication_history()
        assert "solo" in history

    def test_episodic_communication_history_skips_non_message_entries(self, agent):
        """Entries without a 'message' key must be skipped."""
        mem = EpisodicMemory(
            agent=agent, llm_model="gemini/gemini-2.0-flash", display=False
        )
        mem.memory_entries.append(
            MemoryEntry(agent=agent, content={"observation": {"pos": (1, 1)}}, step=1)
        )
        mem.memory_entries.append(
            MemoryEntry(
                agent=agent,
                content={"message": {"sender": "A1", "msg": "hi"}},
                step=2,
            )
        )
        history = mem.get_communication_history()
        assert "hi" in history
        assert "observation" not in history

    def test_episodic_communication_history_empty(self, agent):
        """Empty memory should return empty string."""
        mem = EpisodicMemory(
            agent=agent, llm_model="gemini/gemini-2.0-flash", display=False
        )
        assert mem.get_communication_history() == ""


# ===================================================================
# Tests for MemoryEntry.__str__ edge cases
# ===================================================================
class TestMemoryEntryEdgeCases:
    """Cover edge cases in MemoryEntry formatting."""

    def test_str_with_list_of_non_dict_items(self, agent):
        """Cover the branch where list items are not dicts."""
        content = {"action": ["moved north", "picked up item"]}
        entry = MemoryEntry(content=content, step=1, agent=agent)
        result = str(entry)
        assert "moved north" in result
        assert "picked up item" in result

    def test_str_with_scalar_value(self, agent):
        """Cover the else branch where value is neither list nor dict."""
        content = {"status": "idle"}
        entry = MemoryEntry(content=content, step=1, agent=agent)
        result = str(entry)
        assert "idle" in result


# ===================================================================
# Tests for legacy migration path in add_to_memory
# ===================================================================
class TestLegacyMigration:
    """Cover the migration path from single-dict to list in add_to_memory."""

    def test_legacy_single_dict_migrated_to_list(self, agent):
        """If step_content already has a plain dict for an additive type,
        adding another entry should migrate it to a list."""
        mem = ConcreteMemory(agent=agent)
        # Directly inject a legacy single-dict value
        mem.step_content["message"] = {"sender": "A1", "msg": "old"}
        mem.add_to_memory("message", {"sender": "A2", "msg": "new"})

        msgs = mem.step_content["message"]
        assert isinstance(msgs, list)
        assert len(msgs) == 2
        assert msgs[0]["sender"] == "A1"
        assert msgs[1]["sender"] == "A2"
