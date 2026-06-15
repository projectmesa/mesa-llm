import json
import logging
import math
from collections import deque
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from mesa_llm.memory.memory import Memory, MemoryEntry, _format_message_entry

if TYPE_CHECKING:
    from mesa_llm.llm_agent import LLMAgent

logger = logging.getLogger(__name__)


class EventGrade(BaseModel):
    grade: int = Field(
        description="Integer score representing the importance of the event"
    )


def normalize_dict_values(scores: dict, min_target: float, max_target: float) -> dict:
    """
    Normalize dictionary values to a target range with min-max scaling.

    This mirrors the min-max helper used in the Generative Agents reference
    retrieval implementation:
    https://github.com/joonspk-research/generative_agents/blob/main/reverie/backend_server/persona/cognitive_modules/retrieve.py
    """
    if not scores:
        return {}

    vals = list(scores.values())
    min_val = min(vals)
    max_val = max(vals)

    range_val = max_val - min_val

    if range_val == 0:
        midpoint = (max_target - min_target) / 2 + min_target
        for key in scores:
            scores[key] = midpoint
    else:
        for key, val in scores.items():
            scores[key] = (val - min_val) * (
                max_target - min_target
            ) / range_val + min_target

    return scores


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """
    Compute the cosine similarity between two equal-length vectors.

    Returns ``0.0`` when either vector is empty, the lengths differ, or either
    vector has zero magnitude, so callers can treat it as a safe "no signal"
    value rather than handling exceptions.
    """
    if not a or not b or len(a) != len(b):
        return 0.0

    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot / (norm_a * norm_b)


class EpisodicMemory(Memory):
    """
    Event-level memory with LLM-based importance scoring and recency-aware retrieval.

    Credit / references:
    - Paper: Generative Agents: Interactive Simulacra of Human Behavior
      https://arxiv.org/abs/2304.03442
    - Reference retrieval code:
      https://github.com/joonspk-research/generative_agents/blob/main/reverie/backend_server/persona/cognitive_modules/retrieve.py

    This implementation is inspired by the paper's retrieval scoring design
    (component-wise min-max normalization, then weighted combination). Retrieval
    combines importance, recency (computed from step age) and—when an
    ``embedding_model`` is configured—relevance, scored as the cosine similarity
    between a focal query and each memory's embedding. Without an
    ``embedding_model`` the relevance term is skipped and retrieval falls back to
    importance + recency only.
    """

    def __init__(
        self,
        agent: "LLMAgent",
        llm_model: str | None = None,
        display: bool = True,
        max_capacity: int = 200,
        considered_entries: int = 30,
        recency_decay: float = 0.995,
        api_base: str | None = None,
        embedding_model: str | None = None,
        recency_weight: float = 1.0,
        importance_weight: float = 1.0,
        relevance_weight: float = 1.0,
    ):
        """
        Initialize the EpisodicMemory.

        Args:
            agent : the agent that owns this memory
            llm_model : the model used to grade event importance
            display : whether to display memory entries in the console
            max_capacity : maximum number of finalized episodic entries to keep
            considered_entries : number of entries to consider during retrieval
            recency_decay : exponential decay factor for recency scoring
            api_base : the API base URL to use for the LLM provider
            embedding_model : optional embedding model in ``"{provider}/{model}"``
                format used to score relevance. When ``None`` the relevance term
                is disabled and retrieval uses importance + recency only.
            recency_weight : weight applied to the normalized recency score.
            importance_weight : weight applied to the normalized importance score.
            relevance_weight : weight applied to the normalized relevance score
                (only used when ``embedding_model`` is set).
        """
        if not llm_model:
            raise ValueError(
                "llm_model must be provided for the usage of episodic memory"
            )

        super().__init__(
            agent,
            llm_model=llm_model,
            api_base=api_base,
            display=display,
        )

        self.max_capacity = max_capacity
        self.memory_entries = deque(maxlen=self.max_capacity)
        self.considered_entries = considered_entries
        self.recency_decay = recency_decay
        self.embedding_model = embedding_model
        self.recency_weight = recency_weight
        self.importance_weight = importance_weight
        self.relevance_weight = relevance_weight

        self.system_prompt = """
            You are an assistant that evaluates memory entries on a scale from 1 to 5, based on their importance to a specific problem or task. Your goal is to assign a score that reflects how much each entry contributes to understanding, solving, or advancing the task. Use the following grading scale:

            5 - Critical: Introduces essential, novel information that significantly impacts problem-solving or decision-making.

            4 - High: Provides important context or clarification that meaningfully improves understanding or direction.

            3 - Moderate: Adds somewhat useful information that may assist but is not essential.

            2 - Low: Offers minimal relevance or slight redundancy; impact is marginal.

            1 - Irrelevant: Contains no useful or applicable information for the current problem.

            Only assess based on the entry's content and its value to the task at hand. Ignore style, grammar, or tone.
            """

    def _extract_importance(self, entry) -> int:
        """
        Safely extracts importance score regardless of data structure.
        Handles:
        - Nested: {"msg": {"importance": 5}}
        - Flat:   {"importance": 5}
        """
        if "importance" in entry.content:
            val = entry.content["importance"]
            return val if isinstance(val, (int, float)) else 1

        for value in entry.content.values():
            if isinstance(value, dict) and "importance" in value:
                val = value["importance"]
                return val if isinstance(val, (int, float)) else 1

        return 1

    def _build_grade_prompt(self, type: str, content: dict) -> str:
        """
        This helper assembles a prompt that includes the event type, event content,
        and up to the five most recent memory entries for contextual grounding.
        It is shared by both synchronous and asynchronous grading methods to
        avoid duplicated prompt-construction logic.
        """
        if len(self.memory_entries) > 0:
            entries = list(self.memory_entries)[-5:]
            previous_entries = "previous memory entries:\n\n".join(
                [str(entry) for entry in entries]
            )
        else:
            previous_entries = "No previous memory entries"

        return f"""
            grade the importance of the following event on a scale from 1 to 5:
            {type}: {content}
            ------------------------------
            {previous_entries}
            """

    def grade_event_importance(self, type: str, content: dict) -> float:
        """
        Grade this event based on the content respect to the previous memory entries
        """
        prompt = self._build_grade_prompt(type, content)
        self.llm.system_prompt = self.system_prompt

        rsp = self.llm.generate(
            prompt=prompt,
            response_format=EventGrade,
        )

        formatted_response = json.loads(rsp.choices[0].message.content)
        return formatted_response["grade"]

    async def agrade_event_importance(self, type: str, content: dict) -> float:
        """
        Asynchronous version of grade_event_importance
        """
        prompt = self._build_grade_prompt(type, content)
        self.llm.system_prompt = self.system_prompt

        rsp = await self.llm.agenerate(
            prompt=prompt,
            response_format=EventGrade,
        )

        formatted_response = json.loads(rsp.choices[0].message.content)
        return formatted_response["grade"]

    def _embed_text(
        self, text: str | list[str]
    ) -> list[float] | list[list[float]] | None:
        """
        Embed one or many texts via the configured embedding model.

        Returns ``None`` (rather than raising) when no embedding model is
        configured or the embedding call fails, so retrieval can gracefully fall
        back to importance + recency scoring.
        """
        if not self.embedding_model:
            return None
        try:
            return self.llm.embed(text, embedding_model=self.embedding_model)
        except Exception:
            logger.warning(
                "Embedding call failed for model %s; skipping relevance scoring.",
                self.embedding_model,
                exc_info=True,
            )
            return None

    def _ensure_entry_embeddings(self, entries: list[MemoryEntry]) -> bool:
        """
        Populate ``entry.embedding`` for any entries missing it, in one batched
        embedding call. Returns ``True`` when every entry has an embedding.
        """
        missing = [entry for entry in entries if entry.embedding is None]
        if missing:
            vectors = self._embed_text([str(entry) for entry in missing])
            if vectors is None:
                return False
            for entry, vector in zip(missing, vectors):
                entry.embedding = vector
        return all(entry.embedding is not None for entry in entries)

    def _compute_relevance_scores(
        self, entries: list[MemoryEntry], query: str | None
    ) -> dict[int, float] | None:
        """
        Score each entry's relevance to the focal query via cosine similarity of
        embeddings, normalized to ``[0, 1]``.

        The focal query defaults to the most recent entry's text when no explicit
        query is given. Returns ``None`` when relevance cannot be computed (no
        embedding model, or an embedding call failed), signalling the caller to
        skip the relevance term.
        """
        query_text = query if query is not None else str(entries[-1])
        query_embedding = self._embed_text(query_text)
        if query_embedding is None:
            return None

        if not self._ensure_entry_embeddings(entries):
            return None

        relevance = {
            i: cosine_similarity(query_embedding, entry.embedding)
            for i, entry in enumerate(entries)
        }
        return normalize_dict_values(relevance, 0, 1)

    def retrieve_top_k_entries(
        self, k: int, query: str | None = None
    ) -> list[MemoryEntry]:
        """
        Retrieve the top-k entries using normalized importance, recency and
        (when an embedding model is configured) relevance.

        Inspired by Generative Agents retrieval scoring: recency, importance and
        relevance are normalized separately to ``[0, 1]`` and combined as a
        weighted sum.

        Args:
            k : number of entries to return.
            query : optional focal query used for relevance scoring. When
                omitted, the most recent entry's text is used as the query.
                Ignored entirely when no ``embedding_model`` is configured.

        Notes:
            Relevance is only included when ``embedding_model`` is set and the
            embedding calls succeed; otherwise retrieval falls back to
            importance + recency.
        """
        if not self.memory_entries:
            return []

        importance_dict = {}
        recency_dict = {}

        entries = list(self.memory_entries)
        current_step = self.agent.model.steps

        for i, entry in enumerate(entries):
            importance_dict[i] = self._extract_importance(entry)

            age = current_step - entry.step
            recency_dict[i] = self.recency_decay**age

        importance_scaled = normalize_dict_values(importance_dict, 0, 1)
        recency_scaled = normalize_dict_values(recency_dict, 0, 1)

        relevance_scaled = None
        if self.embedding_model:
            relevance_scaled = self._compute_relevance_scores(entries, query)

        final_scores = []
        for i in range(len(entries)):
            total_score = (
                self.importance_weight * importance_scaled[i]
                + self.recency_weight * recency_scaled[i]
            )
            if relevance_scaled is not None:
                total_score += self.relevance_weight * relevance_scaled[i]
            final_scores.append((total_score, entries[i]))

        final_scores.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in final_scores[:k]]

    def _finalize_entry(self, type: str, graded_content: dict):
        """Create and persist a finalized episodic entry."""
        new_entry = MemoryEntry(
            agent=self.agent,
            content={type: graded_content},
            step=self.agent.model.steps,
        )
        self.memory_entries.append(new_entry)

    def add_to_memory(self, type: str, content: dict):
        """
        grading logic + adding to memory function call
        """
        graded_content = {
            **content,
            "importance": self.grade_event_importance(type, content),
        }
        self._finalize_entry(type, graded_content)

    async def aadd_to_memory(self, type: str, content: dict):
        """
        Async version of add_to_memory + grading logic
        """
        graded_content = {
            **content,
            "importance": await self.agrade_event_importance(type, content),
        }
        self._finalize_entry(type, graded_content)

    def get_prompt_ready(self, query: str | None = None) -> str:
        """
        Format the top retrieved entries for use in a reasoning prompt.

        Args:
            query : optional focal query forwarded to retrieval for relevance
                scoring. When omitted, the most recent entry is used as the
                query (and relevance is only applied when an ``embedding_model``
                is configured).
        """
        return f"Top {self.considered_entries} memory entries:\n\n" + "\n".join(
            [
                str(entry)
                for entry in self.retrieve_top_k_entries(
                    self.considered_entries, query=query
                )
            ]
        )

    def get_communication_history(self) -> str:
        """
        Get the communication history
        """
        lines = []
        for entry in self.memory_entries:
            if "message" not in entry.content:
                continue
            msgs = entry.content["message"]
            if isinstance(msgs, list):
                for msg in msgs:
                    lines.append(f"Step {entry.step}: {_format_message_entry(msg)}\n\n")
            else:
                lines.append(f"Step {entry.step}: {_format_message_entry(msgs)}\n\n")
        return "\n".join(lines)

    async def aprocess_step(self, pre_step: bool = False):
        """
        Asynchronous version of process_step.

        EpisodicMemory persists entries at add-time and does not use two-phase
        pre/post-step buffering.
        """
        return

    def process_step(self, pre_step: bool = False):
        """
        Process step hook (no-op for episodic memory).

        EpisodicMemory persists entries at add-time and does not use two-phase
        pre/post-step buffering.
        """
        return
