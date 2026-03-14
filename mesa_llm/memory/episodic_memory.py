import json
from collections import deque
from typing import TYPE_CHECKING

import litellm
from numpy import array, dot
from numpy.linalg import norm
from pydantic import BaseModel

from mesa_llm.memory.memory import Memory, MemoryEntry

if TYPE_CHECKING:
    from mesa_llm.llm_agent import LLMAgent


class EventGrade(BaseModel):
    grade: int


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


def cos_sim(a: list[float], b: list[float]) -> float:
    """Implement a Cosine Similarity metrics to grade the two input arrays.

    mirrors the cosine similarity helper used in :
    https://github.com/joonspk-research/generative_agents/blob/main/reverie/backend_server/persona/cognitive_modules/retrieve.py
    """
    a_arr, b_arr = array(a), array(b)
    denom = norm(a_arr) * norm(b_arr)
    if denom == 0:
        return 0.0
    return float(dot(a_arr, b_arr) / denom)


class EpisodicMemory(Memory):
    """
    Event-level memory with LLM-based importance scoring and recency-aware retrieval.

    Credit / references:
    - Paper: Generative Agents: Interactive Simulacra of Human Behavior
      https://arxiv.org/abs/2304.03442
    - Reference retrieval code:
      https://github.com/joonspk-research/generative_agents/blob/main/reverie/backend_server/persona/cognitive_modules/retrieve.py

    This implementation is inspired by the paper's retrieval scoring design
    (component-wise min-max normalization, then weighted combination). It is
    not a strict copy of the original code: relevance scoring via embeddings is
    not implemented yet, and recency is computed from step age.
    """

    def __init__(
        self,
        agent: "LLMAgent",
        llm_model: str | None = None,
        display: bool = True,
        max_capacity: int = 200,
        considered_entries: int = 30,
        recency_decay: float = 0.995,
        embedding_model: str | None = None,
        recency_weight: float = 1.0,
        relevance_weight: float = 1.0,
        importance_weight: float = 1.0,
    ):
        """
        Initialize the EpisodicMemory
        """
        if not llm_model:
            raise ValueError(
                "llm_model must be provided for the usage of episodic memory"
            )

        super().__init__(agent, llm_model=llm_model, display=display)

        self.max_capacity = max_capacity
        self.memory_entries = deque(maxlen=self.max_capacity)
        self.considered_entries = considered_entries
        self.recency_decay = recency_decay
        self.embedding_model = embedding_model

        self._embeddings: deque[list[float] | None] = deque(maxlen=max_capacity)
        self.recency_weight = recency_weight
        self.relevance_weight = relevance_weight
        self.importance_weight = importance_weight

        self.system_prompt = """
            You are an assistant that evaluates memory entries on a scale from 1 to 5, based on their importance to a specific problem or task. Your goal is to assign a score that reflects how much each entry contributes to understanding, solving, or advancing the task. Use the following grading scale:

            5 - Critical: Introduces essential, novel information that significantly impacts problem-solving or decision-making.

            4 - High: Provides important context or clarification that meaningfully improves understanding or direction.

            3 - Moderate: Adds somewhat useful information that may assist but is not essential.

            2 - Low: Offers minimal relevance or slight redundancy; impact is marginal.

            1 - Irrelevant: Contains no useful or applicable information for the current problem.

            Only assess based on the entry's content and its value to the task at hand. Ignore style, grammar, or tone.
            """

    def _get_embedding(self, text: str) -> list[float] | None:
        """
        Helper function to convert a text into vector

        Return None if API fails.
        """
        try:
            response = litellm.embedding(model=self.embedding_model, input=[text])
            return response.data[0]["embedding"]
        except Exception:
            return None

    async def _aget_embedding(self, text: str) -> list[float] | None:
        """
        Async version for parallel stepping.
        """
        try:
            response = await litellm.aembedding(
                model=self.embedding_model, input=[text]
            )
            return response.data[0]["embedding"]
        except Exception:
            return None

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

    def _safely_parse_grade(self, raw_content: str) -> int:
        """
        Safely parse the returned grading from the LLM.
        It gracefully handles correctly formatted JSON, raw integers,
        and broken string formats to ensure the simulation does not crash.
        """
        try:
            formatted = json.loads(raw_content)
            return int(formatted["grade"])
        except Exception:
            try:
                return int(raw_content.strip())
            except Exception:
                return 3

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

        return self._safely_parse_grade(rsp.choices[0].message.content)

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
        return self._safely_parse_grade(rsp.choices[0].message.content)

    def retrieve_top_k_entries(
        self, k: int, qry_str: str | None = None
    ) -> list[MemoryEntry]:
        """
        Retrieve the top-k entries using normalized importance and recency.

        Notes:
        - Inspired by Generative Agents retrieval scoring:
          recency/importance/relevance are normalized separately and combined.
        - This implementation currently combines importance + recency + relevance

        - Relevance calculation is carried out using a qry_str(similar to the focal_pt in the paper)
        If a query string (`qry_str`) is provided and an embedding model is available, cosine similarity is computed between the query
        embedding and each memory entry embedding.

        If relevance is not proveided the grading logic falls back to just importance and recency.
        """
        if not self.memory_entries:
            return []

        entries = list(self.memory_entries)[-self.considered_entries :]
        embeddings_slice = list(self._embeddings)[-self.considered_entries :]

        importance_dict = {}
        recency_dict = {}
        relevance_dict = {}
        qry_embedding = None
        current_step = self.agent.model.steps

        for i, entry in enumerate(entries):
            importance_dict[i] = self._extract_importance(entry)

            age = max(0, current_step - (entry.step or 0))
            recency_dict[i] = self.recency_decay**age

        if qry_str is not None and self.embedding_model is not None:
            qry_embedding = self._get_embedding(str(qry_str))

        use_relevance = qry_embedding is not None
        if use_relevance:
            for i, (_, emb) in enumerate(zip(entries, embeddings_slice)):
                if emb is not None:
                    relevance_dict[i] = cos_sim(emb, qry_embedding)
                else:
                    relevance_dict[i] = 0.0

        importance_scaled = normalize_dict_values(dict(importance_dict), 0, 1)
        recency_scaled = normalize_dict_values(dict(recency_dict), 0, 1)
        if use_relevance:
            relevance_scaled = normalize_dict_values(dict(relevance_dict), 0, 1)

        final_scores = {}

        for i in range(len(entries)):
            score = (
                self.recency_weight * recency_scaled[i]
                + self.importance_weight * importance_scaled[i]
            )
            if use_relevance:
                score += self.relevance_weight * relevance_scaled[i]

            final_scores[i] = score

        top_indices = sorted(final_scores, key=final_scores.get, reverse=True)[:k]
        return [entries[i] for i in top_indices]

    def _finalize_entry(self, type: str, graded_content: dict):
        """Create and persist a finalized episodic entry."""
        new_entry = MemoryEntry(
            agent=self.agent,
            content={type: graded_content},
            step=self.agent.model.steps,
        )

        emb = None
        if self.embedding_model:
            try:
                clean_text_parts = [
                    f"{k}: {v}" for k, v in graded_content.items() if k != "importance"
                ]
                clean_str = f"{type} - " + ", ".join(clean_text_parts)
                emb = self._get_embedding(clean_str)
            except Exception:
                emb = None

        self.memory_entries.append(new_entry)
        self._embeddings.append(emb)

    async def _afinalize_entry(self, type: str, graded_content: dict):
        """Async version: Create entry and compute its embedding."""
        new_entry = MemoryEntry(
            agent=self.agent,
            content={type: graded_content},
            step=self.agent.model.steps,
        )

        emb = None
        if self.embedding_model:
            try:
                clean_text_parts = [
                    f"{k}: {v}" for k, v in graded_content.items() if k != "importance"
                ]
                clean_str = f"{type} - " + ", ".join(clean_text_parts)
                emb = await self._aget_embedding(clean_str)
            except Exception:
                emb = None

        self.memory_entries.append(new_entry)
        self._embeddings.append(emb)

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
        await self._afinalize_entry(type, graded_content)

    def get_prompt_ready(self, qry_str: str | None = None) -> str:
        """Pass the query string through to retrieval."""
        entries = self.retrieve_top_k_entries(self.considered_entries, qry_str=qry_str)
        return f"Top {self.considered_entries} memory entries:\n\n" + "\n".join(
            str(entry) for entry in entries
        )

    def get_communication_history(self) -> str:
        """
        Get the communication history accurately formatted for LLMs
        to easily parse sender and message text.
        """
        history = []
        for entry in self.memory_entries:
            if "message" in entry.content:
                msg = entry.content["message"]
                if isinstance(msg, dict):
                    sender = msg.get("sender", "Unknown")
                    text = msg.get("message", str(msg))
                    history.append(f"step {entry.step or 0} - {sender}: {text}")
                else:
                    history.append(f"step {entry.step or 0}: {msg}")
        return "\n\n".join(history)

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
