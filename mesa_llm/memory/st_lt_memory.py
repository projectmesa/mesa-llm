import json
from collections import deque
from math import exp
from typing import TYPE_CHECKING

from mesa_llm.memory.memory import Memory, MemoryEntry, _format_message_entry

if TYPE_CHECKING:
    from mesa_llm.llm_agent import LLMAgent


class STLTMemory(Memory):
    """
    Implements a dual-memory system where recent experiences are stored in short-term memory with limited capacity, and older memories are consolidated into long-term summaries using LLM-based summarization.

    Attributes:
        agent : the agent that the memory belongs to

    Memory is composed of
        - A short term memory who stores the n (int) most recent interactions (observations, planning, discussions)
        - A long term memory that is a summary of the memories that are removed from short term memory (summary
        completed/refactored as it goes)

    Logic behind the implementation
        - **Short-term capacity**: Configurable number of recent memory entries (default: short_term_capacity = 5)
        - **Consolidation**: When capacity is exceeded, oldest entries are summarized into long-term memory (number of entries to summarize is configurable, default: consolidation_capacity = 3)
        - **LLM Summarization**: Uses a separate LLM instance to create meaningful summaries of past experiences

    """

    def __init__(
        self,
        agent: "LLMAgent",
        short_term_capacity: int = 5,
        consolidation_capacity: int = 2,
        display: bool = True,
        llm_model: str | None = None,
        api_base: str | None = None,
        max_tokens: int | None = None,
        enable_salience_pruning: bool = True,
        salience_recency_weight: float = 0.3,
        salience_threshold: float = 0.2,
    ):
        """
        Initialize the memory with enhanced token counting and salience pruning.

        Args:
            short_term_capacity : the number of interactions to store in the short term memory
            consolidation_capacity : number of entries to summarize at once
            llm_model : the model to use for the summarization
            api_base : the API base URL to use for the LLM provider
            agent : the agent that the memory belongs to
            max_tokens : maximum tokens allowed in short-term memory (None = unlimited)
            enable_salience_pruning : whether to use salience-based eviction
            salience_recency_weight : weight for recency in salience calculation (0-1)
            salience_threshold : minimum salience to keep an entry (0-1)
        """
        if not llm_model:
            raise ValueError(
                "llm_model must be provided for the usage of st/lt memory. You can use the pre-built 'short-term-only' memory without a model."
            )

        super().__init__(
            agent=agent,
            llm_model=llm_model,
            api_base=api_base,
            display=display,
        )

        self.capacity = short_term_capacity
        self.consolidation_capacity = (
            consolidation_capacity if consolidation_capacity > 0 else None
        )

        # Token management
        self.max_tokens = max_tokens
        self.token_buffer_threshold = 0.8  # Trigger consolidation at 80% capacity

        # Salience pruning
        self.enable_salience_pruning = enable_salience_pruning
        self.salience_recency_weight = salience_recency_weight
        self.salience_threshold = salience_threshold

        self.short_term_memory = deque()
        self.long_term_memory = ""
        self.memory_creation_times = {}  # Track when each memory was created
        self._token_count_cache = {}  # Cache token counts

        self.system_prompt = """
            You are a helpful assistant that summarizes the short term memory into a long term memory.
            The long term memory should be a summary of the short term memory that is concise and informative.
            If the short term memory is empty, return the long term memory unchanged.
            If the long term memory is not empty, update it to include the new information from the short term memory.
            """

        if self.agent.step_prompt:
            self.system_prompt += f" This is the prompt of the problem you will be tackling:{self.agent.step_prompt}, ensure you summarize the short-term memory into long-term a way that is relevant to the problem at hand."

        self.llm.system_prompt = self.system_prompt

    def _build_consolidation_prompt(self, evicted_entries: list[MemoryEntry]) -> str:
        """
        Build a prompt that asks the LLM to integrate *evicted* memories
        into the existing long-term summary.

        Args:
            evicted_entries: the oldest short-term entries that were just
                removed from the deque and need to be summarized.
        """
        evicted_text = "\n".join(
            f"Step {e.step}: \n{e.content}" for e in evicted_entries
        )
        return (
            "Memories to consolidate (oldest entries being removed "
            "from short-term memory):\n"
            f"{evicted_text}\n\n"
            f"Existing long term memory:\n{self.long_term_memory}\n\n"
            "Please integrate the above memories into a concise, updated "
            "long-term memory summary."
        )

    def _count_tokens(self, text: str) -> int:
        """
        Estimate token count using a simple heuristic: ~4 characters per token.
        For production, use `tiktoken` or litellm's token counting.

        Args:
            text: Text to count tokens for

        Returns:
            Estimated token count
        """
        # Simple heuristic: assume ~4 characters per token (GPT-3 tokenizer)
        # This is a conservative estimate for planning
        return max(1, len(str(text)) // 4)

    def _get_memory_tokens(self, entry: MemoryEntry) -> int:
        """
        Calculate total tokens for a memory entry (step + content).

        Args:
            entry: MemoryEntry to count tokens for

        Returns:
            Total token count
        """
        # Convert entry to string representation
        entry_str = f"Step {entry.step}: {json.dumps(entry.content, default=str)}"
        return self._count_tokens(entry_str)

    def _calculate_short_term_tokens(self) -> int:
        """
        Calculate total tokens currently in short-term memory.

        Returns:
            Total token count of all entries in short-term memory
        """
        total_tokens = 0
        for entry in self.short_term_memory:
            total_tokens += self._get_memory_tokens(entry)
        return total_tokens

    def _calculate_recency_score(self, entry_index: int, total_entries: int) -> float:
        """
        Calculate recency score using exponential decay.
        Most recent entries get higher scores.

        Args:
            entry_index: Position in short-term memory (0 = oldest)
            total_entries: Total entries in memory

        Returns:
            Recency score (0-1), where 1 is most recent
        """
        if total_entries <= 1:
            return 1.0

        # Normalize position (0 = oldest, 1 = newest)
        relative_position = entry_index / (total_entries - 1)

        # Apply exponential decay: newer entries have higher scores
        # Use decay factor of 2 for steeper falloff
        recency_score = exp(relative_position * 2 - 2)
        return min(1.0, recency_score)

    def _calculate_relevance_score(self, entry: MemoryEntry) -> float:
        """
        Calculate relevance score based on keyword/content analysis.
        Checks if entry contains action-related keywords.

        Args:
            entry: MemoryEntry to score

        Returns:
            Relevance score (0-1)
        """
        content_str = json.dumps(entry.content, default=str).lower()

        # List of high-value keywords indicating important memory
        important_keywords = [
            "action",
            "decision",
            "error",
            "failure",
            "success",
            "goal",
            "objective",
            "completed",
            "failed",
            "interaction",
            "result",
            "observation",
            "plan",
            "tool",
        ]

        # Count keyword matches
        keyword_matches = sum(1 for kw in important_keywords if kw in content_str)

        # Score based on keyword density (0-1 scale)
        # Assume max importance ~5 keywords, normalize to 0-1
        relevance_score = min(1.0, keyword_matches / 5.0)

        return relevance_score

    def _calculate_salience_score(self, entry_index: int, entry: MemoryEntry) -> float:
        """
        Calculate combined salience score for an entry.
        Salience = (1 - recency_weight) * relevance + recency_weight * recency

        Args:
            entry_index: Position in short-term memory
            entry: The MemoryEntry to score

        Returns:
            Salience score (0-1), where higher = more important to keep
        """
        total_entries = len(self.short_term_memory)

        recency_score = self._calculate_recency_score(entry_index, total_entries)
        relevance_score = self._calculate_relevance_score(entry)

        # Combined salience score
        salience = (
            1 - self.salience_recency_weight
        ) * relevance_score + self.salience_recency_weight * recency_score

        return salience

    def _update_long_term_memory(self, evicted_entries: list[MemoryEntry]):
        """
        Update the long term memory by summarizing the evicted entries
        """
        prompt = self._build_consolidation_prompt(evicted_entries)
        response = self.llm.generate(prompt)
        self.long_term_memory = response.choices[0].message.content

    async def _aupdate_long_term_memory(self, evicted_entries: list[MemoryEntry]):
        """
        Async version of _update_long_term_memory
        """
        prompt = self._build_consolidation_prompt(evicted_entries)
        response = await self.llm.agenerate(prompt)
        self.long_term_memory = response.choices[0].message.content

    def _process_step_core(self, pre_step: bool):
        """
        Shared core logic for process_step and aprocess_step.

        Update short-term memory and decide if consolidation is needed.
        When entries are evicted for consolidation they are captured and
        returned so the caller can pass them to the LLM for summarization.

        Uses smart eviction strategies:
        - If max_tokens set: evict lowest-salience entries when nearing limit
        - If salience pruning enabled: drop entries below threshold
        - Otherwise: use original FIFO strategy

        Returns:
            ``(new_entry, evicted_entries)`` where *evicted_entries* is a
            (possibly empty) list of MemoryEntry objects that were removed
            from short-term memory and should be consolidated.
        """
        if pre_step:
            new_entry = MemoryEntry(
                agent=self.agent,
                content=self.step_content,
                step=None,
            )
            self.short_term_memory.append(new_entry)
            self.step_content = {}
            return None, []

        if not self.short_term_memory or self.short_term_memory[-1].step is not None:
            return None, []

        pre_step_entry = self.short_term_memory.pop()
        self.step_content.update(pre_step_entry.content)
        new_entry = MemoryEntry(
            agent=self.agent,
            content=self.step_content,
            step=self.agent.model.steps,
        )
        self.short_term_memory.append(new_entry)
        self.step_content = {}

        evicted: list[MemoryEntry] = []

        # **ENHANCED EVICTION LOGIC**

        # First, apply salience-based pruning if enabled
        if self.enable_salience_pruning:
            entries_with_salience = []
            for idx, entry in enumerate(self.short_term_memory):
                salience = self._calculate_salience_score(idx, entry)
                entries_with_salience.append((salience, idx, entry))

            # Drop low-salience entries below threshold
            low_salience_entries = [
                e for s, idx, e in entries_with_salience if s < self.salience_threshold
            ]

            if low_salience_entries:
                # Compact memory by removing low-salience items
                for entry in low_salience_entries:
                    if entry in self.short_term_memory:
                        evicted.append(entry)
                        self.short_term_memory.remove(entry)

        # Second, check token limits if configured
        if self.max_tokens:
            current_tokens = self._calculate_short_term_tokens()
            token_limit = self.max_tokens * self.token_buffer_threshold

            if current_tokens > token_limit:
                # Need to evict entries - use salience-based selection
                entries_with_salience = []
                for idx, entry in enumerate(self.short_term_memory):
                    salience = self._calculate_salience_score(idx, entry)
                    tokens = self._get_memory_tokens(entry)
                    entries_with_salience.append((salience, tokens, entry))

                # Sort by salience ascending (lowest salience first)
                entries_with_salience.sort(key=lambda x: x[0])

                # Evict lowest-salience entries until below token limit
                tokens_freed = 0
                target_tokens = current_tokens * 0.75  # Target 75% of max
<<<<<<< HEAD

                for _salience, tokens, entry in entries_with_salience:
=======

                for salience, tokens, entry in entries_with_salience:
>>>>>>> de89bf431885d8ddf234d06e4d82edae73c0c11f
                    if tokens_freed >= (current_tokens - target_tokens):
                        break

                    if entry in self.short_term_memory:
                        evicted.append(entry)
                        self.short_term_memory.remove(entry)
                        tokens_freed += tokens

        # Third, apply original capacity-based eviction (count-based)
        if (
            len(self.short_term_memory)
            > self.capacity + (self.consolidation_capacity or 0)
            and self.consolidation_capacity
        ):
            # Pop consolidation_capacity oldest entries for summarization
            for _ in range(self.consolidation_capacity):
                if self.short_term_memory:
                    entry = self.short_term_memory.popleft()
                    if entry not in evicted:
                        evicted.append(entry)

        elif (
            len(self.short_term_memory) > self.capacity
            and not self.consolidation_capacity
            and self.short_term_memory
        ):
            # No consolidation configured — just discard the oldest entry
            entry = self.short_term_memory.popleft()
            if entry not in evicted:
                evicted.append(entry)

        return new_entry, evicted

    def process_step(self, pre_step: bool = False):
        """
        Synchronous memory step handler
        """
        new_entry, evicted = self._process_step_core(pre_step)

        if evicted:
            self._update_long_term_memory(evicted)

        if new_entry and self.display:
            new_entry.display()

    async def aprocess_step(self, pre_step: bool = False):
        """
        Async memory step handler (non-blocking consolidation)
        """
        new_entry, evicted = self._process_step_core(pre_step)

        if evicted:
            await self._aupdate_long_term_memory(evicted)

        if new_entry and self.display:
            new_entry.display()

    def format_long_term(self) -> str:
        """
        Get the long term memory
        """
        return str(self.long_term_memory)

    def format_short_term(self) -> str:
        """
        Get the short term memory
        """
        if not self.short_term_memory:
            return "No recent memory."

        else:
            lines = []
            for st_memory_entry in self.short_term_memory:
                lines.append(
                    f"Step {st_memory_entry.step}: \n{st_memory_entry.content}"
                )
            return "\n".join(lines)

    def get_prompt_ready(self) -> str:
        return (
            f"Short term memory:\n {self.format_short_term()}\n\n"
            f"Long term memory: \n{self.format_long_term()}"
        )

    def get_communication_history(self) -> str:
        """
        Get the communication history
        """
        return "\n".join(
            [
                f"step {entry.step}: {_format_message_entry(entry.content['message'])}\n\n"
                for entry in self.short_term_memory
                if "message" in entry.content
            ]
        )

    def get_token_usage(self) -> dict:
        """
        Get token usage statistics for monitoring.

        Returns:
            Dictionary with token metrics:
            - 'short_term_tokens': Current tokens in short-term memory
            - 'max_tokens': Token limit (None if unlimited)
            - 'token_utilization': Percentage of limit used (0-100)
            - 'entries': Number of entries
        """
        st_tokens = self._calculate_short_term_tokens()
        utilization = 0.0

        if self.max_tokens:
            utilization = (st_tokens / self.max_tokens) * 100

        return {
            "short_term_tokens": st_tokens,
            "max_tokens": self.max_tokens,
            "token_utilization_percent": min(100.0, utilization),
            "entries": len(self.short_term_memory),
        }

    def get_memory_stats(self) -> dict:
        """
        Get comprehensive memory statistics including salience scores.

        Returns:
            Dictionary with:
            - 'entries': List of entries with salience scores
            - 'average_salience': Average salience across all entries
            - 'token_usage': Token usage dictionary
        """
        entries_info = []
        salience_scores = []

        for idx, entry in enumerate(self.short_term_memory):
            salience = self._calculate_salience_score(idx, entry)
            tokens = self._get_memory_tokens(entry)
            salience_scores.append(salience)

            entries_info.append(
                {
                    "step": entry.step,
                    "tokens": tokens,
                    "salience": round(salience, 3),
                    "relevance": round(self._calculate_relevance_score(entry), 3),
                    "recency": round(
                        self._calculate_recency_score(idx, len(self.short_term_memory)),
                        3,
                    ),
                }
            )

        avg_salience = (
            sum(salience_scores) / len(salience_scores) if salience_scores else 0.0
        )

        return {
            "entries": entries_info,
            "average_salience": round(avg_salience, 3),
            "token_usage": self.get_token_usage(),
            "pruning_enabled": self.enable_salience_pruning,
            "salience_threshold": self.salience_threshold,
        }

    def format_short_term_with_stats(self) -> str:
        """
        Get short-term memory formatted with salience and token information.
        Useful for debugging memory state.

        Returns:
            Formatted string showing memory entries with their scores
        """
        if not self.short_term_memory:
            return "No recent memory."

        stats = self.get_memory_stats()
        lines = ["=== SHORT-TERM MEMORY (with Salience Scores) ==="]

        token_usage = stats["token_usage"]
        lines.append(
            f"Token usage: {token_usage['short_term_tokens']}/{token_usage['max_tokens'] or 'unlimited'} ({token_usage['token_utilization_percent']:.1f}%)"
        )
        lines.append(f"Average salience: {stats['average_salience']:.3f}\n")

        for entry_info in stats["entries"]:
            lines.append(
                f"Step {entry_info['step']}: "
                f"Salience={entry_info['salience']:.3f} "
                f"(R:{entry_info['relevance']:.2f} + T:{entry_info['recency']:.2f}) "
                f"Tokens={entry_info['tokens']}"
            )

        return "\n".join(lines)
