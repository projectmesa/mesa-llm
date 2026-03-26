from collections import deque
from typing import TYPE_CHECKING

from mesa_llm.memory.memory import Memory, MemoryEntry

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
        long_term_char_limit: int = 2000,
    ):
        """
        Initialize the memory

        Args:
            short_term_capacity : the number of interactions to store in the short term memory
            consolidation_capacity : the amount of items to pop and summarize
            llm_model : the model to use for the summarization
            agent : the agent that the memory belongs to
            long_term_char_limit : maximum characters the long_term_memory is allowed to hold before triggering a pruning pass
        """
        if not llm_model:
            raise ValueError(
                "llm_model must be provided for the usage of st/lt memory. You can use the pre-built 'short-term-only' memory without a model."
            )

        super().__init__(
            agent=agent,
            llm_model=llm_model,
            display=display,
        )

        self.capacity = short_term_capacity
        self.consolidation_capacity = (
            consolidation_capacity if consolidation_capacity > 0 else None
        )
        self.long_term_char_limit = long_term_char_limit

        self.short_term_memory = deque()
        self.long_term_memory = ""
        self.system_prompt = """
            You are a helpful assistant that summarizes the short term memory into a long term memory.
            The long term memory should be a summary of the short term memory that is concise and informative.
            If the short term memory is empty, return the long term memory unchanged.
            If the long term memory is not empty, update it to include the new information from the short term memory.
            """

        self.pruning_prompt = """
            You are a helpful assistant that compresses excessively long text memories.
            The following text represents an agent's long-term memory. It has grown too large.
            Please compress it, retaining only the most critical overarching facts and lessons, and discarding less salient or outdated details.

            Current Long-term memory to prune:
            {long_term_memory}
            """

        if self.agent.step_prompt:
            self.system_prompt += f" This is the prompt of the problem you will be tackling:{self.agent.step_prompt}, ensure you summarize the short-term memory into long-term a way that is relevant to the problem at hand."

        self.llm.system_prompt = self.system_prompt

    def _build_consolidation_prompt(
        self, popped_memories: list[MemoryEntry] | None = None
    ) -> str:
        """
        Prompt builder function to reduce redundancy
        """
        if popped_memories:
            lines = []
            for st_memory_entry in popped_memories:
                lines.append(
                    f"Step {st_memory_entry.step}: \n{st_memory_entry.content}"
                )
            short_term_str = "\n".join(lines)
        else:
            short_term_str = self.format_short_term()

        return f"""
            Short term memory:
                {short_term_str}
            Long term memory:
                {self.long_term_memory}
        """

    def _update_long_term_memory(
        self, popped_memories: list[MemoryEntry] | None = None
    ):
        """
        Update the long term memory by summarizing the short term memory with a LLM
        """
        prompt = self._build_consolidation_prompt(popped_memories)
        response = self.llm.generate(prompt)
        self.long_term_memory = response.choices[0].message.content
        self._prune_long_term_memory()

    async def _aupdate_long_term_memory(
        self, popped_memories: list[MemoryEntry] | None = None
    ):
        """
        Async Function to update long term memory
        """
        prompt = self._build_consolidation_prompt(popped_memories)
        response = await self.llm.agenerate(prompt)
        self.long_term_memory = response.choices[0].message.content
        await self._aprune_long_term_memory()

    def _prune_long_term_memory(self):
        """
        If the long term memory exceeds the character bounds, compress it
        """
        if len(self.long_term_memory) > self.long_term_char_limit:
            prompt = self.pruning_prompt.format(long_term_memory=self.long_term_memory)
            response = self.llm.generate(prompt)
            self.long_term_memory = response.choices[0].message.content

    async def _aprune_long_term_memory(self):
        """
        Async evaluation for compressing the long term memory
        """
        if len(self.long_term_memory) > self.long_term_char_limit:
            prompt = self.pruning_prompt.format(long_term_memory=self.long_term_memory)
            response = await self.llm.agenerate(prompt)
            self.long_term_memory = response.choices[0].message.content

    def _process_step_core(self, pre_step: bool):
        """
        Shared core logic for process_step and aprocess_step
        Update short-term memory and decide if consolidation is needed.

        Returns:
            "(new_entry, should_consolidate)"
        """
        # Add the new entry to the short term memory
        if pre_step:
            new_entry = MemoryEntry(
                agent=self.agent,
                content=self.step_content,
                step=None,
            )
            self.short_term_memory.append(new_entry)
            self.step_content = {}
            return None, False
        if not self.short_term_memory or self.short_term_memory[-1].step is not None:
            return None, False
        pre_step_entry = self.short_term_memory.pop()
        self.step_content.update(pre_step_entry.content)
        new_entry = MemoryEntry(
            agent=self.agent,
            content=self.step_content,
            step=self.agent.model.steps,
        )

        self.short_term_memory.append(new_entry)
        self.step_content = {}

        popped_memories = []
        if (
            len(self.short_term_memory)
            > self.capacity + (self.consolidation_capacity or 0)
            and self.consolidation_capacity
        ):
            for _ in range(self.consolidation_capacity):
                popped_memories.append(self.short_term_memory.popleft())

        elif (
            len(self.short_term_memory) > self.capacity
            and not self.consolidation_capacity
        ):
            self.short_term_memory.popleft()

        return new_entry, popped_memories

    def process_step(self, pre_step: bool = False):
        """
        Synchronous memory step handler
        """
        new_entry, popped_memories = self._process_step_core(pre_step)

        if popped_memories:
            self._update_long_term_memory(popped_memories)

        if new_entry and self.display:
            new_entry.display()

    async def aprocess_step(self, pre_step: bool = False):
        """
        Async memory step handler (non-blocking consolidation)
        """
        new_entry, popped_memories = self._process_step_core(pre_step)

        if popped_memories:
            await self._aupdate_long_term_memory(popped_memories)

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
                f"step {entry.step}: {entry.content['message']}\n\n"
                for entry in self.short_term_memory
                if "message" in entry.content
            ]
        )
