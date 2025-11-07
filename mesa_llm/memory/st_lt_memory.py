from collections import deque
from typing import TYPE_CHECKING

from mesa_llm.memory.memory import Memory, MemoryEntry

if TYPE_CHECKING:
    from mesa_llm.llm_agent import LLMAgent


class STLTMemory(Memory):
    """
    Create a memory object that stores the agent's short and long term memory

    Attributes:
        agent : the agent that the memory belongs to

    Memory is composed of
        - A short term memory who stores the n (int) most recent interactions (observations, planning, discussions)
        - A long term memory that is a summary of the memories that are removed from short term memory (summary
        completed/refactored as it goes)

    """

    def __init__(
        self,
        agent: "LLMAgent",
        short_term_capacity: int = 5,
        consolidation_capacity: int = 2,
        display: bool = True,
        llm_model: str | None = None,
    ):
        """
        Initialize the memory

        Args:
            short_term_capacity : the number of interactions to store in the short term memory
            llm_model : the model to use for the summarization
            agent : the agent that the memory belongs to
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

        self.short_term_memory = deque()
        self.long_term_memory = ""
        self.system_prompt = """
            You are a helpful assistant that summarizes the short term memory into a long term memory.
            The long term memory should be a summary of the short term memory that is concise and informative.
            If the short term memory is empty, return the long term memory unchanged.
            If the long term memory is not empty, update it to include the new information from the short term memory.
            """

        if self.agent.step_prompt:
            self.system_prompt += f" This is the prompt of the problem you will be tackling:{self.agent.step_prompt}, ensure you summarize the short-term memory into long-term a way that is relevant to the problem at hand."

        self.llm.system_prompt = self.system_prompt

    def _update_long_term_memory(self):
        """
        Update the long term memory by summarizing the short term memory with a LLM
        """

        prompt = f"""
            Short term memory:
                {self.format_short_term()}
            Long term memory:
                {self.long_term_memory}
            """

        self.long_term_memory = self.llm.generate(prompt)

    def process_step(self, pre_step: bool = False):
        """
        Process the step of the agent :
        - Add the new entry to the short term memory
        - Consolidate the memory if the short term memory is over capacity
        - Display the new entry
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
            return

        elif not self.short_term_memory[-1].content.get("step", None):
            pre_step_entry = self.short_term_memory.pop()
            self.step_content.update(pre_step_entry.content)
            new_entry = MemoryEntry(
                agent=self.agent,
                content=self.step_content,
                step=self.agent.model.steps,
            )

            self.short_term_memory.append(new_entry)
            self.step_content = {}

            # Consolidate memory if the short term memory is over capacity
            if (
                len(self.short_term_memory)
                > self.capacity + (self.consolidation_capacity or 0)
                and self.consolidation_capacity
            ):
                self.short_term_memory.popleft()
                self._update_long_term_memory()

            elif len(self.short_term_memory) > self.capacity:
                self.short_term_memory.popleft()

            # Display the new entry
            if self.display:
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
        return [
            f"Short term memory:\n {self.format_short_term()}",
            f"Long term memory: \n{self.format_long_term()}",
        ]

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
