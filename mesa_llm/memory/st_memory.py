from collections import deque
from typing import TYPE_CHECKING

from mesa_llm.memory.memory import Memory, MemoryEntry
from mesa_llm.model_step import get_model_step

if TYPE_CHECKING:
    from mesa_llm.llm_agent import LLMAgent


class ShortTermMemory(Memory):
    """
    Simple short-term memory implementation without consolidation (stores recent entries up to capacity limit). Same functionality as `STLTMemory` but without the long-term memory and consolidation mechanism.

    Attributes:
        agent : the agent that the memory belongs to
        n : positive number of short-term memories to remember
        display : whether to display the memory
        llm_model : the model to use for the summarization
    """

    def __init__(
        self,
        agent: "LLMAgent",
        n: int = 5,
        display: bool = True,
    ):
        if n < 1:
            raise ValueError("n must be >= 1 for ShortTermMemory")

        super().__init__(
            agent=agent,
            display=display,
        )
        self.n = n
        self.short_term_memory = deque(maxlen=self.n)
        self._current_step_entry: MemoryEntry | None = None

    async def aprocess_step(self, pre_step: bool = False):
        """
        Asynchronous version of process_step
        """
        return self.process_step(pre_step=pre_step)

    def process_step(self, pre_step: bool = False):
        """
        Process the step of the agent :
        - Capture pre-step content into the current in-progress step entry
        - Merge current and post-step content into one finalized entry
        - Display the new entry
        """

        # Save a temporary pre-step snapshot. This entry is not persisted in deque.
        if pre_step:
            self._current_step_entry = MemoryEntry(
                agent=self.agent,
                content=self.step_content,
                step=None,
            )
            self.step_content = {}
            return

        new_entry = None
        if self._current_step_entry is not None:
            merged_content = dict(self.step_content)
            merged_content.update(self._current_step_entry.content)
            new_entry = MemoryEntry(
                agent=self.agent,
                content=merged_content,
                step=get_model_step(self.agent.model),
            )
            self.short_term_memory.append(new_entry)
            self._current_step_entry = None
            self.step_content = {}

        # Display the new entry
        if self.display and new_entry is not None:
            new_entry.display()

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
        return f"Short term memory:\n {self.format_short_term()}\n"

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
