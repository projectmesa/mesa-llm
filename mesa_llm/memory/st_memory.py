from collections import deque
from typing import TYPE_CHECKING

from mesa_llm.memory.memory import Memory, MemoryEntry

if TYPE_CHECKING:
    from mesa_llm.llm_agent import LLMAgent


class ShortTermMemory(Memory):
    """
    Simple short-term memory implementation without consolidation (stores recent entries up to capacity limit). Same functionality as `STLTMemory` but without the long-term memory and consolidation mechanism.

    Attributes:
        agent : the agent that the memory belongs to
        n : number of short-term memories to remember
        display : whether to display the memory
        llm_model : the model to use for the summarization
    """

    def __init__(
        self,
        agent: "LLMAgent",
        n: int = 5,
        display: bool = True,
    ):
        super().__init__(
            agent=agent,
            display=display,
        )
        self.n = n
        self.short_term_memory = deque()

    def process_step(self, pre_step: bool = False):
        """
        Process the step of the agent :
        - Add the new entry to the short term memory
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
            pre_step = self.short_term_memory.pop()
            self.step_content.update(pre_step.content)
            new_entry = MemoryEntry(
                agent=self.agent,
                content=self.step_content,
                step=self.agent.model.steps,
            )

            self.short_term_memory.append(new_entry)
            self.step_content = {}

        # Display the new entry
        if self.display:
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
