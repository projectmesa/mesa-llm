from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

from rich.console import Console
from rich.panel import Panel

from mesa_llm.module_llm import ModuleLLM

if TYPE_CHECKING:
    from mesa_llm.llm_agent import LLMAgent


@dataclass
class MemoryEntry:
    """
    A data structure that stores individual memory records with content, step number, and agent reference. Each entry includes `rich` formatting for display. Content is a nested dictionary of arbitrary depth containing the entry's information. Each entry is designed to hold all the information of a given step for an agent, but can also be used to store a single event if needed.
    """

    content: dict
    step: int
    agent: "LLMAgent"

    def __str__(self) -> str:
        """
        Format the memory entry as a string.
        Note : 'content' is a dict that can have nested dictionaries of arbitrary depth
        """

        def format_nested_dict(data, indent_level=0):
            lines = []
            indent = "   " * indent_level

            for key, value in data.items():
                if isinstance(value, dict):
                    lines.append(f"{indent}[blue]└──[/blue] [cyan]{key} :[/cyan]")
                    lines.extend(format_nested_dict(value, indent_level + 1))
                else:
                    lines.append(
                        f"{indent}[blue]└──[/blue] [cyan]{key} : [/cyan]{value}"
                    )

            return lines

        lines = []
        for key, value in self.content.items():
            if not value:
                continue

            lines.append(f"\n[bold cyan][{key.title()}][/bold cyan]")
            if isinstance(value, dict):
                lines.extend(format_nested_dict(value, 1))
            else:
                lines.append(f"   [blue]└──[/blue] [cyan]{value} :[/cyan]")

        content = "\n".join(lines)

        return content

    def display(self):
        if self.agent and hasattr(self.agent, "memory") and self.agent.memory.display:
            title = f"Step [bold purple]{self.agent.model.steps}[/bold purple] [bold]|[/bold] {type(self.agent).__name__} [bold purple]{self.agent.unique_id}[/bold purple]"
            panel = Panel(
                self.__str__(),
                title=title,
                title_align="left",
                border_style="bright_blue",
                padding=(0, 1),
            )
            console = Console()
            console.print(panel)


class Memory(ABC):
    """
    Create a memory generic parent class that can be used to create different types of memories

    Attributes:
        agent : the agent that the memory belongs to
        llm_model : the model to use for the summarization if used
        display : whether to display the memory

    Content Addition
        - Before each agent step, the agent can add new events to the memory through `add_to_memory(type, content)` so that the memory can be used to reason about the most recent events as well as the past events.
        - During the step, actions, messages, and plans are added to the memory through `add_to_memory(type, content)`
        - At the end of the step, the memory is processed via `process_step()`, managing when memory entries are added,consolidated, displayed, or removed
    """

    def __init__(
        self,
        agent: "LLMAgent",
        llm_model: str | None = None,
        display: bool = True,
    ):
        """
        Initialize the memory

        Args:
            llm_model : the model to use for the summarization
            agent : the agent that the memory belongs to
        """
        self.agent = agent
        if llm_model:
            self.llm = ModuleLLM(llm_model=llm_model)

        self.display = display

        self.step_content: dict = {}
        self.last_observation: dict = {}

    @abstractmethod
    def get_prompt_ready(self) -> str:
        """
        Get the memory in a format that can be used for reasoning
        """

    @abstractmethod
    def get_communication_history(self) -> str:
        """
        Get the communication history in a format that can be used for reasoning
        """

    @abstractmethod
    def process_step(self, pre_step: bool = False):
        r"""
        A function that is called before and after the step of the agent is called.
        It is implemented to ensure that the memory is up to date when the agent is starting a new step.

        /!\ If you consider that you do not need this function, you can write "pass" in its implementation.
        """

    def add_to_memory(self, type: str, content: dict):
        """
        Add a new entry to the memory
        """
        if type == "observation":
            # Only store changed parts of observation
            changed_parts = {
                k: v for k, v in content.items() if v != self.last_observation.get(k)
            }
            if changed_parts:
                self.step_content[type] = changed_parts
            self.last_observation = content
        else:
            self.step_content[type] = content
