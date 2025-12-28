import json
from collections import deque
from typing import TYPE_CHECKING

from pydantic import BaseModel

from mesa_llm.memory.memory import Memory, MemoryEntry

if TYPE_CHECKING:
    from mesa_llm.llm_agent import LLMAgent


class EventGrade(BaseModel):
    grade: int


class EpisodicMemory(Memory):
    """
    Stores memories based on event importance scoring. Each new memory entry is evaluated by a LLM
    for its relevance and importance (1-5 scale) relative to the agent's current task and previous
    experiences. Based on a Stanford/DeepMind paper:
    [Generative Agents: Interactive Simulacra of Human Behavior](https://arxiv.org/pdf/2304.03442)
    """

    def __init__(
        self,
        agent: "LLMAgent",
        llm_model: str | None = None,
        display: bool = True,
        max_capacity: int = 10,
        considered_entries: int = 5,
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

        self.system_prompt = """
            You are an assistant that evaluates memory entries on a scale from 1 to 5, based on their importance to a specific problem or task. Your goal is to assign a score that reflects how much each entry contributes to understanding, solving, or advancing the task. Use the following grading scale:

            5 - Critical: Introduces essential, novel information that significantly impacts problem-solving or decision-making.

            4 - High: Provides important context or clarification that meaningfully improves understanding or direction.

            3 - Moderate: Adds somewhat useful information that may assist but is not essential.

            2 - Low: Offers minimal relevance or slight redundancy; impact is marginal.

            1 - Irrelevant: Contains no useful or applicable information for the current problem.

            Only assess based on the entry's content and its value to the task at hand. Ignore style, grammar, or tone.
            """

    def grade_event_importance(self, type: str, content: dict) -> float:
        """
        Grade this event based on the content respect to the previous memory entries
        """
        if len(self.memory_entries) in range(5):
            previous_entries = "previous memory entries:\n\n".join(
                [str(entry) for entry in self.memory_entries]
            )
        elif len(self.memory_entries) > 5:
            previous_entries = "previous memory entries:\n\n".join(
                [str(entry) for entry in self.memory_entries[-5:]]
            )
        else:
            previous_entries = "No previous memory entries"

        prompt = f"""
            grade the importance of the following event on a scale from 1 to 5:
            {type}: {content}
            ------------------------------
            {previous_entries}
            """

        self.llm.system_prompt = self.system_prompt

        rsp = self.agent.llm.generate(
            prompt=prompt,
            response_format=EventGrade,
        )

        formatted_response = json.loads(rsp.choices[0].message.content)
        return formatted_response["grade"]

    def retrieve_top_k_entries(self, k: int) -> list[MemoryEntry]:
        """
        Retrieve the top k entries based on the importance and recency
        """
        top_list = sorted(
            self.memory_entries,
            key=lambda x: x.content["importance"] - (self.agent.model.steps - x.step),
            reverse=True,
        )

        return top_list[:k]

    def add_to_memory(self, type: str, content: dict):
        """
        Add a new memory entry to the memory
        """
        content["importance"] = self.grade_event_importance(type, content)

        super().add_to_memory(type, content)

    def get_prompt_ready(self) -> str:
        return f"Top {self.considered_entries} memory entries:\n\n" + "\n".join(
            [
                str(entry)
                for entry in self.retrieve_top_k_entries(self.considered_entries)
            ]
        )

    def get_communication_history(self) -> str:
        """
        Get the communication history
        """
        return "\n".join(
            [
                f"step {entry.step}: {entry.content['message']}\n\n"
                for entry in self.memory_entries
                if "message" in entry.content
            ]
        )

    def process_step(self, pre_step: bool = False):
        """
        Process the step of the agent :
        - Add the new entry to the memory
        - Display the new entry
        """
        if pre_step:
            self.add_to_memory(type="observation", content=self.step_content)
            self.step_content = {}
            return
