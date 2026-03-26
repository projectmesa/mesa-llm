from typing import TYPE_CHECKING

from mesa_llm.memory.memory import Memory, MemoryEntry

if TYPE_CHECKING:
    from mesa_llm.llm_agent import LLMAgent


class LongTermMemory(Memory):
    """
    Purely long-term memory class that tries to store everything the agent experiences.

    Attributes:
        agent : the agent that the memory belongs to
        display : whether to display the memory
        llm_model : the model to use for the summarization

    """

    def __init__(
        self,
        agent: "LLMAgent",
        display: bool = True,
        llm_model: str = "openai/gpt-4o-mini",
        api_base: str | None = None,
    ):
        if not llm_model:
            raise ValueError(
                "llm_model must be provided for the usage of long term memory"
            )

        super().__init__(
            agent=agent,
            llm_model=llm_model,
            api_base=api_base,
            display=display,
        )

        self.long_term_memory = ""
        self.system_prompt = """
            You are a helpful assistant that summarizes all memory entries and stores it into long-term.
            The long term memory should be a summary of the individual memory entries such that it is concise and informative.
            """
        self.buffer = None
        if self.agent.step_prompt:
            self.system_prompt += f" This is the prompt of the problem you will be tackling:{self.agent.step_prompt}, ensure you summarize the memory entries into long-term a way that is relevant to the problem at hand."

        self.llm.system_prompt = self.system_prompt

    def _build_consolidation_prompt(self) -> str:
        """
        Common function for both _update_long_term_memory() and _aupdate_long_term_memory()
        that provides it with a common prompt to redeuce code redundancy
        """
        return f"""
            This is the current Long term memory:
                {self.long_term_memory}
            This is the new memory entry:
                {self.buffer}

            """

    def _update_long_term_memory(self):
        """
        Update the long term memory by summarizing the short term memory with a LLM
        """
        prompt = self._build_consolidation_prompt()
        response = self.llm.generate(prompt)
        self.long_term_memory = response.choices[0].message.content

    async def _aupdate_long_term_memory(self):
        """
        Asynchronous version of _update_long_term_memory
        """
        prompt = self._build_consolidation_prompt()
        response = await self.llm.agenerate(prompt)
        self.long_term_memory = response.choices[0].message.content

    def process_step(self, pre_step: bool = False):
        """
        Process the step of the agent:
        - Merge the new entry into long term memory
        - Display the new entry (Will display it only when a new entry is created in this call)
        """
        created = False

        if pre_step:
            new_entry = MemoryEntry(
                agent=self.agent,
                content=self.step_content,
                step=None,
            )
            self.buffer = new_entry
            self.step_content = {}
            return

        elif self.buffer and self.buffer.step is None:
            self.step_content.update(self.buffer.content)
            new_entry = MemoryEntry(
                agent=self.agent,
                content=self.step_content,
                step=self.agent.model.steps,
            )
            self.buffer = new_entry
            self._update_long_term_memory()
            self.step_content = {}
            created = True

        if self.display and created:
            self.buffer.display()

    async def aprocess_step(self, pre_step: bool = False):
        """
        Asynchronous version of process_step (non-blocking)
        """
        created = False

        if pre_step:
            new_entry = MemoryEntry(
                agent=self.agent,
                content=self.step_content,
                step=None,
            )
            self.buffer = new_entry
            self.step_content = {}
            return

        elif self.buffer and self.buffer.step is None:
            self.step_content.update(self.buffer.content)
            new_entry = MemoryEntry(
                agent=self.agent,
                content=self.step_content,
                step=self.agent.model.steps,
            )
            self.buffer = new_entry
            await self._aupdate_long_term_memory()
            self.step_content = {}
            created = True

        if self.display and created:
            self.buffer.display()

    def format_long_term(self) -> str:
        """
        Get the long term memory
        """
        return str(self.long_term_memory)

    def get_prompt_ready(self) -> str:
        return f"Long term memory: \n{self.format_long_term()}"

    def get_communication_history(self) -> str:
        """
        Get the communication history
        """
        return "communication history is in memory of the agent"
