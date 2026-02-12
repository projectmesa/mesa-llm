from typing import TYPE_CHECKING

from mesa_llm.reasoning.reasoning import Observation, Plan, Reasoning

if TYPE_CHECKING:
    from mesa_llm.llm_agent import LLMAgent


class CoTReasoning(Reasoning):
    """
    Chain of Thought reasoning with explicit step-by-step analysis before action execution. Uses structured numbered thoughts followed by tool execution. Integrates memory context for informed decision-making.

    Attributes:
        - **agent** (LLMAgent reference)

    Methods:
        - **plan(prompt, obs=None, ttl=1, selected_tools=None)** → *Plan* - Generate synchronous plan with CoT reasoning
        - **async aplan(prompt, obs=None, ttl=1, selected_tools=None)** → *Plan* - Generate asynchronous plan with CoT reasoning

    Reasoning Format:
        Thought 1: [Initial reasoning based on observation]
        Thought 2: [How memory informs the situation]
        Thought 3: [Possible alternatives or risks]
        Thought 4: [Final decision and justification]
        Action: [The action you decide to take]
    """

    def __init__(self, agent: "LLMAgent"):
        super().__init__(agent=agent)

    def get_cot_system_prompt(self, obs: Observation) -> str:
        memory = getattr(self.agent, "memory", None)
        long_term_memory = ""
        if (
            memory
            and hasattr(memory, "format_long_term")
            and callable(memory.format_long_term)
        ):
            long_term_memory = memory.format_long_term()

        short_term_memory = ""
        if (
            memory
            and hasattr(memory, "format_short_term")
            and callable(memory.format_short_term)
        ):
            short_term_memory = memory.format_short_term()

        obs_str = str(obs)

        system_prompt = f"""
        You are an autonomous agent operating in a simulation.
        Use a detailed step-by-step reasoning process (Chain-of-Thought) to decide your next action.
        Your memory contains information from past experiences, and your observation provides the current context.

        ---

        # Long-Term Memory
        {long_term_memory}

        ---

        # Short-Term Memory (Recent History)
        {short_term_memory}

        ---

        # Current Observation
        {obs_str}

        ---

        # Instructions
        First think through the situation step-by-step, and explain it in the format given below.
        ------------------------------------------------------
        Thought 1: [Initial reasoning based on the observation]
        Thought 2: [How memory informs the situation]
        Thought 3: [Possible alternatives or risks]
        Thought 4: [Final decision and justification]
        Action: [The action you decide to take]
        ------------------------------------------------------
        Keep the reasoning grounded in the current context and relevant history.


        """
        return system_prompt

    def plan(
        self,
        obs: Observation,
        ttl: int = 1,
        prompt: str | None = None,
        selected_tools: list[str] | None = None,
    ) -> Plan:
        """
        Plan the next (CoT) action based on the current observation and the agent's memory.
        """
        # If no prompt is provided, use the agent's default step prompt
        if prompt is None:
            if self.agent.step_prompt is not None:
                prompt = self.agent.step_prompt
            else:
                raise ValueError("No prompt provided and agent.step_prompt is None.")

        step = obs.step + 1
        llm = self.agent.llm
        obs_str = str(obs)

        # Add current observation to memory (for record)
        self.agent.memory.add_to_memory(type="Observation", content=obs_str)
        system_prompt = self.get_cot_system_prompt(obs)

        llm.system_prompt = system_prompt
        rsp = llm.generate(
            prompt=prompt,
            tool_schema=self.agent.tool_manager.get_all_tools_schema(selected_tools),
            tool_choice="none",
        )

        chaining_message = rsp.choices[0].message.content
        self.agent.memory.add_to_memory(type="Plan", content=chaining_message)

        # Pass plan content to agent for display
        if hasattr(self.agent, "_step_display_data"):
            self.agent._step_display_data["plan_content"] = chaining_message
        system_prompt = "You are an executor that executes the plan given to you in the prompt through tool calls."
        llm.system_prompt = system_prompt
        rsp = llm.generate(
            prompt=chaining_message,
            tool_schema=self.agent.tool_manager.get_all_tools_schema(selected_tools),
            tool_choice="required",
        )
        response_message = rsp.choices[0].message
        cot_plan = Plan(step=step, llm_plan=response_message, ttl=1)

        self.agent.memory.add_to_memory(type="Plan-Execution", content=str(cot_plan))

        return cot_plan

    async def aplan(
        self,
        prompt: str,
        obs: Observation,
        ttl: int = 1,
        selected_tools: list[str] | None = None,
    ) -> Plan:
        """
        Asynchronous version of plan() method for parallel planning.
        """
        step = obs.step + 1
        llm = self.agent.llm

        system_prompt = self.get_cot_system_prompt(obs)
        llm.system_prompt = system_prompt

        rsp = await llm.agenerate(
            prompt=prompt,
            tool_schema=self.agent.tool_manager.get_all_tools_schema(selected_tools),
            tool_choice="none",
        )

        chaining_message = rsp.choices[0].message.content
        self.agent.memory.add_to_memory(type="Plan", content=chaining_message)

        # Pass plan content to agent for display
        if hasattr(self.agent, "_step_display_data"):
            self.agent._step_display_data["plan_content"] = chaining_message
        system_prompt = "You are an executor that executes the plan given to you in the prompt through tool calls."
        llm.system_prompt = system_prompt
        rsp = await llm.agenerate(
            prompt=chaining_message,
            tool_schema=self.agent.tool_manager.get_all_tools_schema(selected_tools),
            tool_choice="required",
        )
        response_message = rsp.choices[0].message
        cot_plan = Plan(step=step, llm_plan=response_message, ttl=1)

        self.agent.memory.add_to_memory(type="Plan-Execution", content=str(cot_plan))

        return cot_plan
