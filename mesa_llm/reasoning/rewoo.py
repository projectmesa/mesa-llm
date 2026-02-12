from typing import TYPE_CHECKING

from mesa_llm.reasoning.reasoning import (
    Observation,
    Plan,
    Reasoning,
)

if TYPE_CHECKING:
    from mesa_llm.llm_agent import LLMAgent


class ReWOOReasoning(Reasoning):
    """
    Reasoning Without Observation for multi-step planning without environmental feedback. Enables multi-step planning without requiring immediate environmental feedback. Plans remain valid across multiple simulation steps with extended TTL. Reduces computational overhead through strategic long-term thinking.

    Attributes:
        - **agent** (LLMAgent reference)
        - **remaining_tool_calls** (int) - Number of tool calls remaining in current plan
        - **current_plan** (Plan) - Currently active multi-step plan
        - **current_obs** (Observation) - Last observation used for planning

    Methods:
        - **plan(prompt, obs=None, ttl=1, selected_tools=None)** → *Plan* - Generate synchronous plan with ReWOO reasoning
        - **async aplan(prompt, obs=None, ttl=1, selected_tools=None)** → *Plan* - Generate asynchronous plan with ReWOO reasoning
    """

    def __init__(self, agent: "LLMAgent"):
        super().__init__(agent=agent)
        self.remaining_tool_calls = 0  # Initialize remaining tool calls
        self.current_plan: Plan | None = None
        self.current_obs: Observation | None = None

    def get_rewoo_system_prompt(self, obs: Observation) -> str:
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

        system_prompt = f"""
        You are an autonomous agent that creates multi-step plans without re-observing during execution.
        Using the ReWOO (Reasoning WithOut Observation) approach, you will create a comprehensive plan
        that anticipates multiple steps ahead based on your current observation and memory.

        ---

        # Long-Term Memory
        {long_term_memory}

        ---

        # Short-Term Memory (Recent History)
        {short_term_memory}

        ---

        # Current Observation
        {self.current_obs}

        ---

        # Instructions
        Create a detailed multi-step plan that can be executed without needing new observations.
        Your plan should anticipate likely scenarios and include contingencies.

        Determine the optimal number of steps (1-5) based on the complexity of the task and available tools.
        Use this format:


            "plan": "Describe your overall strategy and reasoning",
            "step_1": "First action with expected outcome",
            "step_2": "Second action building on Step 1 (optional)",
            "step_3": "Third action if needed (optional)",
            "step_4": "Fourth action if needed (optional)",
            "step_5": "Final action if needed (optional)",
            "contingency": "What to do if things don't go as expected"


        Only include the steps you need (step_1 is required, step_2 through step_5 are optional).
        Set unused step fields to null. The plan should be comprehensive enough to execute
        for multiple simulation steps without requiring new environmental observations.
        Refer to available tools when planning actions.

        ---
        """
        return system_prompt

    def plan(
        self,
        prompt: str | None = None,
        obs: Observation | None = None,
        selected_tools: list[str] | None = None,
    ) -> Plan:
        """
        Plan the next (ReWOO) action based on the current observation and the agent's memory.
        """
        # If no prompt is provided, use the agent's default step prompt
        if prompt is None:
            if self.agent.step_prompt is not None:
                prompt = self.agent.step_prompt
            else:
                raise ValueError("No prompt provided and agent.step_prompt is None.")

        # If we have remaining tool calls, skip observation and plan generation
        if self.remaining_tool_calls > 0:
            index_of_tool = (
                len(self.current_plan.tool_calls) - self.remaining_tool_calls
            )
            self.remaining_tool_calls -= 1
            tool_call = [self.current_plan.tool_calls[index_of_tool]]
            current_plan = self.current_plan
            current_plan.tool_calls = tool_call
            return Plan(llm_plan=current_plan, step=self.current_obs.step, ttl=1)

        self.current_obs = self.agent.generate_obs()
        llm = self.agent.llm
        system_prompt = self.get_rewoo_system_prompt(self.current_obs)

        llm.system_prompt = system_prompt
        rsp = llm.generate(
            prompt=prompt,
            tool_schema=self.agent.tool_manager.get_all_tools_schema(selected_tools),
            tool_choice="none",
        )

        self.agent.memory.add_to_memory(
            type="plan", content=rsp.choices[0].message.content
        )

        rewoo_plan = self.execute_tool_call(
            rsp.choices[0].message.content, selected_tools
        )
        # Count the number of tool calls in the response and set remaining_tool_calls
        if hasattr(rewoo_plan.llm_plan, "tool_calls"):
            self.remaining_tool_calls = len(rewoo_plan.llm_plan.tool_calls)
        else:
            self.remaining_tool_calls = 0
        self.current_plan = rewoo_plan.llm_plan

        return rewoo_plan

    async def aplan(self, prompt: str, selected_tools: list[str] | None = None) -> Plan:
        """
        Asynchronous version of plan() method for parallel planning.
        """

        # If we have remaining tool calls, skip observation and plan generation
        if self.remaining_tool_calls > 0:
            index_of_tool = (
                len(self.current_plan.tool_calls) - self.remaining_tool_calls
            )
            self.remaining_tool_calls -= 1
            tool_call = [self.current_plan.tool_calls[index_of_tool]]
            current_plan = self.current_plan
            current_plan.tool_calls = tool_call
            return Plan(llm_plan=current_plan, step=self.current_obs.step, ttl=1)

        self.current_obs = self.agent.generate_obs()
        llm = self.agent.llm
        system_prompt = self.get_rewoo_system_prompt(self.current_obs)

        llm.system_prompt = system_prompt
        rsp = await llm.agenerate(
            prompt=prompt,
            tool_schema=self.agent.tool_manager.get_all_tools_schema(selected_tools),
            tool_choice="none",
        )

        self.agent.memory.add_to_memory(
            type="plan", content=rsp.choices[0].message.content
        )

        rewoo_plan = await self.aexecute_tool_call(
            rsp.choices[0].message.content, selected_tools
        )
        # Count the number of tool calls in the response and set remaining_tool_calls
        if hasattr(rewoo_plan.llm_plan, "tool_calls"):
            self.remaining_tool_calls = len(rewoo_plan.llm_plan.tool_calls)
        else:
            self.remaining_tool_calls = 0
        self.current_plan = rewoo_plan.llm_plan

        return rewoo_plan
