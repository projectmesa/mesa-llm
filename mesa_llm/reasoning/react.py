import json
from typing import TYPE_CHECKING

from pydantic import BaseModel

from mesa_llm.reasoning.reasoning import Observation, Plan, Reasoning

if TYPE_CHECKING:
    from mesa_llm.llm_agent import LLMAgent


class ReActOutput(BaseModel):
    reasoning: str
    action: str


class ReActReasoning(Reasoning):
    """
    Reasoning + Acting with alternating reasoning and action in flexible conversational format. Combines thinking and acting in natural language flow. Less structured than CoT but incorporates memory and communication history.

    Attributes:
        - **agent** (LLMAgent reference)

    Methods:
        - **plan(prompt, obs=None, ttl=1, selected_tools=None)** → *Plan* - Generate synchronous plan with ReAct reasoning
        - **async aplan(prompt, obs=None, ttl=1, selected_tools=None)** → *Plan* - Generate asynchronous plan with ReAct reasoning
    """

    def __init__(self, agent: "LLMAgent"):
        super().__init__(agent=agent)

    def get_react_system_prompt(self) -> str:
        system_prompt = """
        You are an autonomous agent in a simulation environment.
        You can think about your situation and describe your plan.
        Use your short-term and/or long-term memory to guide your behavior.
        You should also use the current observation you have made of the environrment to take suitable actions.

        # Instructions
        Based on the information given to you, think about what you should do with proper reasoning, And then decide your plan of action. Respond in the
        following format:
        reasoning: [Your reasoning about the situation, including how your memory informs your decision]
        action: [The action you decide to take - Do NOT use any tools here, just describe the action you will take]

        """
        return system_prompt

    def get_react_prompt(self, obs: Observation) -> list[str]:
        prompt_list = self.agent.memory.get_prompt_ready()
        last_communication = self.agent.memory.get_communication_history()

        if last_communication:
            prompt_list.append("last communication: \n" + str(last_communication))
        if obs:
            prompt_list.append("current observation: \n" + str(obs))

        return prompt_list

    def plan(
        self,
        obs: Observation,
        ttl: int = 1,
        prompt: str | None = None,
        selected_tools: list[str] | None = None,
    ) -> Plan:
        """
        Plan the next (ReAct) action based on the current observation and the agent's memory.
        """

        # ---------------- prepare the prompt ----------------
        self.agent.llm.system_prompt = self.get_react_system_prompt()
        prompt_list = self.get_react_prompt(obs)

        # If no prompt is provided, use the agent's default step prompt
        if prompt is None:
            if self.agent.step_prompt is not None:
                prompt_list.append(self.agent.step_prompt)
            else:
                raise ValueError("No prompt provided and agent.step_prompt is None.")

        selected_tools_schema = self.agent.tool_manager.get_all_tools_schema(
            selected_tools
        )

        # ---------------- generate the plan ----------------
        rsp = self.agent.llm.generate(
            prompt=prompt_list,
            tool_schema=selected_tools_schema,
            tool_choice="none",
            response_format=ReActOutput,
        )

        formatted_response = json.loads(rsp.choices[0].message.content)

        self.agent.memory.add_to_memory(type="plan", content=formatted_response)

        # ---------------- execute the plan ----------------
        react_plan = self.execute_tool_call(
            formatted_response["action"], selected_tools
        )

        return react_plan

    async def aplan(
        self,
        obs: Observation,
        ttl: int = 1,
        prompt: str | None = None,
        selected_tools: list[str] | None = None,
    ) -> Plan:
        """
        Asynchronous version of plan() method for parallel planning.
        """

        # ---------------- prepare the prompt ----------------
        self.agent.llm.system_prompt = self.get_react_system_prompt()
        prompt_list = self.get_react_prompt(obs)

        # If no prompt is provided, use the agent's default step prompt
        if prompt is None:
            if self.agent.step_prompt is not None:
                prompt_list.append(self.agent.step_prompt)
            else:
                raise ValueError("No prompt provided and agent.step_prompt is None.")

        selected_tools_schema = self.agent.tool_manager.get_all_tools_schema(
            selected_tools
        )

        # ---------------- generate the plan ----------------

        rsp = await self.agent.llm.agenerate(
            prompt=prompt_list,
            tool_schema=selected_tools_schema,
            tool_choice="none",
            response_format=ReActOutput,
        )

        formatted_response = json.loads(rsp.choices[0].message.content)

        self.agent.memory.add_to_memory(type="plan", content=formatted_response)

        # ---------------- execute the plan ----------------
        react_plan = await self.aexecute_tool_call(
            formatted_response["action"], selected_tools
        )

        return react_plan
