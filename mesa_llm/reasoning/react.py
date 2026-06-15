import json
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from mesa_llm.reasoning.reasoning import (
    _UNSET,
    Observation,
    Plan,
    Reasoning,
    ToolSelection,
)

if TYPE_CHECKING:
    from mesa_llm.llm_agent import LLMAgent


class ReActOutput(BaseModel):
    reasoning: str = Field(
        description="Step-by-step reasoning about the situation based on memory and observation"
    )
    action: str = Field(description="The specific action to take without using tools")


class ReActReasoning(Reasoning):
    """
    Reasoning + Acting with alternating reasoning and action in flexible conversational format. Combines thinking and acting in natural language flow. Less structured than CoT but incorporates memory and communication history.

    Attributes:
        - **agent** (LLMAgent reference)

    Methods:
        - **plan(prompt, obs=None, ttl=1, tools=<inherit>, tool_calls="auto")** → *Plan* - Generate synchronous plan with ReAct reasoning
        - **async aplan(prompt, obs=None, ttl=1, tools=<inherit>, tool_calls="auto")** → *Plan* - Generate asynchronous plan with ReAct reasoning
    """

    def __init__(self, agent: "LLMAgent"):
        super().__init__(agent=agent)

    def get_react_system_prompt(self) -> str:
        agent_persona = getattr(self.agent, "system_prompt", None)
        persona_section = ""
        if isinstance(agent_persona, str) and agent_persona.strip():
            persona_section = (
                f"\n        # Agent Persona\n        {agent_persona.strip()}\n"
            )
        system_prompt = f"""
        You are an autonomous agent in a simulation environment.
        You can think about your situation and describe your plan.
        Use your short-term and/or long-term memory to guide your behavior.
        You should also use the current observation you have made of the environrment to take suitable actions.
{persona_section}

        # Instructions
        Based on the information given to you, think about what you should do with proper reasoning, And then decide your plan of action. Respond in the
        following format:
        reasoning: [Your reasoning about the situation, including how your memory informs your decision]
        action: [The action you decide to take - Do NOT use any tools here, just describe the action you will take]

        """
        return system_prompt

    def get_react_prompt(self, obs: Observation) -> list[str]:
        prompt_list = [self.agent.memory.get_prompt_ready()]
        last_communication = self.agent.memory.get_communication_history()

        if last_communication:
            prompt_list.append("last communication: \n" + str(last_communication))
        if obs:
            prompt_list.append("current observation: \n" + str(obs))

        return prompt_list

    def plan(
        self,
        prompt: str | None = None,
        obs: Observation | None = None,
        ttl: int = 1,
        tools: ToolSelection | object = _UNSET,
        tool_calls: str | None = "auto",
        selected_tools: ToolSelection | object = _UNSET,
    ) -> Plan:
        """
        Plan the next (ReAct) action based on the current observation and the
        agent's memory.

        ``tools`` controls provider tool exposure. Omitting it inherits the
        agent's configured tools. Explicit ``None`` or ``[]`` exposes no tools.
        A callable, string name, or sequence exposes exactly those configured
        tools.

        ``tool_calls`` controls the execution-phase LiteLLM ``tool_choice``.
        The reasoning pass still keeps tool use disabled with ``"none"``.

        Supported values in Mesa-LLM are:
        - ``None``: defer to LiteLLM/provider default behavior. In practice,
          this usually means no tool calls when no tools are provided and
          behavior similar to ``"auto"`` when tools are available.
        - ``"none"``: never return tool calls; return a normal assistant
          message instead.
        - ``"auto"``: allow the model to either return a normal assistant
          message or call one or more tools.
        - ``"required"``: require the model to call one or more tools.
        """

        if obs is None:
            obs = self.agent.generate_obs()

        # ---------------- prepare the prompt ----------------
        react_system_prompt = self.get_react_system_prompt()
        prompt_list = self.get_react_prompt(obs)

        # Add user prompt (explicit prompt takes precedence over default step prompt)
        if prompt is not None:
            prompt_list.append(prompt)
        elif self.agent.step_prompt is not None:
            prompt_list.append(self.agent.step_prompt)
        else:
            raise ValueError("No prompt provided and agent.step_prompt is None.")

        tools = self._resolve_tools_argument(tools, selected_tools)
        tools_schema = self._get_tools_schema(tools)

        # ---------------- generate the plan ----------------
        rsp = self.agent.llm.generate(
            prompt=prompt_list,
            tool_schema=tools_schema,
            tool_choice="none",
            response_format=ReActOutput,
            system_prompt=react_system_prompt,
        )

        formatted_response = json.loads(rsp.choices[0].message.content)

        self.agent.memory.add_to_memory(type="plan", content=formatted_response)

        # ---------------- execute the plan ----------------
        execute_kwargs = {"ttl": ttl, "tool_calls": tool_calls}
        if tools is not _UNSET:
            execute_kwargs["tools"] = tools
        react_plan = self.execute_tool_call(
            formatted_response["action"], **execute_kwargs
        )

        return react_plan

    async def aplan(
        self,
        prompt: str | None = None,
        obs: Observation | None = None,
        ttl: int = 1,
        tools: ToolSelection | object = _UNSET,
        tool_calls: str | None = "auto",
        selected_tools: ToolSelection | object = _UNSET,
    ) -> Plan:
        """
        Asynchronous version of plan() method for parallel planning.

        ``tools`` follows the same contract as ``plan()``.

        ``tool_calls`` controls the execution-phase LiteLLM ``tool_choice``.
        The reasoning pass still keeps tool use disabled with ``"none"``.

        Supported values in Mesa-LLM are:
        - ``None``: defer to LiteLLM/provider default behavior. In practice,
          this usually means no tool calls when no tools are provided and
          behavior similar to ``"auto"`` when tools are available.
        - ``"none"``: never return tool calls; return a normal assistant
          message instead.
        - ``"auto"``: allow the model to either return a normal assistant
          message or call one or more tools.
        - ``"required"``: require the model to call one or more tools.
        """
        if obs is None:
            obs = await self.agent.agenerate_obs()

        # ---------------- prepare the prompt ----------------
        react_system_prompt = self.get_react_system_prompt()
        prompt_list = self.get_react_prompt(obs)

        # Add user prompt (explicit prompt takes precedence over default step prompt)
        if prompt is not None:
            prompt_list.append(prompt)
        elif self.agent.step_prompt is not None:
            prompt_list.append(self.agent.step_prompt)
        else:
            raise ValueError("No prompt provided and agent.step_prompt is None.")

        tools = self._resolve_tools_argument(tools, selected_tools)
        tools_schema = self._get_tools_schema(tools)

        # ---------------- generate the plan ----------------

        rsp = await self.agent.llm.agenerate(
            prompt=prompt_list,
            tool_schema=tools_schema,
            tool_choice="none",
            response_format=ReActOutput,
            system_prompt=react_system_prompt,
        )

        formatted_response = json.loads(rsp.choices[0].message.content)

        await self.agent.memory.aadd_to_memory(type="plan", content=formatted_response)

        # ---------------- execute the plan ----------------
        execute_kwargs = {"ttl": ttl, "tool_calls": tool_calls}
        if tools is not _UNSET:
            execute_kwargs["tools"] = tools
        react_plan = await self.aexecute_tool_call(
            formatted_response["action"], **execute_kwargs
        )

        return react_plan
