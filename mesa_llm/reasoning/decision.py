from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from mesa_llm.reasoning.reasoning import Observation, Plan, Reasoning

if TYPE_CHECKING:
    from mesa_llm.llm_agent import LLMAgent


class DecisionOption(BaseModel):
    name: str
    description: str
    tradeoffs: list[str]
    score: float = Field(
        description="Relative evaluation score for this option in the current context."
    )


class DecisionOutput(BaseModel):
    goal: str
    constraints: list[str]
    known_facts: list[str]
    unknowns: list[str]
    assumptions: list[str]
    options: list[DecisionOption]
    chosen_option: str
    rationale: str
    confidence: float = Field(ge=0.0, le=1.0)
    risks: list[str]
    next_action: str


class DecisionReasoning(Reasoning):
    """
    Structured decision-making reasoning that returns a strict JSON object before
    converting the selected next action into tool calls.
    """

    def __init__(self, agent: "LLMAgent"):
        super().__init__(agent=agent)

    def get_decision_system_prompt(self) -> str:
        return """
        You are an autonomous agent operating within a simulation environment.

        Your task is to analyze your current observation and memory to make a highly structured, optimal decision.
        Do not produce free-form chain-of-thought prose. You must evaluate the situation and return a strict JSON object matching the required schema.

        Your response must include:
        - goal: Your current primary objective within the simulation.
        - constraints: Any rules, resource limits, or environmental boundaries restricting your actions.
        - known_facts: Verified data strictly grounded in your current observation or historical memory.
        - unknowns: Critical missing information required for perfect decision-making.
        - assumptions: Logical inferences made to bridge the gap between known facts and unknowns.
        - options: A list of distinct, executable choices currently available to you. Each must include a name, description, tradeoffs, and a relative evaluation score.
        - chosen_option: The exact name of the best option selected from the list above.
        - rationale: A concise, logical justification for why this option was chosen over the alternatives.
        - confidence: A float between 0.0 and 1.0 representing your certainty in this decision.
        - risks: Potential negative outcomes or failure states associated with the chosen option.
        - next_action: A single, concrete, and strictly formatted executable command.

        Execution Requirements:
        1. Ground all known_facts entirely in the provided observation context. Do not hallucinate simulation state or capabilities.
        2. next_action must strictly match an available execution command. Do not invent tools.
        3. If information is heavily constrained or missing, explicitly reflect this by lowering the confidence score and detailing the danger in risks.
        """

    def get_decision_prompt(self, obs: Observation) -> list[str]:
        prompt_list = []

        get_prompt_ready = getattr(self.agent.memory, "get_prompt_ready", None)
        if callable(get_prompt_ready):
            prompt_list.append(get_prompt_ready())

        get_communication_history = getattr(
            self.agent.memory, "get_communication_history", None
        )
        last_communication = (
            get_communication_history() if callable(get_communication_history) else ""
        )

        if last_communication:
            prompt_list.append("last communication: \n" + str(last_communication))
        if obs:
            prompt_list.append("current observation: \n" + str(obs))

        return prompt_list

    def _parse_decision_output(self, response) -> dict:
        """Normalize structured LLM output into a dictionary."""
        message = response.choices[0].message
        parsed = getattr(message, "parsed", None)

        if isinstance(parsed, DecisionOutput):
            return parsed.model_dump()

        if isinstance(parsed, dict):
            return DecisionOutput.model_validate(parsed).model_dump()

        return DecisionOutput.model_validate_json(message.content).model_dump()

    def plan(
        self,
        prompt: str | None = None,
        obs: Observation | None = None,
        ttl: int = 1,
        selected_tools: list[str] | None = None,
    ) -> Plan:
        """
        Plan the next action through a structured decision artifact.
        """
        if obs is None:
            obs = self.agent.generate_obs()

        self.agent.llm.system_prompt = self.get_decision_system_prompt()
        prompt_list = self.get_decision_prompt(obs)

        if prompt is not None:
            prompt_list.append(prompt)
        elif self.agent.step_prompt is not None:
            prompt_list.append(self.agent.step_prompt)
        else:
            raise ValueError("No prompt provided and agent.step_prompt is None.")

        selected_tools_schema = self.agent.tool_manager.get_all_tools_schema(
            selected_tools
        )

        rsp = self.agent.llm.generate(
            prompt=prompt_list,
            tool_schema=selected_tools_schema,
            tool_choice="none",
            response_format=DecisionOutput,
        )

        formatted_response = self._parse_decision_output(rsp)
        self.agent.memory.add_to_memory(type="decision", content=formatted_response)

        if hasattr(self.agent, "_step_display_data"):
            self.agent._step_display_data["plan_content"] = formatted_response[
                "rationale"
            ]

        return self.execute_tool_call(
            formatted_response["next_action"],
            selected_tools=selected_tools,
            ttl=ttl,
        )

    async def aplan(
        self,
        prompt: str | None = None,
        obs: Observation | None = None,
        ttl: int = 1,
        selected_tools: list[str] | None = None,
    ) -> Plan:
        """
        Asynchronous version of plan() method for parallel planning.
        """
        if obs is None:
            obs = await self.agent.agenerate_obs()

        self.agent.llm.system_prompt = self.get_decision_system_prompt()
        prompt_list = self.get_decision_prompt(obs)

        if prompt is not None:
            prompt_list.append(prompt)
        elif self.agent.step_prompt is not None:
            prompt_list.append(self.agent.step_prompt)
        else:
            raise ValueError("No prompt provided and agent.step_prompt is None.")

        selected_tools_schema = self.agent.tool_manager.get_all_tools_schema(
            selected_tools
        )

        rsp = await self.agent.llm.agenerate(
            prompt=prompt_list,
            tool_schema=selected_tools_schema,
            tool_choice="none",
            response_format=DecisionOutput,
        )

        formatted_response = self._parse_decision_output(rsp)
        await self.agent.memory.aadd_to_memory(
            type="decision", content=formatted_response
        )

        if hasattr(self.agent, "_step_display_data"):
            self.agent._step_display_data["plan_content"] = formatted_response[
                "rationale"
            ]

        return await self.aexecute_tool_call(
            formatted_response["next_action"],
            selected_tools=selected_tools,
            ttl=ttl,
        )
