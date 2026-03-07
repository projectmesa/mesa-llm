import math
from typing import TYPE_CHECKING, Any

from mesa.agent import Agent
from mesa.discrete_space import (
    OrthogonalMooreGrid,
    OrthogonalVonNeumannGrid,
)
from mesa.experimental.continuous_space import ContinuousSpace
from mesa.model import Model

from mesa_llm import Plan
from mesa_llm.memory.st_lt_memory import STLTMemory
from mesa_llm.module_llm import ModuleLLM
from mesa_llm.reasoning.reasoning import (
    Observation,
    Reasoning,
)
from mesa_llm.tools.tool_manager import ToolManager

if TYPE_CHECKING:
    pass


class LLMAgent(Agent):
    """
    LLMAgent manages an LLM backend and optionally connects to a memory module.

    Parameters:
        model (Model): The mesa model the agent is linked to.
        reasoning (type[Reasoning]): The reasoning class to use for planning.
        llm_model (str): The model to use for the LLM in the format 'provider/model'.
            Defaults to 'gemini/gemini-2.0-flash'.
        system_prompt (str | None): Optional system prompt for LLM completions.
        vision (float | None): Radius within which the agent observes neighbors.
            Use -1 to observe all agents, None or 0 for no observation.
        internal_state (list[str] | str | None): Initial internal state attributes.
        step_prompt (str | None): Optional prompt passed to the memory module.

    Attributes:
        llm (ModuleLLM): The internal LLM interface used by the agent.
        memory (STLTMemory): The memory module attached to this agent.
        tool_manager (ToolManager): Manages available tools for the agent.
        reasoning (Reasoning): The reasoning instance used for planning.
    """

    def __init__(
        self,
        model: Model,
        reasoning: type[Reasoning],
        llm_model: str = "gemini/gemini-2.0-flash",
        system_prompt: str | None = None,
        vision: float | None = None,
        internal_state: list[str] | str | None = None,
        step_prompt: str | None = None,
    ) -> None:
        super().__init__(model=model)

        self.model: Model = model
        self.step_prompt: str | None = step_prompt
        self.llm: ModuleLLM = ModuleLLM(llm_model=llm_model, system_prompt=system_prompt)

        self.memory: STLTMemory = STLTMemory(
            agent=self,
            short_term_capacity=5,
            consolidation_capacity=2,
            llm_model=llm_model,
        )

        self.tool_manager: ToolManager = ToolManager()
        self.vision: float | None = vision
        self.reasoning: Reasoning = reasoning(agent=self)
        self.system_prompt: str | None = system_prompt
        self.is_speaking: bool = False
        self._current_plan: Plan | None = None

        self._step_display_data: dict[str, Any] = {}

        if isinstance(internal_state, str):
            internal_state = [internal_state]
        elif internal_state is None:
            internal_state = []

        self.internal_state: list[str] = internal_state

    def __str__(self) -> str:
        return f"LLMAgent {self.unique_id}"

    async def aapply_plan(self, plan: Plan) -> list[dict[str, Any]]:
        """
        Asynchronous version of apply_plan.

        Args:
            plan (Plan): The plan to execute.

        Returns:
            list[dict[str, Any]]: List of tool call responses.
        """
        self._current_plan = plan

        tool_call_resp = await self.tool_manager.acall_tools(
            agent=self, llm_response=plan.llm_plan
        )

        await self.memory.aadd_to_memory(
            type="action",
            content={
                "tool_calls": [
                    {k: v for k, v in tc.items() if k not in ["tool_call_id", "role"]}
                    for tc in tool_call_resp
                ]
            },
        )

        return tool_call_resp

    def apply_plan(self, plan: Plan) -> list[dict[str, Any]]:
        """
        Execute the plan in the simulation.

        Args:
            plan (Plan): The plan to execute.

        Returns:
            list[dict[str, Any]]: List of tool call responses.
        """
        self._current_plan = plan

        tool_call_resp = self.tool_manager.call_tools(
            agent=self, llm_response=plan.llm_plan
        )

        self.memory.add_to_memory(
            type="action",
            content={
                "tool_calls": [
                    {k: v for k, v in tc.items() if k not in ["tool_call_id", "role"]}
                    for tc in tool_call_resp
                ]
            },
        )

        return tool_call_resp

    def _build_observation(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        Construct the observation data visible to the agent at the current model step.

        This method encapsulates the shared logic used by both sync and
        async observation generation. It constructs the agent's self state
        and determines which other agents are observable based on the
        configured vision:

        - vision > 0: The agent observes all agents within the specified vision radius.
        - vision == -1: The agent observes all agents present in the simulation.
        - vision == 0 or vision is None: The agent observes no other agents.

        The method supports grid-based and continuous spaces and builds a local
        state representation for all visible neighboring agents.

        Returns:
            tuple[dict[str, Any], dict[str, Any]]: A tuple of (self_state, local_state).
        """
        self_state: dict[str, Any] = {
            "agent_unique_id": self.unique_id,
            "system_prompt": self.system_prompt,
            "location": (
                getattr(self, "pos", None)
                if getattr(self, "pos", None) is not None
                else (
                    getattr(self, "cell", None).coordinate
                    if getattr(self, "cell", None) is not None
                    else None
                )
            ),
            "internal_state": self.internal_state,
        }

        neighbors: list[Agent] = []

        if self.vision is not None and self.vision > 0:
            grid = getattr(self.model, "grid", None)
            space = getattr(self.model, "space", None)

            if grid and isinstance(
                grid, OrthogonalMooreGrid | OrthogonalVonNeumannGrid
            ):
                agent_cell = getattr(self, "cell", None)
                if agent_cell:
                    neighborhood = agent_cell.get_neighborhood(radius=self.vision)
                    neighbors = [
                        a
                        for cell in neighborhood
                        for a in list(cell.agents)
                        if a is not self
                    ]

            elif space and isinstance(space, ContinuousSpace):
                neighbors = [
                    a
                    for a in self.model.agents
                    if a is not self
                    and getattr(a, "pos", None) is not None
                    and getattr(self, "pos", None) is not None
                    and math.dist(self.pos, a.pos) <= self.vision
                ]

        elif self.vision == -1:
            neighbors = [a for a in self.model.agents if a is not self]

        local_state: dict[str, Any] = {}
        for i in neighbors:
            local_state[i.__class__.__name__ + " " + str(i.unique_id)] = {
                "position": (
                    i.pos
                    if i.pos is not None
                    else (
                        getattr(i, "cell", None).coordinate
                        if getattr(i, "cell", None) is not None
                        else None
                    )
                ),
                "internal_state": [
                    s for s in getattr(i, "internal_state", []) if not s.startswith("_")
                ],
            }
        return self_state, local_state

    async def agenerate_obs(self) -> Observation:
        """
        Async version of generate_obs.

        Builds the agent's observation, stores it in memory asynchronously,
        and returns it as an Observation instance.

        Returns:
            Observation: The current observation for this agent.
        """
        step: int = self.model.step
        self_state, local_state = self._build_observation()
        await self.memory.aadd_to_memory(
            type="observation",
            content={
                "self_state": self_state,
                "local_state": local_state,
            },
        )

        return Observation(step=step, self_state=self_state, local_state=local_state)

    def generate_obs(self) -> Observation:
        """
        Build the agent's current observation, store it in memory, and return it.

        Returns:
            Observation: The current observation for this agent.
        """
        step: int = self.model.step
        self_state, local_state = self._build_observation()
        self.memory.add_to_memory(
            type="observation",
            content={
                "self_state": self_state,
                "local_state": local_state,
            },
        )

        return Observation(step=step, self_state=self_state, local_state=local_state)

    async def asend_message(self, message: str, recipients: list[Agent]) -> str:
        """
        Asynchronous version of send_message.

        Args:
            message (str): The message content to send.
            recipients (list[Agent]): List of agents to receive the message.

        Returns:
            str: A formatted string describing the message exchange.
        """
        for recipient in [*recipients, self]:
            await recipient.memory.aadd_to_memory(
                type="message",
                content={
                    "message": message,
                    "sender": self.unique_id,
                    "recipients": [r.unique_id for r in recipients],
                },
            )

        return f"{self} → {recipients} : {message}"

    def send_message(self, message: str, recipients: list[Agent]) -> str:
        """
        Send a message to the recipients.

        Args:
            message (str): The message content to send.
            recipients (list[Agent]): List of agents to receive the message.

        Returns:
            str: A formatted string describing the message exchange.
        """
        for recipient in [*recipients, self]:
            recipient.memory.add_to_memory(
                type="message",
                content={
                    "message": message,
                    "sender": self.unique_id,
                    "recipients": [r.unique_id for r in recipients],
                },
            )

        return f"{self} → {recipients} : {message}"

    async def apre_step(self) -> None:
        """Asynchronous version of pre_step."""
        await self.memory.aprocess_step(pre_step=True)

    async def apost_step(self) -> None:
        """Asynchronous version of post_step."""
        await self.memory.aprocess_step()

    def pre_step(self) -> None:
        """Execute code before the child agent's step method is called."""
        self.memory.process_step(pre_step=True)

    def post_step(self) -> None:
        """
        Execute code after the child agent's step method is called.

        This works via the __init_subclass__ wrapper that wraps the step
        method of child agents automatically.
        """
        self.memory.process_step()

    async def astep(self) -> None:
        """
        Default asynchronous step method for parallel agent execution.

        Subclasses should override this method for custom async behavior.
        If not overridden, falls back to calling the synchronous step() method.
        """
        await self.apre_step()

        if hasattr(self, "step") and self.__class__.step != LLMAgent.step:
            self.step()

        await self.apost_step()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """
        Automatically wrap the step and astep methods of subclasses to integrate
        pre_step and post_step hooks.
        """
        super().__init_subclass__(**kwargs)
        user_step = cls.__dict__.get("step")
        user_astep = cls.__dict__.get("astep")

        if user_step:

            def wrapped(self, *args: Any, **kwargs: Any) -> Any:
                """Wrapper integrating pre_step and post_step into the child step."""
                LLMAgent.pre_step(self, *args, **kwargs)
                result = user_step(self, *args, **kwargs)
                LLMAgent.post_step(self, *args, **kwargs)
                return result

            cls.step = wrapped

        if user_astep:

            async def awrapped(self, *args: Any, **kwargs: Any) -> Any:
                """Async wrapper integrating pre_step and post_step into astep."""
                await self.apre_step()
                result = await user_astep(self, *args, **kwargs)
                await self.apost_step()
                return result

            cls.astep = awrapped
