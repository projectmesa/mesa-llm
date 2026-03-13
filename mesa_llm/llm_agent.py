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
import math

class LLMAgent(Agent):
    """
    LLMAgent manages an LLM backend and optionally connects to a memory module.

    Parameters:
        model (Model): The mesa model the agent in linked to.
        llm_model (str): The model to use for the LLM in the format 'provider/model'. Defaults to 'gemini/gemini-2.0-flash'.
        system_prompt (str | None): Optional system prompt to be used in LLM completions.
        reasoning (str): Optional reasoning method to be used in LLM completions.

    Attributes:
        llm (ModuleLLM): The internal LLM interface used by the agent.
        memory (Memory | None): The memory module attached to this agent, if any.

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
    ):
        super().__init__(model=model)

        self.model = model
        self.step_prompt = step_prompt
        self.llm = ModuleLLM(llm_model=llm_model, system_prompt=system_prompt)

        self.memory = STLTMemory(
            agent=self,
            short_term_capacity=5,
            consolidation_capacity=2,
            llm_model=llm_model,
        )

        self.tool_manager = ToolManager()
        self.vision = vision
        self.reasoning_instance = reasoning(agent=self)
        self.reasoning = reasoning
        self.system_prompt = system_prompt
        self.llm_model = llm_model
        self.is_speaking = False
        self._current_plan = None

        self._step_display_data = {}

        if isinstance(internal_state, str):
            internal_state = [internal_state]
        elif internal_state is None:
            internal_state = []

        self.internal_state = internal_state

    def __str__(self) -> str:
        return f"LLMAgent {self.unique_id}"

    def __repr__(self) -> str:
        memory_size = (
            len(self.memory.short_term_memory)
            if hasattr(self.memory, "short_term_memory")
            else 0
        )
        return (
            f"LLMAgent("
            f"unique_id={self.unique_id}, "
            f"llm_model='{self.llm.llm_model}', "
            f"reasoning={self.reasoning.__name__}, "
            f"vision={self.vision}, "
            f"memory_size={memory_size}, "
            f"internal_state={self.internal_state}"
            f")"
        )

    async def aapply_plan(self, plan: Plan) -> list[dict]:
        """
        Asynchronous version of apply_plan.
        """
        self._current_plan = plan

        tool_call_resp = await self.tool_manager.acall_tools(
            agent=self, llm_response=plan.llm_plan
        )

        await self.memory.aadd_to_memory(
            type="action",
            content={
                k: v
                for tool_call in tool_call_resp
                for k, v in tool_call.items()
                if k not in ["tool_call_id", "role"]
            },
        )

        return tool_call_resp

    def apply_plan(self, plan: Plan) -> list[dict]:
        """
        Execute the plan in the simulation.
        """
        self._current_plan = plan

        tool_call_resp = self.tool_manager.call_tools(
            agent=self, llm_response=plan.llm_plan
        )

        self.memory.add_to_memory(
            type="action",
            content={
                k: v
                for tool_call in tool_call_resp
                for k, v in tool_call.items()
                if k not in ["tool_call_id", "role"]
            },
        )

        return tool_call_resp

    def _build_observation(self):
        """
        Construct the observation data visible to the agent at the current model step.
        """
        self_state = {
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
        if self.vision is not None and self.vision > 0:
            grid = getattr(self.model, "grid", None)
            space = getattr(self.model, "space", None)

            if grid and isinstance(
                grid, OrthogonalMooreGrid | OrthogonalVonNeumannGrid
            ):
                agent_cell = next(
                    (cell for cell in grid.all_cells if self in cell.agents),
                    None,
                )
                if agent_cell:
                    neighborhood = agent_cell.get_neighborhood(radius=self.vision)
                    neighbors = [a for cell in neighborhood for a in cell.agents]
                else:
                    neighbors = []

            elif space and isinstance(space, ContinuousSpace):
                

                my_pos = getattr(self, "pos", None)
                if my_pos is not None:
                    neighbors = [
                        a
                        for a in self.model.agents
                        if a is not self
                        and getattr(a, "pos", None) is not None
                        and math.dist(my_pos, a.pos) <= self.vision
                    ]
                else:
                    neighbors = []

            else:
                neighbors = []

        elif self.vision == -1:
            all_agents = list(self.model.agents)
            neighbors = [agent for agent in all_agents if agent is not self]

        else:
            neighbors = []

        local_state = {}
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
        Async observation generation.
        """
        step = int(self.model._time) if hasattr(self.model, "_time") else 0
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
        Sync observation generation.
        """
        step = int(self.model._time) if hasattr(self.model, "_time") else 0
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

    async def apre_step(self):
        await self.memory.aprocess_step(pre_step=True)

    async def apost_step(self):
        await self.memory.aprocess_step()

    def pre_step(self):
        self.memory.process_step(pre_step=True)

    def post_step(self):
        self.memory.process_step()

    async def astep(self):
        """
        Default asynchronous step method for parallel agent execution.
        """
        await self.apre_step()

        if hasattr(self, "step") and self.__class__.step != LLMAgent.step:
            self.step()

        await self.apost_step()

    def __init_subclass__(cls, **kwargs):
        """
        Wrapper - allows to automatically integrate code to be executed after the step method.
        """
        super().__init_subclass__(**kwargs)
        user_step = cls.__dict__.get("step")
        user_astep = cls.__dict__.get("astep")

        if user_step:

            def wrapped(self, *args, **kwargs):
                LLMAgent.pre_step(self, *args, **kwargs)
                result = user_step(self, *args, **kwargs)
                LLMAgent.post_step(self, *args, **kwargs)
                return result

            cls.step = wrapped

        if user_astep:

            async def awrapped(self, *args, **kwargs):
                await self.apre_step()
                result = await user_astep(self, *args, **kwargs)
                await self.apost_step()
                return result

            cls.astep = awrapped
