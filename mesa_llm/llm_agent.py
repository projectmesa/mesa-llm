from mesa.agent import Agent
from mesa.discrete_space import (
    OrthogonalMooreGrid,
    OrthogonalVonNeumannGrid,
)
from mesa.model import Model
from mesa.space import (
    ContinuousSpace,
    MultiGrid,
    SingleGrid,
)

from mesa_llm import Plan
from mesa_llm.memory.st_lt_memory import STLTMemory
from mesa_llm.module_llm import ModuleLLM
from mesa_llm.reasoning.reasoning import (
    Observation,
    Reasoning,
)
from mesa_llm.tools.tool_manager import ToolManager


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
        self.reasoning = reasoning(agent=self)
        self.system_prompt = system_prompt
        self.is_speaking = False
        self._current_plan = None  # Store current plan for formatting

        # display coordination
        self._step_display_data = {}

        if isinstance(internal_state, str):
            internal_state = [internal_state]
        elif internal_state is None:
            internal_state = []

        self.internal_state = internal_state

    def __str__(self):
        return f"LLMAgent {self.unique_id}"

    def apply_plan(self, plan: Plan) -> list[dict]:
        """
        Execute the plan in the simulation.
        """
        # Store current plan for display
        self._current_plan = plan

        # Execute tool calls
        tool_call_resp = self.tool_manager.call_tools(
            agent=self, llm_response=plan.llm_plan
        )

        # Add to memory
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

    def generate_obs(self) -> Observation:
        """
        Returns an instance of the Observation dataclass enlisting everything the agent can see in the model in that step.

        If the agents vision is set to anything above 0, the agent will get the details of all agents falling in that radius.
        If the agents vision is set to -1, then the agent will get the details of all the agents present in the simulation at that step.
        If it is set to 0 or None, then no information is returned to the agent.

        """
        step = self.model.steps

        self_state = {
            "agent_unique_id": self.unique_id,
            "system_prompt": self.system_prompt,
            "location": self.pos if self.pos is not None else self.cell.coordinate,
            "internal_state": self.internal_state,
        }
        if self.vision is not None and self.vision > 0:
            if isinstance(self.model.grid, SingleGrid | MultiGrid):
                neighbors = self.model.grid.get_neighbors(
                    tuple(self.pos), moore=True, include_center=False, radius=1
                )
            elif isinstance(
                self.model.grid, OrthogonalMooreGrid | OrthogonalVonNeumannGrid
            ):
                neighbors = []
                for neighbor in self.cell.connections.values():
                    neighbors.extend(neighbor.agents)

            elif isinstance(self.model.space, ContinuousSpace):
                neighbors, _ = self.get_neighbors_in_radius(radius=self.vision)

        elif self.vision == -1:
            all_agents = list(self.model.agents)
            neighbors = [agent for agent in all_agents if agent is not self]

        else:
            neighbors = []

        local_state = {}
        for i in neighbors:
            local_state[i.__class__.__name__ + " " + str(i.unique_id)] = {
                "position": i.pos if i.pos is not None else i.cell.coordinate,
                "internal_state": [
                    s for s in i.internal_state if not s.startswith("_")
                ],
            }

        # Add to memory (memory handles its own display separately)
        self.memory.add_to_memory(
            type="observation",
            content={
                "self_state": self_state,
                "local_state": local_state,
            },
        )

        return Observation(step=step, self_state=self_state, local_state=local_state)

    def send_message(self, message: str, recipients: list[Agent]) -> str:
        """
        Send a message to the recipients.
        """
        for recipient in [*recipients, self]:
            recipient.memory.add_to_memory(
                type="message",
                content={
                    "message": message,
                    "sender": self,
                    "recipients": recipients,
                },
            )

        return f"{self} → {recipients} : {message}"

    def pre_step(self):
        """
        This is some code that is executed before the step method of the child agent is called.
        """
        self.memory.process_step(pre_step=True)

    def post_step(self):
        """
        This is some code that is executed after the step method of the child agent is called.
        It functions because of the __init_subclass__ method that creates a wrapper around the step method of the child agent.
        """
        self.memory.process_step()

    async def astep(self):
        """
        Default asynchronous step method for parallel agent execution.
        Subclasses should override this method for custom async behavior.
        If not overridden, falls back to calling the synchronous step() method.
        """
        self.pre_step()

        if hasattr(self, "step") and self.__class__.step != LLMAgent.step:
            self.step()

        self.post_step()

    def __init_subclass__(cls, **kwargs):
        """
        Wrapper - allows to automatically integrate code to be executed after the step method of the child agent (created by the user) is called.
        """
        super().__init_subclass__(**kwargs)
        # only wrap if subclass actually defines its own step
        user_step = cls.__dict__.get("step")
        user_astep = cls.__dict__.get("astep")

        if user_step:

            def wrapped(self, *args, **kwargs):
                """
                This is the wrapper that is used to integrate the pre_step and post_step methods into the step method of the child agent.
                """
                LLMAgent.pre_step(self, *args, **kwargs)
                result = user_step(self, *args, **kwargs)
                LLMAgent.post_step(self, *args, **kwargs)
                return result

            cls.step = wrapped

        if user_astep:

            async def awrapped(self, *args, **kwargs):
                """
                Async wrapper for astep method.
                """
                self.pre_step()
                result = await user_astep(self, *args, **kwargs)
                self.post_step()
                return result

            cls.astep = awrapped
