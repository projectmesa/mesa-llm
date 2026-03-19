import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mesa_llm.llm_agent import LLMAgent

logger = logging.getLogger(__name__)


class ReasoningError(Exception):
    """Custom exception for reasoning failures."""

    def __init__(
        self, message: str, reasoning_type: str | None = None, step: int | None = None
    ):
        super().__init__(message)
        self.reasoning_type = reasoning_type
        self.step = step


class PlanExecutionError(Exception):
    """Custom exception for plan execution failures."""

    def __init__(
        self, message: str, plan_step: int | None = None, tool_name: str | None = None
    ):
        super().__init__(message)
        self.plan_step = plan_step
        self.tool_name = tool_name


@dataclass
class Observation:
    """
    A structured snapshot containing the agent's current step, self-state (internal attributes, location, system context), and local-state (neighboring agents and their properties). This provides complete situational awareness for decision-making.

    Attributes:
        step (int): The current simulation time step when the observation is made.

        self_state (dict): A dictionary containing comprehensive information about the observing agent itself.
            This includes:
            - System prompt or role-specific context for LLM reasoning (if used)
            - Internal state such as morale, fear, aggression, fatigue, etc (behavioural).
            - Agent's current location or spatial coordinates
            - Any other agent-specific metadata that could influence decision-making

        local_state (dict): A dictionary summarizing the state of nearby agents (within the vision radius).
            - A dictionary of neighboring agents, where each key is the "angent's class name + id" and the value is a dictionary containing the following:
            - position of neighbors
            - Internal state or attributes of neighboring agents

    """

    step: int
    self_state: dict
    local_state: dict


@dataclass
class Plan:
    """
    An LLM-generated plan containing the step number, complete LLM response with tool calls, and a time-to-live (TTL) indicating how many steps the plan remains valid. Plans encapsulate both reasoning content and executable actions.
    """

    step: int  # step when the plan was generated
    llm_plan: Any  # complete LLM response message object (contains both content and tool_calls)
    ttl: int = 1  # steps until planning again (ReWOO sets >1)

    def __str__(self) -> str:
        # Extract content from the message object for display
        if hasattr(self.llm_plan, "content") and self.llm_plan.content:
            llm_plan_str = str(self.llm_plan.content).strip()
        else:
            llm_plan_str = str(self.llm_plan).strip()
        return f"{llm_plan_str}\n"


class Reasoning(ABC):
    """
    Abstract base class providing the interface for all reasoning strategies, with both synchronous `plan()` and asynchronous `aplan()` methods for parallel execution scenarios.


    Attributes:
        - **agent** (LLMAgent reference)

    Methods:
        - **abstract plan(prompt, obs=None, ttl=1, selected_tools=None)** → *Plan* - Generate synchronous plan
        - **async aplan(prompt, obs=None, ttl=1, selected_tools=None)** → *Plan* - Generate asynchronous plan


    Reasoning Flow:
        1. Agent generates **observation** of current situation through `generate_obs()`
        2. Reasoning strategies access **memory** to inform decisions
        3. Selected reasoning approach processes observation and memory into a structured **plan**
        4. Plans are automatically converted to **tool schemas** for LLM function calling
        5. Tool manager **executes the planned actions** in the simulation environment
    """

    def __init__(self, agent: "LLMAgent"):
        self.agent = agent

    @abstractmethod
    def plan(
        self,
        prompt: str | None = None,
        obs: Observation | None = None,
        ttl: int = 1,
        selected_tools: list[str] | None = None,
    ) -> Plan:
        pass

    async def aplan(
        self,
        prompt: str | None = None,
        obs: Observation | None = None,
        ttl: int = 1,
        selected_tools: list[str] | None = None,
    ) -> Plan:
        """
        Asynchronous version of plan() method for parallel planning.
        Default implementation calls the synchronous plan() method.
        """
        return self.plan(
            prompt=prompt,
            obs=obs,
            ttl=ttl,
            selected_tools=selected_tools,
        )

    def _validate_tool_call_inputs(self, chaining_message, ttl: int) -> None:
        """Validate inputs shared by both sync and async tool-call paths."""
        if not chaining_message or not isinstance(chaining_message, str):
            raise ReasoningError(
                "Invalid chaining message: must be a non-empty string",
                reasoning_type="tool_execution",
                step=getattr(self.agent.model, "steps", 0),
            )
        if ttl <= 0:
            raise ReasoningError(
                f"Invalid TTL value: {ttl}. Must be positive.",
                reasoning_type="tool_execution",
                step=getattr(self.agent.model, "steps", 0),
            )

    def _build_plan_from_response(self, rsp, ttl: int) -> "Plan":
        """Validate an LLM response and extract a Plan.  Raises PlanExecutionError on failure."""
        current_step = getattr(self.agent.model, "steps", 0)
        # Validate response structure
        if not hasattr(rsp, "choices") or not rsp.choices:
            raise PlanExecutionError(
                "Invalid LLM response: no choices found",
                plan_step=current_step,
                tool_name="response_validation",
            )

        response_message = rsp.choices[0].message
        if not getattr(response_message, "tool_calls", None):
            raise PlanExecutionError(
                "Invalid LLM response: no tool calls found",
                plan_step=current_step,
                tool_name="response_validation",
            )

        plan = Plan(step=current_step, llm_plan=response_message, ttl=ttl)
        return plan

    def execute_tool_call(
        self,
        chaining_message,
        selected_tools: list[str] | None = None,
        ttl: int = 1,
    ) -> "Plan":
        """
        Execute tool call with comprehensive error handling.
        """
        try:
            # Validate inputs
            self._validate_tool_call_inputs(chaining_message, ttl)

            system_prompt = "You are an executor that executes the plan given to you in the prompt through tool calls."
            self.agent.llm.system_prompt = system_prompt

            try:
                rsp = self.agent.llm.generate(
                    prompt=chaining_message,
                    tool_schema=self.agent.tool_manager.get_all_tools_schema(
                        selected_tools=selected_tools
                    ),
                    tool_choice="required",
                )
            except Exception as e:
                raise PlanExecutionError(
                    f"LLM generation failed during tool execution: {e}",
                    plan_step=getattr(self.agent.model, "steps", 0),
                    tool_name="llm_generation",
                ) from e

            plan = self._build_plan_from_response(rsp, ttl)

            logger.info(
                f"Successfully executed tool call for agent {self.agent.unique_id} at step {self.agent.model.steps}"
            )
            return plan

        except (ReasoningError, PlanExecutionError):
            raise
        except Exception as e:
            logger.exception(
                f"Unexpected error in execute_tool_call for agent {self.agent.unique_id}: {e}"
            )
            raise ReasoningError(
                f"Unexpected error during tool execution: {e}",
                reasoning_type="tool_execution",
                step=getattr(self.agent.model, "steps", 0),
            ) from e

    async def aexecute_tool_call(
        self,
        chaining_message,
        selected_tools: list[str] | None = None,
        ttl: int = 1,
    ) -> "Plan":
        """
        Asynchronous version of execute_tool_call() method with error handling.
        """
        try:
            # Validate inputs
            self._validate_tool_call_inputs(chaining_message, ttl)

            system_prompt = "You are an executor that executes the plan given to you in the prompt through tool calls."
            self.agent.llm.system_prompt = system_prompt

            try:
                rsp = await self.agent.llm.agenerate(
                    prompt=chaining_message,
                    tool_schema=self.agent.tool_manager.get_all_tools_schema(
                        selected_tools=selected_tools
                    ),
                    tool_choice="required",
                )
            except Exception as e:
                raise PlanExecutionError(
                    f"Async LLM generation failed during tool execution: {e}",
                    plan_step=getattr(self.agent.model, "steps", 0),
                    tool_name="llm_generation",
                ) from e

            plan = self._build_plan_from_response(rsp, ttl)

            logger.info(
                f"Successfully executed async tool call for agent {self.agent.unique_id} at step {self.agent.model.steps}"
            )
            return plan

        except (ReasoningError, PlanExecutionError):
            raise
        except Exception as e:
            logger.exception(
                f"Unexpected error in aexecute_tool_call for agent {self.agent.unique_id}: {e}"
            )
            raise ReasoningError(
                f"Unexpected error during async tool execution: {e}",
                reasoning_type="tool_execution",
                step=getattr(self.agent.model, "steps", 0),
            ) from e
