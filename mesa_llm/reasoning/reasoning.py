import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mesa_llm.llm_agent import LLMAgent
    from mesa_llm.tools.tool_manager import ToolManager


_UNSET = object()
ToolRef = Callable | str
ToolSelection = ToolRef | list[ToolRef] | tuple[ToolRef, ...] | None


@dataclass
class Observation:
    """
    A structured snapshot containing the agent's current step, self-state
    (internal attributes and location), and local-state (neighboring agents
    and their properties). This provides complete situational awareness for
    decision-making.

    Attributes:
        step (int): The current simulation time step when the observation is made.

        self_state (dict): A dictionary containing comprehensive information about the observing agent itself.
            This includes:
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
    tools: ToolSelection | object = _UNSET

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
        - **abstract plan(prompt, obs=None, ttl=1, tools=<inherit>, tool_calls="auto")** → *Plan* - Generate synchronous plan
        - **async aplan(prompt, obs=None, ttl=1, tools=<inherit>, tool_calls="auto")** → *Plan* - Generate asynchronous plan


    Reasoning Flow:
        1. Agent generates **observation** of current situation through `generate_obs()`
        2. Reasoning strategies access **memory** to inform decisions
        3. Selected reasoning approach processes observation and memory into a structured **plan**
        4. Plans are automatically converted to **tool schemas** for LLM function calling
        5. Tool manager **executes the planned actions** in the simulation environment
    """

    def __init__(self, agent: "LLMAgent"):
        self.agent = agent

    def _tool_manager(self) -> "ToolManager":
        manager = getattr(self.agent, "__dict__", {}).get("_tool_manager")
        if manager is not None:
            return manager
        return self.agent.tool_manager

    def _resolve_tools_argument(
        self,
        tools: ToolSelection | object = _UNSET,
        selected_tools: ToolSelection | object = _UNSET,
    ) -> ToolSelection | object:
        if selected_tools is not _UNSET:
            warnings.warn(
                "`selected_tools` is deprecated; use `tools=` instead.",
                DeprecationWarning,
                stacklevel=3,
            )
            if tools is not _UNSET:
                raise ValueError("Use either `tools` or `selected_tools`, not both.")
            tools = _UNSET if selected_tools is None else selected_tools
        return tools

    def _get_tools_schema(self, tools: ToolSelection | object = _UNSET) -> list[dict]:
        manager = self._tool_manager()
        if tools is _UNSET:
            return manager.get_tools_schema()
        return manager.get_tools_schema(tools=tools)

    @abstractmethod
    def plan(
        self,
        prompt: str | None = None,
        obs: Observation | None = None,
        ttl: int = 1,
        tools: ToolSelection | object = _UNSET,
        tool_calls: str | None = "auto",
        selected_tools: ToolSelection | object = _UNSET,
    ) -> Plan:
        """Generate a plan for the next action.

        Args:
            prompt: Optional prompt override for the reasoning strategy.
            obs: Optional observation to plan against.
            ttl: Time-to-live for the generated plan.
            tools: Optional explicit tool selector. If omitted, the reasoning call
                inherits the agent's configured tools. Explicit ``None`` or
                ``[]`` exposes no tools, and a callable, string name, or
                sequence exposes exactly those configured tools.
            tool_calls: Execution-phase LiteLLM ``tool_choice`` override used
                when converting the natural-language plan into tool calls.
                Planning still keeps tool use disabled.

                Supported values in Mesa-LLM are:
                - ``None``: defer to LiteLLM/provider default behavior. In
                  practice, this usually means no tool calls when no tools are
                  provided and behavior similar to ``"auto"`` when tools are
                  available.
                - ``"none"``: never return tool calls; return a normal
                  assistant message instead.
                - ``"auto"``: allow the model to either return a normal
                  assistant message or call one or more tools.
                - ``"required"``: require the model to call one or more tools.

                Mesa-LLM currently exposes only these string choices, not
                provider-specific object forms. See LiteLLM docs:
                https://docs.litellm.ai/
        """

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
        Default implementation calls the synchronous plan() method.

        ``tools`` follows the same contract as ``plan()``: omitted inherits
        the agent's configured tools, explicit ``None`` or ``[]`` exposes no
        tools, and a callable, string name, or sequence exposes exactly those
        configured tools.
        """
        return self.plan(
            prompt=prompt,
            obs=obs,
            ttl=ttl,
            tools=tools,
            tool_calls=tool_calls,
            selected_tools=selected_tools,
        )

    def execute_tool_call(
        self,
        chaining_message,
        tools: ToolSelection | object = _UNSET,
        ttl: int = 1,
        tool_calls: str | None = "auto",
        selected_tools: ToolSelection | object = _UNSET,
    ):
        """Turn a natural-language plan into tool calls.

        Args:
            chaining_message: Natural-language plan or action text to execute.
            tools: Optional explicit tool selector. If omitted, the execution call
                inherits the agent's configured tools. Explicit ``None`` or
                ``[]`` exposes no tools, and a callable, string name, or
                sequence exposes exactly those configured tools.
            ttl: Time-to-live for the returned plan.
            tool_calls: LiteLLM ``tool_choice`` passed to the execution call.
                Supported values in Mesa-LLM are:
                - ``None``: defer to LiteLLM/provider default behavior. In
                  practice, this usually means no tool calls when no tools are
                  provided and behavior similar to ``"auto"`` when tools are
                  available.
                - ``"none"``: never return tool calls; return a normal
                  assistant message instead.
                - ``"auto"``: allow the model to either return a normal
                  assistant message or call one or more tools.
                - ``"required"``: require the model to call one or more tools.

                Mesa-LLM currently exposes only these string choices, not
                provider-specific object forms. See LiteLLM docs:
                https://docs.litellm.ai/
        """
        system_prompt = (
            "You are an executor that executes the plan given to you in the prompt through tool calls. "
            "If the plan concludes that no action should be taken, do not call any tool."
        )
        tools = self._resolve_tools_argument(tools, selected_tools)
        rsp = self.agent.llm.generate(
            prompt=chaining_message,
            tool_schema=self._get_tools_schema(tools),
            tool_choice=tool_calls,
            system_prompt=system_prompt,
        )
        response_message = rsp.choices[0].message
        plan = Plan(
            step=self.agent.model.steps,
            llm_plan=response_message,
            ttl=ttl,
            tools=tools,
        )
        self.agent.memory.add_to_memory(
            type="plan_execution", content={"content": str(plan)}
        )

        return plan

    async def aexecute_tool_call(
        self,
        chaining_message,
        tools: ToolSelection | object = _UNSET,
        ttl: int = 1,
        tool_calls: str | None = "auto",
        selected_tools: ToolSelection | object = _UNSET,
    ):
        """
        Asynchronous version of execute_tool_call() method.

        ``tools`` follows the same contract as ``execute_tool_call()``:
        omitted inherits the agent's configured tools, explicit ``None`` or
        ``[]`` exposes no tools, and a callable, string name, or sequence
        exposes exactly those configured tools.
        """
        system_prompt = (
            "You are an executor that executes the plan given to you in the prompt through tool calls. "
            "If the plan concludes that no action should be taken, do not call any tool."
        )
        tools = self._resolve_tools_argument(tools, selected_tools)
        rsp = await self.agent.llm.agenerate(
            prompt=chaining_message,
            tool_schema=self._get_tools_schema(tools),
            tool_choice=tool_calls,
            system_prompt=system_prompt,
        )
        response_message = rsp.choices[0].message
        plan = Plan(
            step=self.agent.model.steps,
            llm_plan=response_message,
            ttl=ttl,
            tools=tools,
        )
        await self.agent.memory.aadd_to_memory(
            type="plan_execution", content={"content": str(plan)}
        )

        return plan
