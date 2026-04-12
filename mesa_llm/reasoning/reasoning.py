from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mesa_llm.llm_agent import LLMAgent


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
        - **abstract plan(prompt, obs=None, ttl=1, selected_tools=None, tool_calls="auto")** → *Plan* - Generate synchronous plan
        - **async aplan(prompt, obs=None, ttl=1, selected_tools=None, tool_calls="auto")** → *Plan* - Generate asynchronous plan


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
        tool_calls: str | None = "auto",
    ) -> Plan:
        """Generate a plan for the next action.

        Args:
            prompt: Optional prompt override for the reasoning strategy.
            obs: Optional observation to plan against.
            ttl: Time-to-live for the generated plan.
            selected_tools: Optional explicit tool allowlist forwarded to
                ``ToolManager.get_all_tools_schema()``. If omitted or ``None``,
                the default behavior exposes all tools. ``[]`` exposes no
                tools, and a non-empty list restricts planning/execution to
                the named tools.
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
        selected_tools: list[str] | None = None,
        tool_calls: str | None = "auto",
    ) -> Plan:
        """
        Asynchronous version of plan() method for parallel planning.
        Default implementation calls the synchronous plan() method.

        ``selected_tools`` follows the same contract as ``plan()``: omitting
        it or passing ``None`` uses the default behavior of exposing all
        tools, ``[]`` exposes no tools, and a non-empty list restricts
        planning/execution to the named tools.
        """
        return self.plan(
            prompt=prompt,
            obs=obs,
            ttl=ttl,
            selected_tools=selected_tools,
            tool_calls=tool_calls,
        )

    def execute_tool_call(
        self,
        chaining_message,
        selected_tools: list[str] | None = None,
        ttl: int = 1,
        tool_calls: str | None = "auto",
    ):
        """Turn a natural-language plan into tool calls.

        Args:
            chaining_message: Natural-language plan or action text to execute.
            selected_tools: Optional explicit tool allowlist forwarded to
                ``ToolManager.get_all_tools_schema()``. Omitting it or passing
                ``None`` uses the default behavior of exposing all tools,
                ``[]`` exposes no tools, and a non-empty list restricts
                execution to the named tools.
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
        self.agent.llm.system_prompt = system_prompt
        rsp = self.agent.llm.generate(
            prompt=chaining_message,
            tool_schema=self.agent.tool_manager.get_all_tools_schema(
                selected_tools=selected_tools
            ),
            tool_choice=tool_calls,
        )
        response_message = rsp.choices[0].message
        plan = Plan(step=self.agent.model.steps, llm_plan=response_message, ttl=ttl)
        self.agent.memory.add_to_memory(
            type="plan_execution", content={"content": str(plan)}
        )

        return plan

    async def aexecute_tool_call(
        self,
        chaining_message,
        selected_tools: list[str] | None = None,
        ttl: int = 1,
        tool_calls: str | None = "auto",
    ):
        """
        Asynchronous version of execute_tool_call() method.

        ``selected_tools`` follows the same contract as
        ``execute_tool_call()``: omitting it or passing ``None`` uses the
        default behavior of exposing all tools, ``[]`` exposes no tools, and
        a non-empty list restricts execution to the named tools.
        """
        system_prompt = (
            "You are an executor that executes the plan given to you in the prompt through tool calls. "
            "If the plan concludes that no action should be taken, do not call any tool."
        )
        self.agent.llm.system_prompt = system_prompt
        rsp = await self.agent.llm.agenerate(
            prompt=chaining_message,
            tool_schema=self.agent.tool_manager.get_all_tools_schema(
                selected_tools=selected_tools
            ),
            tool_choice=tool_calls,
        )
        response_message = rsp.choices[0].message
        plan = Plan(step=self.agent.model.steps, llm_plan=response_message, ttl=ttl)
        await self.agent.memory.aadd_to_memory(
            type="plan_execution", content={"content": str(plan)}
        )

        return plan
