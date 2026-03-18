import asyncio
import logging
import time
from typing import Any

from mesa import Agent, Model
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

logger = logging.getLogger(__name__)


class AgentStateError(Exception):
    """Exception for agent state consistency errors."""


class CheckpointError(Exception):
    """Exception for checkpoint/restart failures."""


class OptimizedMessageBus:
    """
    Optimized message bus for O(n) agent communication instead of O(n²).
    """

    def __init__(self):
        self.message_queue = asyncio.Queue()
        self.subscribers = {}
        self.batch_processor = None

    async def broadcast_message(self, sender, message, recipients):
        """O(n) message broadcasting with batching."""
        message_data = {
            "sender": sender.unique_id,
            "message": message,
            "recipients": [r.unique_id for r in recipients],
            "timestamp": time.time(),
        }

        # Add to batch queue
        await self.message_queue.put(message_data)

    async def process_message_batch(self):
        """Process messages in batches."""
        batch = []
        while not self.message_queue.empty() and len(batch) < 50:
            batch.append(await self.message_queue.get())

        # Group by recipients for efficient delivery
        recipient_groups = {}
        for msg in batch:
            for recipient_id in msg["recipients"]:
                if recipient_id not in recipient_groups:
                    recipient_groups[recipient_id] = []
                recipient_groups[recipient_id].append(msg)

        # Deliver to each recipient
        delivery_tasks = []
        for recipient_id, messages in recipient_groups.items():
            recipient = self.get_agent_by_id(recipient_id)
            if recipient:
                delivery_tasks.append(self.deliver_messages_batch(recipient, messages))

        await asyncio.gather(*delivery_tasks, return_exceptions=True)

    async def deliver_messages_batch(self, recipient, messages):
        """Deliver batch of messages to a recipient."""
        for msg in messages:
            await recipient.memory.aadd_to_memory(
                type="message",
                content={
                    "message": msg["message"],
                    "sender": msg["sender"],
                    "recipients": msg["recipients"],
                },
            )

    def get_agent_by_id(self, agent_id):
        """Get agent by ID from model."""
        if hasattr(self, "model") and hasattr(self.model, "agents"):
            for agent in self.model.agents:
                if hasattr(agent, "unique_id") and agent.unique_id == agent_id:
                    return agent
        return None


# Global message bus instance
_global_message_bus = OptimizedMessageBus()
MAX_CHECKPOINTS = 5


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
        enable_checkpoints: bool = True,
        checkpoint_interval: int = 10,
    ):
        super().__init__(model=model)

        self.model = model
        self.step_prompt = step_prompt
        self.llm = ModuleLLM(
            llm_model=llm_model,
            system_prompt=system_prompt,
            enable_caching=True,  # Enable response caching
            enable_batching=True,  # Enable request batching
            cache_size=1000,  # Cache up to 1000 responses
            cache_ttl=300.0,  # Cache for 5 minutes
            batch_size=10,  # Batch up to 10 requests
        )

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

        # Checkpoint and state management
        self.enable_checkpoints = enable_checkpoints
        self.checkpoint_interval = checkpoint_interval
        self._checkpoint_data = {}
        self._state_validation_errors = []
        self._recovery_mode = False

        if isinstance(internal_state, str):
            internal_state = [internal_state]
        elif internal_state is None:
            internal_state = []

        self.internal_state = internal_state

    def __str__(self):
        return f"LLMAgent {self.unique_id}"

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
                "tool_calls": [
                    {k: v for k, v in tc.items() if k not in ["tool_call_id", "role"]}
                    for tc in tool_call_resp
                ]
            },
        )

        return tool_call_resp

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
                "tool_calls": [
                    {k: v for k, v in tc.items() if k not in ["tool_call_id", "role"]}
                    for tc in tool_call_resp
                ]
            },
        )

        return tool_call_resp

    def _build_observation(self):
        """
        Construct the observation data visible to the agent at the current model step.

        This method encapsulates the shared logic used by both sync and
        async observation generation.
        This method constructs the agent's self state and determines which other
        agents are observable based on the configured vision:

        - vision > 0:
            The agent observes all agents within the specified vision radius.
        - vision == -1:
            The agent observes all agents present in the simulation.
        - vision == 0 or vision is None:
            The agent observes no other agents.

        The method supports grid-based and continuous spaces and builds a local
        state representation for all visible neighboring agents.

        Returns self_state and local_state of the agent
        """
        self_state = {
            "agent_unique_id": self.unique_id,
            "system_prompt": self.system_prompt,
            "location": (
                self.pos
                if self.pos is not None
                else (
                    getattr(self, "cell", None).coordinate
                    if getattr(self, "cell", None) is not None
                    else None
                )
            ),
            "internal_state": self.internal_state,
        }
        if self.vision is not None and self.vision > 0:
            # Check which type of space/grid the model uses
            grid = getattr(self.model, "grid", None)
            space = getattr(self.model, "space", None)

            if grid and isinstance(grid, SingleGrid | MultiGrid):
                neighbors = grid.get_neighbors(
                    tuple(self.pos),
                    moore=True,
                    include_center=False,
                    radius=self.vision,
                )
            elif space and isinstance(space, ContinuousSpace):
                all_nearby = space.get_neighbors(
                    self.pos, radius=self.vision, include_center=True
                )
                neighbors = [a for a in all_nearby if a is not self]

            else:
                # No recognized grid/space type
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
        This method builds the agent's observation using the shared observation
        construction logic, stores it in the agent's memory module using
        async memory operations, and returns it as an Observation instance.
        """
        step = self.model.steps
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
        This method delegates observation construction to the shared observation
        builder, stores the resulting observation in the agent's memory module,
        and returns it as an Observation instance.
        """
        step = self.model.steps
        self_state, local_state = self._build_observation()
        # Add to memory (memory handles its own display separately)
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
        # For now, use the original synchronous implementation
        # The optimized async message bus can be used in async contexts
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
        """
        Asynchronous version of pre_step.
        """
        await self.memory.aprocess_step(pre_step=True)

    async def apost_step(self):
        """
        Asynchronous version of post_step.
        """
        await self.memory.aprocess_step()

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

    def _validate_agent_state(self) -> None:
        """
        Validate agent state consistency after operations.
        Raises AgentStateError if inconsistencies are found.
        """
        errors = []

        # Check essential attributes
        if not hasattr(self, "unique_id") or self.unique_id is None:
            errors.append("Agent missing unique_id")

        if not hasattr(self, "model") or self.model is None:
            errors.append("Agent missing model reference")

        # Check memory consistency
        if hasattr(self, "memory") and self.memory:
            try:
                # Test memory access
                if hasattr(self.memory, "get_recent_observations"):
                    _ = self.memory.get_recent_observations(1)
                else:
                    # Minimal sanity check available on all Memory subclasses
                    _ = self.memory.step_content

            except Exception as e:
                errors.append(f"Memory access error: {e}")

        # Check reasoning consistency
        if (
            hasattr(self, "reasoning")
            and self.reasoning
            and (not hasattr(self.reasoning, "agent") or self.reasoning.agent != self)
        ):
            errors.append("Reasoning agent reference inconsistent")

        # Check tool manager consistency
        if hasattr(self, "tool_manager") and self.tool_manager:
            try:
                stats = self.tool_manager.get_execution_stats()
                if not isinstance(stats, dict):
                    errors.append("Tool manager stats invalid")
            except Exception as e:
                errors.append(f"Tool manager access error: {e}")

        # Check LLM consistency
        if hasattr(self, "llm") and self.llm:
            try:
                _ = self.llm.get_performance_stats()
            except Exception as e:
                errors.append(f"LLM access error: {e}")

        if errors:
            self._state_validation_errors.extend(errors)
            error_msg = f"Agent state validation failed: {'; '.join(errors)}"
            logger.error(f"Agent {self.unique_id}: {error_msg}")
            raise AgentStateError(error_msg)

    def _create_checkpoint(self) -> dict[str, Any]:
        """
        Create a checkpoint of the current agent state.
        """
        try:
            checkpoint = {
                "timestamp": time.time(),
                "step": getattr(self.model, "steps", 0),
                "unique_id": self.unique_id,
                "pos": self.pos,
                "internal_state": self.internal_state.copy()
                if self.internal_state
                else [],
                "system_prompt": self.system_prompt,
                "vision": self.vision,
                "memory_data": None,  # Will be populated below
                "tool_stats": None,  # Will be populated below
                "llm_stats": None,  # Will be populated below
                "current_plan": None,  # Will be populated below
            }

            # Safely capture memory state
            if hasattr(self, "memory") and self.memory:
                try:
                    checkpoint["memory_data"] = {
                        "recent_observations": len(
                            getattr(self.memory, "short_term_memory", [])
                        ),
                        "consolidated_memory": len(
                            getattr(self.memory, "consolidated_memory", [])
                        ),
                    }
                except Exception as e:
                    logger.warning(f"Failed to capture memory in checkpoint: {e}")

            # Safely capture tool manager stats
            if hasattr(self, "tool_manager") and self.tool_manager:
                try:
                    checkpoint["tool_stats"] = self.tool_manager.get_execution_stats()
                except Exception as e:
                    logger.warning(f"Failed to capture tool stats in checkpoint: {e}")

            # Safely capture LLM stats
            if hasattr(self, "llm") and self.llm:
                try:
                    checkpoint["llm_stats"] = self.llm.get_performance_stats()
                except Exception as e:
                    logger.warning(f"Failed to capture LLM stats in checkpoint: {e}")

            # Safely capture current plan
            if hasattr(self, "_current_plan") and self._current_plan:
                try:
                    checkpoint["current_plan"] = str(self._current_plan)
                except Exception as e:
                    logger.warning(f"Failed to capture current plan in checkpoint: {e}")

            return checkpoint

        except Exception as e:
            logger.error(f"Failed to create checkpoint for agent {self.unique_id}: {e}")
            raise CheckpointError(f"Checkpoint creation failed: {e}") from e

    def _restore_from_checkpoint(self, checkpoint: dict) -> None:
        try:
            self.pos = checkpoint.get("pos")
            if "internal_state" in checkpoint:
                self.internal_state = checkpoint["internal_state"].copy()

            # Reset error tracking on restore
            self._state_validation_errors.clear()
            self._recovery_mode = False

            # Warn clearly that memory is not restored
            if checkpoint.get("memory_data"):
                logger.warning(
                    "Agent %s: memory state NOT restored from checkpoint — "
                    "STLTMemory does not support restoration yet. "
                    "Agent will continue with current memory.",
                    self.unique_id,
                )

            logger.info(
                "Agent %s restored pos and internal_state from checkpoint %s",
                self.unique_id,
                checkpoint.get("step"),
            )
        except Exception as e:
            raise CheckpointError(f"Checkpoint restoration failed: {e}") from e

    def checkpoint_now(self) -> str:
        """
        Create and return a checkpoint identifier.
        Returns the checkpoint ID.
        """
        if not self.enable_checkpoints:
            logger.warning(f"Checkpoints disabled for agent {self.unique_id}")
            return ""

        checkpoint_id = f"{self.unique_id}_{int(time.time())}"
        self._checkpoint_data[checkpoint_id] = self._create_checkpoint()

        # Evict previous checkpoints beyond the cap
        while len(self._checkpoint_data) > MAX_CHECKPOINTS:
            prev_key = next(
                iter(self._checkpoint_data)
            )  # dict is insertion-ordered (3.7+)
            del self._checkpoint_data[prev_key]
            logger.debug(
                "Evicted prev checkpoint %s for agent %s", prev_key, self.unique_id
            )

        logger.info(f"Created checkpoint {checkpoint_id} for agent {self.unique_id}")
        return checkpoint_id

    def restore_from_checkpoint_id(self, checkpoint_id: str) -> bool:
        """
        Restore from a previously created checkpoint.
        Returns True if successful, False otherwise.
        """
        if checkpoint_id not in self._checkpoint_data:
            logger.error(
                f"Checkpoint {checkpoint_id} not found for agent {self.unique_id}"
            )
            return False

        try:
            self._restore_from_checkpoint(self._checkpoint_data[checkpoint_id])
            return True
        except CheckpointError:
            return False

    def get_available_checkpoints(self) -> list[str]:
        """
        Get list of available checkpoint IDs.
        """
        return list(self._checkpoint_data.keys())

    def clear_checkpoints(self) -> None:
        """
        Clear all checkpoint data.
        """
        self._checkpoint_data.clear()
        logger.info(f"Cleared all checkpoints for agent {self.unique_id}")

    def get_state_validation_errors(self) -> list[str]:
        """
        Get list of state validation errors.
        """
        return self._state_validation_errors.copy()

    def clear_state_validation_errors(self) -> None:
        """
        Clear state validation errors.
        """
        self._state_validation_errors.clear()

    def _checkpoint_if_needed(self) -> None:
        """
        Create checkpoint if interval has passed.
        """
        if not self.enable_checkpoints:
            return

        current_step = getattr(self.model, "steps", 0)
        if current_step > 0 and current_step % self.checkpoint_interval == 0:
            self.checkpoint_now()

    async def astep(self):
        """
        Default asynchronous step method with checkpoint and state validation.
        """
        try:
            # Validate state before step
            if not self._recovery_mode:
                self._validate_agent_state()

            # Execute pre-step
            await self.apre_step()

            # Call the RAW user step, not the wrapped one, to avoid double pre/post
            user_step_fn = self.__class__.__dict__.get("step")
            if user_step_fn is not None:
                user_step_fn(self)

            # Execute post-step
            await self.apost_step()

            # Create checkpoint if needed
            self._checkpoint_if_needed()

        except AgentStateError as e:
            logger.error(f"Agent {self.unique_id} state error during step: {e}")
            # Enter recovery mode
            self._recovery_mode = True
            # Try to restore from last checkpoint
            available_checkpoints = self.get_available_checkpoints()
            if available_checkpoints:
                logger.info(f"Attempting recovery for agent {self.unique_id}")
                if self.restore_from_checkpoint_id(available_checkpoints[-1]):
                    self._recovery_mode = False
                    logger.info(f"Agent {self.unique_id} recovered successfully")
                else:
                    logger.error(f"Agent {self.unique_id} recovery failed")
            else:
                logger.error(
                    f"No checkpoints available for agent {self.unique_id} recovery"
                )

        except Exception as e:
            logger.exception(f"Unexpected error in agent {self.unique_id} step: {e}")
            raise

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
                # State validation before step (mirrors astep behaviour)
                if not self._recovery_mode:
                    try:
                        self._validate_agent_state()
                    except AgentStateError as e:
                        logger.error(
                            "Agent %s state error before sync step: %s",
                            self.unique_id,
                            e,
                        )
                        # Attempt recovery; proceed regardless so Mesa's scheduler isn't blocked
                        available = self.get_available_checkpoints()
                        if available:
                            self.restore_from_checkpoint_id(available[-1])
                LLMAgent.pre_step(self, *args, **kwargs)
                result = user_step(self, *args, **kwargs)
                LLMAgent.post_step(self, *args, **kwargs)

                # Checkpoint after step
                self._checkpoint_if_needed()
                return result

            cls.step = wrapped

        if user_astep:

            async def awrapped(self, *args, **kwargs):
                """
                Async wrapper for astep method.
                """
                await self.apre_step()
                result = await user_astep(self, *args, **kwargs)
                await self.apost_step()
                return result

            cls.astep = awrapped
