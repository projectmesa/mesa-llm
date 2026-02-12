from mesa_llm.llm_agent import LLMAgent
from mesa_llm.tools.tool_manager import ToolManager

seller_tool_manager = ToolManager()
buyer_tool_manager = ToolManager()


def get_dialogue_history(agent, max_messages: int = 5) -> str:
    """Extract and format recent dialogue from an agent's memory.

    This helper function supports both STLTMemory (short_term_memory) and
    EpisodicMemory (memory_entries). It efficiently extracts the last N
    dialogue messages by iterating in reverse order.

    Args:
        agent: The LLMAgent whose memory to extract dialogue from
        max_messages: Maximum number of dialogue messages to return (default: 5)

    Returns:
        Formatted dialogue history string, or "No recent dialogue." if empty
    """
    dialogue = []

    # Support both STLTMemory and EpisodicMemory
    memory_source = None
    if hasattr(agent.memory, "short_term_memory"):
        memory_source = agent.memory.short_term_memory
    elif hasattr(agent.memory, "memory_entries"):
        memory_source = agent.memory.memory_entries

    if memory_source:
        # Iterate in reverse to efficiently get last N messages
        # We check at most max_messages * 2 recent entries to account for
        # non-dialogue entries (observations, movements, etc.)
        entries_to_check = min(len(memory_source), max_messages * 2)

        for entry in reversed(list(memory_source)[-entries_to_check:]):
            # Stop if we already have enough dialogue messages
            if len(dialogue) >= max_messages:
                break

            # Check if entry.content is a dict and has 'message'
            if isinstance(entry.content, dict) and "message" in entry.content:
                sender = entry.content.get("sender", "Unknown")
                msg = entry.content.get("message", "")

                # Handle both agent objects and agent IDs
                if hasattr(sender, "unique_id"):
                    # sender is an agent object (from send_message())
                    sender_name = f"{type(sender).__name__} {sender.unique_id}"
                elif isinstance(sender, int):
                    # sender is an ID (from speak_to tool)
                    # Try to find the agent by ID to get its type
                    try:
                        agent_obj = next(
                            a for a in agent.model.agents if a.unique_id == sender
                        )
                        sender_name = f"{type(agent_obj).__name__} {sender}"
                    except StopIteration:
                        sender_name = f"Agent {sender}"
                else:
                    sender_name = str(sender)

                dialogue.append(f"- {sender_name}: {msg}")

    # Reverse to get chronological order (oldest first)
    dialogue.reverse()
    return "\n".join(dialogue) if dialogue else "No recent dialogue."


class SellerAgent(LLMAgent):
    def __init__(
        self,
        model,
        reasoning,
        llm_model,
        system_prompt,
        vision,
        internal_state,
    ):
        super().__init__(
            model=model,
            reasoning=reasoning,
            llm_model=llm_model,
            system_prompt=system_prompt,
            vision=vision,
            internal_state=internal_state,
        )

        self.tool_manager = seller_tool_manager
        self.sales = 0

    def step(self):
        observation = self.generate_obs()
        dialogue_history = get_dialogue_history(self)

        prompt = (
            f"DIALOGUE HISTORY:\n{dialogue_history}\n\n"
            "INSTRUCTIONS:\n"
            "Don't move around. If there are any buyers in your cell or in the neighboring cells, "
            "pitch them your product using the speak_to tool. "
            "Talk to them until they agree or definitely refuse to buy your product. "
            "Use the dialogue history to inform your next response (e.g., if you already offered a price, stick to it or negotiate)."
        )

        plan = self.reasoning.plan(
            prompt=prompt, obs=observation, selected_tools=["speak_to"]
        )
        self.apply_plan(plan)

    async def astep(self):
        observation = self.generate_obs()
        dialogue_history = get_dialogue_history(self)

        prompt = (
            f"DIALOGUE HISTORY:\n{dialogue_history}\n\n"
            "INSTRUCTIONS:\n"
            "Don't move around. If there are any buyers in your cell or in the neighboring cells, "
            "pitch them your product using the speak_to tool. "
            "Talk to them until they agree or definitely refuse to buy your product. "
            "Use the dialogue history to inform your next response."
        )

        plan = await self.reasoning.aplan(
            prompt=prompt, obs=observation, selected_tools=["speak_to"]
        )
        self.apply_plan(plan)


class BuyerAgent(LLMAgent):
    def __init__(
        self,
        model,
        reasoning,
        llm_model,
        system_prompt,
        vision,
        internal_state,
        budget,
    ):
        super().__init__(
            model=model,
            reasoning=reasoning,
            llm_model=llm_model,
            system_prompt=system_prompt,
            vision=vision,
            internal_state=internal_state,
        )
        self.tool_manager = buyer_tool_manager
        self.budget = budget
        self.products = []

    def step(self):
        observation = self.generate_obs()
        dialogue_history = get_dialogue_history(self)

        prompt = (
            f"DIALOGUE HISTORY:\n{dialogue_history}\n\n"
            "INSTRUCTIONS:\n"
            f"Your budget is ${self.budget}. "
            f"Move around by using the teleport_to_location tool if you are not talking to a seller, "
            f"grid dimensions are {self.model.grid.width} x {self.model.grid.height}. "
            "Seller agents around you might try to pitch their product by sending you messages, get as much information as possible. "
            "When you have enough information, decide what to buy the product. "
            "Refer to the dialogue history to recall previous prices offered."
        )
        plan = self.reasoning.plan(
            prompt=prompt,
            obs=observation,
            selected_tools=["teleport_to_location", "speak_to", "buy_product"],
        )
        self.apply_plan(plan)

    async def astep(self):
        observation = self.generate_obs()
        dialogue_history = get_dialogue_history(self)

        prompt = (
            f"DIALOGUE HISTORY:\n{dialogue_history}\n\n"
            "INSTRUCTIONS:\n"
            f"Your budget is ${self.budget}. "
            f"Move around by using the teleport_to_location tool if you are not talking to a seller, "
            f"grid dimensions are {self.model.grid.width} x {self.model.grid.height}. "
            "Seller agents around you might try to pitch their product by sending you messages, get as much information as possible. "
            "When you have enough information, decide what to buy the product. "
            "Refer to the dialogue history to recall previous prices offered."
        )
        plan = await self.reasoning.aplan(
            prompt=prompt,
            obs=observation,
            selected_tools=["teleport_to_location", "speak_to", "buy_product"],
        )
        self.apply_plan(plan)
