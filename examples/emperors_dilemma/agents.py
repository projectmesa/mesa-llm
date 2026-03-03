from mesa.discrete_space import FixedAgent
from mesa_llm.llm_agent import LLMAgent
from mesa_llm.reasoning.react import ReActReasoning


from tools import emperor_tool_manager


class EmperorLLMAgent(LLMAgent, FixedAgent):
    """An LLM-powered agent in the Emperor's Dilemma model.

    Each agent has a private belief about the norm but makes public decisions
    about compliance and enforcement based on LLM reasoning about social pressure
    from neighbors. This replaces the mathematical threshold model with
    language-driven decision making.
    """

    def __init__(
        self,
        model,
        reasoning,
        llm_model,
        system_prompt,
        internal_state,
        private_belief,
        conviction,
        k,
    ):
        """Initializes the EmperorLLMAgent.

        Args:
            model: The Mesa model instance.
            reasoning: The reasoning class (e.g. ReActReasoning).
            llm_model (str): The LLM model string (e.g. 'groq/llama-3.3-70b-versatile').
            system_prompt (str): The agent's persona and background.
            internal_state (str): Initial internal state description.
            private_belief (int): The agent's true belief — 1 (supports norm) or -1 (rejects norm).
            conviction (float): How strongly the agent holds their private belief.
            k (float): The personal cost of enforcing the norm on others.
        """
        LLMAgent.__init__(
            self,
            model=model,
            reasoning=reasoning,
            llm_model=llm_model,
            system_prompt=system_prompt,
            internal_state=internal_state,
        )

        self.tool_manager = emperor_tool_manager
        self.private_belief = private_belief
        self.conviction = conviction
        self.k = k

        # Start with private belief as public stance, not enforcing
        self.compliance = self.private_belief
        self.enforcement = 0

    def step(self):
        """Executes one step of the agent using LLM reasoning.

        The agent observes neighbor behavior, builds a natural language
        description of social pressure, and uses the LLM to decide
        whether to publicly comply and whether to enforce on neighbors.
        """
        # 1. Observe neighbors
        neighbors = []
        if self.cell is not None:
            neighbors = list(self.cell.neighborhood.agents)

        num_neighbors = len(neighbors)
        num_complying = sum(1 for n in neighbors if n.compliance == 1)
        num_enforcing = sum(1 for n in neighbors if n.enforcement == 1)
        num_resisting = sum(1 for n in neighbors if n.compliance != 1)

        # 2. Generate observation
        observation = self.generate_obs()

        # 3. Build prompt with social context
        belief_description = "support" if self.private_belief == 1 else "privately reject"
        conviction_description = (
            "very strongly" if self.conviction > 0.3
            else "moderately" if self.conviction > 0.15
            else "weakly"
        )

        prompt = f"""
You are a citizen in a society governed by a powerful norm — publicly praising the Emperor's hat.

YOUR PRIVATE SITUATION:
- You {belief_description} the norm in your heart.
- You hold this belief {conviction_description} (conviction level: {self.conviction:.2f}).
- Enforcing the norm costs you personally (cost k={self.k:.3f}).

WHAT YOU SEE AROUND YOU RIGHT NOW:
- You have {num_neighbors} neighbors.
- {num_complying} of them are publicly complying with the norm.
- {num_enforcing} of them are actively pressuring others to comply.
- {num_resisting} of them are visibly resisting or staying quiet.

YOUR DECISION:
Based on your private belief, your conviction, and the social pressure you observe,
decide two things:

1. Do you publicly COMPLY with the norm today? (Even if you privately disagree?)
2. Do you ENFORCE the norm on your neighbors? (Pressure them to comply?)

Use set_compliance to record your compliance decision.
Use set_enforcement to record your enforcement decision.

Think carefully — you may comply out of fear even if you disagree, and you may enforce
even if you privately hate the norm, just to avoid suspicion.
"""

        # 4. Plan and execute — split into two calls for Groq compatibility
        compliance_prompt = prompt + "\n\nFirst, use set_compliance to record your compliance decision only."
        plan = self.reasoning.plan(
            prompt=compliance_prompt,
            obs=observation,
            selected_tools=["set_compliance"],
        )
        self.apply_plan(plan)

        enforcement_prompt = prompt + "\n\nNow use set_enforcement to record your enforcement decision only."
        plan = self.reasoning.plan(
            prompt=enforcement_prompt,
            obs=observation,
            selected_tools=["set_enforcement"],
        )
        self.apply_plan(plan)