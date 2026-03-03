import random

from agents import EmperorLLMAgent
from mesa import Model
from mesa.datacollection import DataCollector
from mesa.discrete_space.grid import OrthogonalMooreGrid
from mesa_llm.reasoning.react import ReActReasoning


LLM_MODEL = ""  # Change this to your preferred model


class EmperorLLMModel(Model):
    """The Emperor's Dilemma Model with LLM-powered agents.

    Agents use language model reasoning to decide whether to publicly comply
    with an unpopular norm and whether to enforce it on their neighbors,
    replacing the original mathematical threshold model.
    """

    def __init__(
        self,
        width=5,
        height= 5,
        fraction_true_believers=0.05,
        k=0.125,
        homophily=False,
        rng=None,
    ):
        """Initializes the EmperorLLMModel.

        Args:
            width: Width of the grid. Defaults to 10
            height: Height of the grid. Defaults to 10.
            fraction_true_believers: Fraction of true believers. Defaults to 0.05.
            k : Enforcement cost. Defaults to 0.125.
            homophily : Whether to cluster believers. Defaults to False.
            rng : Random seed. Defaults to None.
        """
        super().__init__(rng=rng)

        self.width = width
        self.height = height
        self.fraction_true_believers = fraction_true_believers
        self.k = k
        self.homophily = homophily

        self.grid = OrthogonalMooreGrid(
            (width, height), torus=True, capacity=1, random=self.random
        )

        self.datacollector = DataCollector(
            model_reporters={
                "Compliance": compute_compliance,
                "Enforcement": compute_enforcement,
                "False Enforcement": compute_false_enforcement,
            }
        )

        self.init_agents()
        self.running = True
        self.datacollector.collect(self)

    def init_agents(self):
        """Initializes LLM agents and places them on the grid."""
        num_agents = self.width * self.height
        num_believers = int(num_agents * self.fraction_true_believers)

        all_coords = [(x, y) for x in range(self.width) for y in range(self.height)]
        believer_coords = set()

        if self.homophily:
            center_x = self.random.randint(0, self.width - 1)
            center_y = self.random.randint(0, self.height - 1)
            start_x = center_x - int(num_believers**0.5) // 2
            start_y = center_y - int(num_believers**0.5) // 2
            for i in range(num_believers):
                bx = (start_x + (i % int(num_believers**0.5 + 1))) % self.width
                by = (start_y + (i // int(num_believers**0.5 + 1))) % self.height
                believer_coords.add((bx, by))
        else:
            believer_coords = set(random.sample(all_coords, num_believers))

        for x, y in all_coords:
            if (x, y) in believer_coords:
                private_belief = 1
                conviction = 1.0
                system_prompt = """You are a true believer in the Emperor's norm.
You genuinely think the Emperor's hat is magnificent and beautiful.
You feel it is your duty to uphold and defend this norm in your community.
You will comply with and enforce the norm because you truly believe in it."""
            else:
                private_belief = -1
                conviction = random.uniform(0.01, 0.38)
                system_prompt = f"""You are a secret dissenter in a society ruled by a powerful norm.
Deep down, you think the Emperor's hat is ugly and the norm is absurd.
You hold this private belief with conviction level {conviction:.2f} out of 1.0.
However, you are aware that openly resisting can be dangerous and costly.
You must carefully weigh your private beliefs against the social pressure around you."""

            agent = EmperorLLMAgent(
                model=self,
                reasoning=ReActReasoning,
                llm_model=LLM_MODEL,
                system_prompt=system_prompt,
                internal_state=f"Private belief: {'support' if private_belief == 1 else 'reject'} the norm.",
                private_belief=private_belief,
                conviction=conviction,
                k=self.k,
            )

            cell = self.grid[(x, y)]
            agent.cell = cell
            agent.pos = (x, y)
            self.agents.add(agent)

    def step(self):
        """Executes one step — all agents reason and decide, then data is collected."""
        self.agents.shuffle_do("step")
        self.datacollector.collect(self)


def compute_compliance(model):
    """Computes the fraction of agents publicly complying with the norm."""
    if not model.agents:
        return 0
    return sum(1 for a in model.agents if a.compliance == 1) / len(model.agents)


def compute_enforcement(model):
    """Computes the fraction of agents actively enforcing the norm."""
    if not model.agents:
        return 0
    return sum(1 for a in model.agents if a.enforcement == 1) / len(model.agents)


def compute_false_enforcement(model):
    """Computes the false enforcement rate — disbelievers who enforce the norm."""
    disbelievers = [a for a in model.agents if a.private_belief == -1]
    if not disbelievers:
        return 0
    return sum(1 for a in disbelievers if a.enforcement == 1) / len(disbelievers)
