# Negotiation Model Tutorial (mesa-llm)

## About This Tutorial

This tutorial is inspired by the official
[Mesa-llm Negotiation Example](https://github.com/mesa/mesa-llm/tree/main/examples/negotiation)

The goal here is **not to replicate** the original example, but to present a **simplified and tutorial-friendly version** that focuses on reasoning structure and agent interaction.

Key differences from the original example include:
- A reduced number of agents
- No enforced negotiation protocol
- No spatial environment or grid
- Emphasis on understanding ReAct reasoning output

This makes the example easier to follow for new users while still demonstrating core mesa-llm concepts.

## Model Description

This `Model` simulates a basic negotiation scenario involving:
- One seller agent with a minimum acceptable price
- Two buyer agents, each with a different budget

### Each buyer:

- Has a different budget
- Reasons independently using `ReActReasoning`
- Buyers do not coordinate with each other

### Seller Agent

- Inherits from `LLMAgent`
- Uses `ReActReasoning` to reason about negotiation decisions
- Considers its minimum acceptable price and the current simulation step


## Tutorial Setup
Ensure you are using Python 3.12 or later.

## Install mesa-llm and required packages

Install mesa-llm

```bash
pip install -U mesa-llm
```

## Why This Model Is Non-Spatial

Negotiation is a conceptual process rather than a spatial one.
Buyers and sellers negotiate based on preferences, constraints, and reasoning—not physical position.
To keep this tutorial simple and focused, no grid or physical environment is introduced.
Adding a grid would increase complexity without improving the clarity of the negotiation logic.


## Model Execution
At each model step:
- The model advances one step
- All agents are activated using `shuffle_do("step")`.
Each agent generates a reasoning plan using `ReActReasoning` and applies it.
The console output displays the internal reasoning traces.

## Agent Messaging
In negotiation scenarios, agents need to exchange information.
mesa-llm supports agent-to-agent messaging, allowing agents to communicate
explicitly rather than relying on shared state.

In this tutorial:
- The seller sends its minimum acceptable price to buyers
- Buyers read incoming messages before reasoning
- Messaging is used only for demonstration purposes

## Creating the Seller class (Agent)
Using the previously imported dependencies, we define the agent class:
The Seller agent inherits from LLMAgent and uses `ReActReasoning` to decide how to respond during negotiation.
It reasons about the current simulation step and its minimum acceptable price.

``` python
# Import Dependencies
from mesa.model import Model
from mesa_llm.llm_agent import LLMAgent
from mesa_llm.reasoning.react import ReActReasoning
from mesa_llm.memory.st_lt_memory import STLTMemory

# ---------------- SELLER ----------------
class Seller(LLMAgent):
    """
    Seller agent participating in a simple negotiation scenario.

    The seller has a minimum acceptable price and communicates this
    information to other agents using the messaging system.
    """

    def __init__(self, *args, **kwargs):
        # Initialize the agent using Mesa's base Agent initialization.
        # `LLMAgent` is a wrapper around Mesa's Agent, so all arguments
        # are forwarded to ensure proper registration in the model.
        super().__init__(*args, **kwargs)

        # Attach memory to allow the LLM to maintain context
        # across multiple simulation steps.
        self.memory = STLTMemory(
            agent=self,
            llm_model="ollama/llama3",
        )

    def step(self):
        """
        Called once per model step.

        The seller:
        1. Sends its minimum acceptable price to other agents
        2. Builds an observation for the reasoning module
        3. Uses ReActReasoning to generate a reasoning trace
        """

        # Send the seller's minimum acceptable price to all other agents.
        # This demonstrates explicit agent-to-agent communication.
        for agent in self.model.agents:
            if agent is not self:
                self.send_message(
                    f"My minimum acceptable price is {self.internal_state['min_price']}.",
                    [agent]
                    )

        # Observation passed to the reasoning module.
        # Includes the current step and seller-specific constraints.
        observation = {
            "step": self.model.steps,
            "min_price": self.internal_state["min_price"]
        }

        # Prompt describing the seller's role in the negotiation.
        # The LLM is asked to reason about the situation,
        # not to execute any concrete actions.
        prompt = """
        You are a seller negotiating the price of a single item.
        Reason about your position given your minimum acceptable price.
        """

        # Generate a reasoning plan using the configured reasoning strategy.
        plan = self.reasoning.plan(
            prompt=prompt,
            obs=observation
        )

        # Apply the plan.
        # In this tutorial, actions are not executed and appear only
        # as part of the reasoning trace.
        self.apply_plan(plan)
```

## Creating the Buyer Class
### Each buyer:
- Has a different budget
- Reasons independently using `ReActReasoning`
- Buyers do not coordinate with each other

``` python
# ---------------- BUYER ----------------
class Buyer(LLMAgent):
    """
    Buyer agent participating in the negotiation.

    The buyer has a fixed budget and reasons about whether a potential
    purchase is worthwhile based on its constraints and any messages
    received from other agents.
    """

    def __init__(self, *args, **kwargs):
        # Standard LLMAgent initialization.
        # This ensures the buyer is properly registered in the Mesa model.
        super().__init__(*args, **kwargs)

        # Attach memory so the LLM can retain context across steps.
        # This is optional, but useful for observing evolving reasoning.
        self.memory = STLTMemory(
            agent=self,
            llm_model="ollama/llama3",

        )

    def step(self):
        """
        Called once per model step.

        The buyer reasons about the negotiation using its budget
        and any information stored in memory (including messages
        sent by other agents).

        """
        # Observation passed to the reasoning module.
        # Messages sent by other agents are already stored in memory
        # and implicitly available to the reasoning process.

        observation = {

            "step": self.model.steps,
            "budget": self.internal_state["budget"],
        }

        prompt = """
        You are a buyer negotiating to purchase a single item.
        You have a fixed budget.

        Use any relevant information you have received so far
        when reasoning about the negotiation.

        """

        plan = self.reasoning.plan(
            prompt=prompt,
            obs=observation
        )
        self.apply_plan(plan)
```
## Creating Negotiation Model
The NegotiationModel sets up and runs the negotiation between buyers and a seller.
It creates the agents, assigns their roles, and controls when each agent acts.

### How it works:
- Inherits from Mesa’s `Model` class.
- Creates one seller agent with a minimum acceptable price.
- Creates two buyer agents with different budgets.
- Assigns `ReActReasoning` to all agents so they can reason using an LLM.
- Uses `shuffle_do("step")` to let all agents act once per model step.
- Prints the current step number to track simulation progress.
- Repeats this process for a fixed number of steps in the main loop.
``` python
# ---------------- MODEL ----------------
class NegotiationModel(Model):
    """
    Model coordinating a simple negotiation scenario.

    The model:
    - Creates one seller and two buyers
    - Assigns different constraints to each agent
    - Advances the simulation step by step
    """

    def __init__(self, llm_model="ollama/llama3", seed=None):
        # Initialize the Mesa model.
        # The seed is optional and can be used for reproducibility.
        super().__init__(seed=seed)

        # Create a single seller agent with a minimum acceptable price.
        # Agents are created using Mesa's create_agents() helper.
        Seller.create_agents(
            model=self,
            n=1,
            reasoning=ReActReasoning,
            llm_model=llm_model,
            system_prompt="You are a seller.",
            internal_state={"min_price": 60},
        )

        # Create the first buyer with a higher budget.
        Buyer.create_agents(
            model=self,
            n=1,
            reasoning=ReActReasoning,
            llm_model=llm_model,
            system_prompt="You are a buyer.",
            internal_state={"budget": 100},
        )

        # Create the second buyer with a lower budget.
        Buyer.create_agents(
            model=self,
            n=1,
            reasoning=ReActReasoning,
            llm_model=llm_model,
            system_prompt="You are a buyer.",
            internal_state={"budget": 70},
        )

    def step(self):
        """
        Advance the model by one step.

        At each step:
        - All agents are activated once
        - Activation order is randomized using `shuffle_do`
        - Each agent performs its reasoning for this step
        """

        # Print the current step number for clarity in the output
        print(f"\n--- Model step {self.steps} ---")

        # Activate all agents in random order
        self.agents.shuffle_do("step")
```
## Running the Model
- The `Model` runs for a few steps using a loop.
- In each step, all agents think and act once.
- The reasoning output is printed to the console.
``` python
# ---------------- RUN ----------------
if __name__ == "__main__":
    # Create an instance of the negotiation model
    model = NegotiationModel()

    # Run the model for a fixed number of steps
    # Each step activates all agents once and prints their reasoning output
    for _ in range(3):
        model.step()
```

## Understanding the Output
Below is an example of the reasoning output produced by `ReActReasoning`:
``` bash
Message                                                                           │
│    └── message : My minimum acceptable price is 60.                                 │
│    └── sender : LLMAgent 1                                                          │
│    └── recipients : [<__main__.Buyer object at 0x13d015450>]                        │
│                                                                                     │
[Plan]                                                                              │
│    └── reasoning : As I am at step 1, I don't have any information about the        │
│ environment or my position. Since my short-term memory is empty and my long-term    │
│ memory doesn't provide any relevant context, I will focus on getting more           │
│ information about my surroundings. The current observation suggests that the        │
│ minimum price is 60, but this might not be directly related to my actions.          │
│ Therefore, I decide to move one step in a random direction to gather more           │
│ information about the environment and potentially find some clues or objects.       │
│    └── action : move_one_step
```
## About Actions in the Output
`ReActReasoning` produces both a reasoning trace and an action suggestion.

In this tutorial:
- Actions are not executed
- Actions appear only as part of the reasoning trace
- Action execution is introduced in later tutorials

## Exercises
Try the following exercises to better understand agent communication and reasoning:

1. **Modify the seller’s message**
   Change the content of the message sent by the seller and observe how buyer
   reasoning changes.

2. **Add another buyer**
   Create a third buyer agent with a different budget and compare its reasoning
   with the existing buyers.

3. **Change buyer budgets**
   Adjust buyer budgets and observe how this affects negotiation-related reasoning.

4. **Increase the number of steps**
   Run the `Model` for more steps and observe how agent messaging influences
   reasoning over time.

