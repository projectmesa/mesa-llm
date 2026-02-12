# Creating Your First mesa-llm Model

## Tutorial Overview
This tutorial introduces mesa-llm by walking through the construction of a simple language-driven agent model built on top of Mesa. Mesa-llm enables agents to reason using natural language while preserving Mesa’s standard execution model. If it's your first time using mesa, we suggest starting with the classic [creating your first model tutorials](https://mesa.readthedocs.io/latest/tutorials/0_first_model.html) before diving into mesa-llm.

The goal of this tutorial is **not** to build a complex simulation or environment.
Instead, it focuses on the **core idea** behind mesa-llm:

> How language-based reasoning can be embedded into Mesa’s standard agent execution workflow.

By the end of this tutorial, you will understand:
- What `LLMAgent` is
- How mesa-llm integrates with Mesa models
- How agents perform language-based reasoning at each step
- Why some reasoning strategies also suggest actions
- How to structure a clean, extensible starting model

## About mesa-llm

[Mesa LLM](https://github.com/mesa/mesa-llm) is a set of tools that integrates Large Language Models (LLMs) with [Agent-based modeling](https://en.wikipedia.org/wiki/Agent-based_model) using the Mesa framework. Agents use natural language to reason about their state and prompts.

This approach is particularly useful for exploring how complex or emergent
behavior can arise from language-driven agents in simulated systems.
By combining Mesa’s structured simulation framework with LLM-based reasoning,
Mesa-LLM allows researchers and developers to experiment with more flexible,
human-like agent behavior.

While mesa-llm adds:
- Prompt-driven reasoning
- Reasoning strategies (e.g. ReAct)
- Optional memory components for contextual reasoning across steps
- Integration with LLM backends such as OpenAI, Ollama, and others.

## Model Description
The model consists of:
- A Mesa `Model`
- One or more `LLMAgent` instances

At each simulation step:
1. The model activates all agents
2. Each agent constructs a natural-language prompt
3. The agent reasons using an LLM
4. The reasoning output is printed

This tutorial **focuses on reasoning output only**.
Action execution and environments are intentionally deferred to later tutorials.

## Tutorial Setup
Create and activate a virtual environment. Python version 3.12 or higher is required.

## Install mesa-llm and required packages

Install mesa-llm

```bash
pip install -U mesa-llm
```

Mesa-llm pre-releases can be installed with:
```bash
pip install -U --pre mesa-llm
```

You can also use pip to install the GitHub version:
```bash
pip install -U -e git+https://github.com/mesa/mesa-llm.git#egg=mesa-llm
```

Or any other (development) branch on this repo or your own fork:
```bash
pip install -U -e git+https://github.com/YOUR_FORK/mesa-llm@YOUR_BRANCH#egg=mesa-llm
```

## Building the Model
After mesa-llm is installed, a 'model' can be built.
This tutorial can be followed in a regular Python script or in a [Jupyter](https://jupyter.org/) notebook.


Start Jupyter from the command line:
 ```bash
 jupyter lab
 ```

Create a new notebook named example.ipynb or whatever you want.

## Important Dependencies
This section includes the dependencies needed for the tutorial.
```python
from mesa.model import Model
from mesa_llm.llm_agent import LLMAgent
from mesa_llm.reasoning.react import ReActReasoning
from mesa_llm.memory.st_lt_memory import STLTMemory
```

## Creating the Agent
We begin by defining a minimal agent that inherits from 'LLMAgent'.

Unlike traditional Mesa agents, 'LLMAgent' delegates its reasoning to a language model through a configurable reasoning strategy.
In this tutorial, we use ReActReasoning, which produces both a reasoning trace and a suggested action.

Using the previously imported dependencies, we define the agent class:
```python
# ---------------- AGENT ----------------
class SimpleAgent(LLMAgent):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.memory = STLTMemory(
            agent=self,
            llm_model="ollama/llama3",
            display=True
        )

    def step(self):
        observation = {"step": self.model.steps}

        prompt = """
        This is a new simulation step.
        Based on the current step number, explain how you reason about
        the situation and what you might consider doing next.
        Focus on describing your reasoning process clearly.
        """

        plan = self.reasoning.plan(
            prompt=prompt,
            obs=observation,
            selected_tools=[]
        )

        print(plan)
```

## Create the Model
The 'model' manages agent creation and advances the simulation.

'Agent' are created using the `create_agents()` helper provided by Mesa.
mesa-llm integrates with this mechanism by allowing `LLMAgent` to be used
wherever a standard Mesa `Agent` is expected.

`LLMAgent` is a thin wrapper around Mesa’s base `Agent` class, which is why
agent initialization calls `super().__init__(*args, **kwargs)` to ensure
proper registration with Mesa’s internal `AgentSet`.

The SimpleModel class is created with the following code:
```python
# ---------------- MODEL ----------------
class SimpleModel(Model):

    def __init__(self, seed=None):
        super().__init__(seed=seed)

        SimpleAgent.create_agents(
            model=self,
            n=1,
            reasoning=ReActReasoning,
            llm_model="ollama/llama3",
            system_prompt="""
            You are an LLM-powered agent used to demonstrate how LLMAgent works
            inside a Mesa simulation.
            Your role is to reason about the current simulation step and explain
            your thinking process clearly.

            """,
            internal_state=""
        )

    def step(self):
        print(f"\nModel step {self.steps}")
        self.agents.shuffle_do("step")
```

## Running the Model
To run the simulation:

```python
# ---------------- RUN ----------------
if __name__ == "__main__":
    model = SimpleModel()

    for _ in range(3):
        model.step()
```
Each call to 'model.step()' activates all agents once and prints their language-based reasoning.

- An example output from running the 'model' is shown below

```bash
╭─ Step 1 | SimpleAgent 1 ──────────────────────────────────────────────────────────────────────────────╮
│                                                                                                       │
│ [Plan]                                                                                                │
│    └── reasoning : Since I am in the initial state and there is no information in my long-term        │
│ memory, I will base my decision on my current observation. The 'step' parameter indicates that this   │
│ is the first step, so I should explore the environment to get a better understanding of it. I will    │
│ decide to move one step in a random direction.                                                        │
│    └── action : move_one_step                                                                         │
╰────────────────────────────────────────────────

```

## About Actions in the Output
This is expected behavior when using ReActReasoning, which always produces both reasoning and an action suggestion as part of its design.

### Important Note:
- In this introductory tutorial, action suggestions are not executed.
- Actions are shown only as part of the reasoning trace.
- Environments and action execution are introduced in later tutorials.

## Exercises

Try the following small exercises to better understand how agent reasoning works in this model:

1. **Modify the prompt**
   Change the prompt passed to the agent and observe how the reasoning trace changes.
   For example, encourage the agent to be more cautious or more verbose.

2. **Add another agent**
   Create a second agent with a different initial `internal_state` and compare how
   their reasoning differs during the same model steps.

3. **Extend the observation**
   Add an additional value to the observation dictionary passed to the reasoning module
   and see how it affects the agent’s reasoning output.

4. **Increase the number of steps**
   Run the model for more steps and observe how the reasoning evolves over time.

In this tutorial, actions are not executed and are shown only as part of the reasoning
trace. Later tutorials will introduce environments and action execution.






