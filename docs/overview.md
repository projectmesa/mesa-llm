# Overview
[Mesa-llm](https://github.com/mesa/mesa-llm) is a set of tools that integrates Large Language Models (LLMs) with [Agent-based modeling](https://en.wikipedia.org/wiki/Agent-based_model) using the Mesa framework. Agents use natural language to reason about their state and prompts.

Mesa-llm allows agents to generate decisions dynamically through natural-language reasoning. This makes it possible to explore more flexible, adaptive, and human-like agent behavior within a structured simulation.

Agents in mesa-llm are still built just like standard Mesa agents. They are created, scheduled, and executed using Mesa’s existing mechanisms. The only difference is that their decision-making process is delegated to a language model through a reasoning module. Mesa-llm does not replace Mesa.

### Mesa remains responsible for:
- Simulation execution
- Agent scheduling
- Time progression
- Data collection
- Visualization

Mesa-llm simply adds a new layer for agent reasoning on top of Mesa’s core functionality.

## How mesa-llm Fits into Mesa
Mesa-llm is designed to integrate seamlessly with Mesa, not to modify or replace it. All core simulation mechanics are still handled entirely by Mesa. mesa-llm focuses only on how agents reason.

## What Mesa Provides
Mesa remains responsible for the full simulation infrastructure, including:

**Model**
Defines the simulation lifecycle and global state.
Controls when and how the simulation advances through time.

**Agent**
The base class for all agents in a Mesa model.
Agents are registered with the model and managed through Mesa’s AgentSet.

**Scheduling and Activation**
Mesa controls when agents act using mechanisms such as `model.step()` and `agents.do()` / `agents.shuffle_do()`.

**Data Collection**
Mesa provides tools for collecting model-level and agent-level data over time.

**Visualization**
Mesa handles visualization through its browser-based visualization system.

## What mesa-llm Adds
mesa-llm extends Mesa by adding language-based reasoning capabilities to agents:

**LLMAgent**
A thin wrapper around Mesa’s base `Agent` class.
It behaves exactly like a normal Mesa agent in terms of scheduling and execution, but delegates decision-making to a language model.
```python
class MyAgent(LLMAgent):
    def step(self):
        plan = self.reasoning.plan(prompt, obs)
```
**Reasoning Modules**
Reasoning modules define how an agent thinks.
For example, `ReActReasoning` produces a reasoning trace and an action suggestion based on the agent’s prompt and observations.
```python
reasoning=ReActReasoning
```
**Optional Memory**
Memory components allow agents to retain information across simulation steps. This enables more consistent and contextual reasoning but is not required for all models.
```python
self.memory = STLTMemory(agent=self)
```

**Optional Tools**
Tools allow agents to **request** structured actions, when supported by the reasoning module and LLM backend.
```python
selected_tools=["some_tool"]
```

## Mesa vs mesa-llm (Quick Comparison)
| Feature | Mesa | mesa-llm |
|-------|------|----------|
| Core purpose | Agent-based modeling framework | Extension that adds LLM-powered reasoning |
| Agent definition | Defines `Model` and `Agent` classes | Introduces `LLMAgent` as a wrapper around `Agent` |
| Simulation control | Controls simulation execution and time | Controls how agents reason and decide |
| Scheduling | Provides scheduling and activation | Uses Mesa’s scheduling without modification |
| Data handling | Manages data collection and analysis | Adds optional memory for contextual reasoning |
| Visualization | Handles visualization and environments | Adds optional tools for structured agent actions |
| AI integration | Independent of AI or language models | Integrates external LLM providers |

## Core Concepts in mesa-llm

### 1. LLMAgent
`LLMAgent` is the central abstraction introduced by mesa-llm.

From Mesa’s perspective, an `LLMAgent` behaves exactly like a standard Mesa `Agent`:

1. It is created and registered in the same way
2. It is scheduled and activated using the same mechanisms

Its `step()` method is called during model execution.
The key difference lies in how decisions are made.
Instead of relying solely on hard-coded logic, an `LLMAgent` delegates its decision-making process to a language model through a reasoning module.

This design ensures that mesa-llm remains fully compatible with Mesa’s execution model while enabling language-based reasoning inside agents.

### 2. Reasoning Modules
In mesa-llm, reasoning is separated from the agent itself.

This separation is intentional:
- It keeps agent logic modular and extensible
- It allows different reasoning strategies to be swapped without changing agent code
- It makes experimentation with reasoning approaches easier

#### A reasoning module defines how an agent thinks, given:
- A prompt
- Observations
- Optional memory or tools

#### One example of a reasoning module is ReActReasoning, which produces:
- A natural-language reasoning trace
- A suggested action

The internal implementation of reasoning modules is abstracted away from the agent, allowing users to focus on modeling rather than LLM orchestration.

#### 3. Memory
Memory allows agents to retain information across simulation steps.

mesa-llm typically distinguishes between:
- Short-term memory, which stores recent observations or interactions
- Long-term memory, which stores more persistent contextual information

Memory is especially important for reasoning agents because:
- It enables consistent behavior over time
- It allows agents to reference past interactions
- It supports more coherent decision-making

Memory is optional in mesa-llm. Simple models may not require memory, while more complex scenarios can benefit significantly from it.

#### 4. Tools
Tools provide a structured way for agents to interact with their environment or perform specific actions.

Rather than relying solely on free-form text output, tools allow agents to:
- Trigger predefined functions
- Produce structured, machine-readable actions

Tools are not required for beginner models and are typically introduced in more advanced tutorials. They are most useful when agent actions need to be constrained or explicitly executed within the simulation.

#### 5. Model Execution
mesa-llm follows Mesa’s standard execution model without modification. The simulation still advances through discrete time steps using `model.step()`.
Each call to `model.step()` represents one unit of simulated time and triggers agent activation according to Mesa’s scheduling rules.

Agents created with mesa-llm are activated in the same way as standard Mesa agents. When an agent is activated, its `step()` method is called by the model.

The difference lies inside the agent’s `step()` method:
- Instead of executing only predefined logic,
- The agent constructs observations and prompts,
- Then reasons about its decision using a language model.

This design ensures that:
- Mesa remains responsible for time progression and execution order
- mesa-llm is responsible only for agent reasoning
As a result, mesa-llm models behave like standard Mesa models from an execution standpoint, while enabling agents to perform language-based reasoning at each step.

## Typical mesa-llm Model Structure
``` bash

Model
├── LLMAgent(s)
│   ├── Reasoning
│   │   └── (e.g. ReActReasoning)
│   ├── Memory (optional)
│   │   ├── Short-term memory
│   │   └── Long-term memory
│   └── Tools (optional)
│       └── Structured actions / function calls
└── Execution (Mesa)
    ├── model.step()
    └── Agent activation & scheduling

```
## Intended Use
Mesa-llm is designed for exploration, learning, and research, rather than for building production-grade autonomous systems.

### Who mesa-llm is for
#### Learners
Mesa-llm is well suited for users who want to understand how language-based reasoning can be integrated into agent-based models. Its modular design makes it easy to experiment with prompts, reasoning strategies, and agent behavior.

#### Researchers
The framework enables rapid prototyping of models that explore emergent behavior, decision-making, and interaction driven by language models. Researchers can easily modify reasoning components without changing the underlying simulation structure.

#### Experimentation and Prototyping
Mesa-llm is ideal for testing ideas, comparing reasoning strategies, and studying how different prompts or memory configurations influence agent behavior within controlled simulations.