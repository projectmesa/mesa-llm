# Overview
[Mesa-llm](https://github.com/mesa/mesa-llm) is an extension of [Mesa](https://github.com/mesa) that enables language-model-based reasoning inside agents. It does not modify Mesa’s execution model, scheduling, or environments.
Agents created with mesa-llm are standard Mesa agents. The only difference is that their decision-making logic is delegated to a reasoning component backed by an LLM.

## Separation
### What Mesa Provides
- Agent
- Model
- AgentSet & Scheduling
- Space / Environment
- Time Progression
- Data Collection
- Visualization

## What Mesa-llm Adds
- Language Model Based reasoning inside `Agent`

## Core Mesa-llm Concepts
#### 1. LLMAgent
`LLMAgent` is the core abstraction introduced by mesa-llm.
From Mesa’s perspective, an LLMAgent behaves exactly like a standard Mesa Agent:
- It is created and registered in the same way.
- It is scheduled and activated using Mesa’s normal execution flow.

The difference lies in how decisions are made.
Instead of relying only on hard-coded logic, an LLMAgent delegates part (or all) of its decision-making process to a language model through a reasoning module.

``` python
class MyAgent(LLMAgent):
    def step(self):
        plan = self.reasoning.plan(prompt, obs)
```
#### 2. Reasoning
A reasoning module defines how an agent thinks, given:
- A prompt.
- Observations from the model or environment.
- Optional memory and tools.

Reasoning modules are attached to agents to encapsulate how language-based reasoning is performed. While mesa-llm currently provides `ReActReasoning`, the reasoning logic is kept separate from the agent to allow extensibility without changing agent structure.

#### 3. Memory
Memory allows agents to retain information across simulation steps.
In mesa-llm, memory is useful for reasoning-based agents. It enables agents to:
- Reference past observations or interactions.
- Maintain more consistent behavior over time.
- Produce more coherent and contextual reasoning.

Memory is optional in mesa-llm. When present, it allows agents to retain information across steps and reason with context. When absent, agents reason purely from the current observation and prompt.

#### 4. Tools
Tools provide a structured way for agents to express or request actions when the simulation supports action execution.

Tools are optional and typically become relevant only when:
- An environment exists (e.g. a grid).
- Actions can be meaningfully executed.

In introductory models, action suggestions may appear only as part of the reasoning trace and are not executed. Mesa-llm does not introduce new space or environment abstractions; it relies entirely on Mesa’s existing space modules.

#### 5. Execution Model
Mesa-llm follows Mesa’s standard execution model.
The simulation advances through calls to `model.step()`, and agents are activated using Mesa’s scheduling mechanisms.

When an agent is activated, its `step()` method is called.
In mesa-llm, this method typically includes language-based reasoning before deciding what to do next.

