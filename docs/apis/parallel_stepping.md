# Parallel Stepping

The parallel stepping system in Mesa-LLM is made for simultaneous execution of multiple agents within a single simulation step, to improve performance for LLM-based simulations. It provides both automatic and manual parallel execution modes with support for async/await patterns and threading-based approaches.

The system monkey-patches Mesa's `AgentSet.shuffle_do()` method to automatically detect parallel stepping opportunities:

- **Model Flag Detection**: Activates when `model.parallel_stepping = True`
- **Transparent Integration**: Works with existing Mesa simulation code without modifications
- **Backward Compatibility**: Falls back to sequential execution when parallel stepping is disabled

There are two modes of parallel stepping:

- Asyncio Mode (Default)

> Uses Python's asyncio event loop to execute multiple agent `astep()` methods concurrently. This is ideal for I/O-bound operations like LLM API calls where agents spend time waiting for responses. It allows simultaneous LLM requests with minimal CPU overhead, scales to hundreds of agents, and handles event loop management automatically.

- Threading Mode

> Uses ThreadPoolExecutor to run agent `step()` methods in parallel threads. Suitable for CPU-bound tasks or when asyncio integration is problematic. It automatically manages worker thread lifecycle, executes both sync and async methods in threads, provides fallback compatibility for agents without async methods, and offers configurable thread pool size for resource management.

## Usage in Mesa Simulations

### Parallel Execution in MODEL file

```python
from mesa_llm.parallel_stepping import enable_automatic_parallel_stepping
from mesa import Model

class MyModel(Model):
   def __init__(self):
      super().__init__()
      # Enable automatic parallel stepping
      self.parallel_stepping = True
      enable_automatic_parallel_stepping(mode="asyncio")

      # Add agents as normal
      for i in range(100):
            agent = MyLLMAgent(self, ...)
            self.agents.add(agent)

   def step(self):
      # This will automatically execute agents in parallel
      self.agents.shuffle_do("step")
```

### Agent Implementation for Parallel Execution

```python
from mesa_llm.llm_agent import LLMAgent

class ParallelAgent(LLMAgent):
   def step(self):
      """Synchronous step - will be executed in threads if parallel"""
      obs = self.generate_obs()
      plan = self.reasoning.plan(obs=obs)
      self.apply_plan(plan)

   async def astep(self):
      """Asynchronous step - preferred for parallel execution"""
      obs = self.generate_obs()
      # Use async planning for true concurrency
      plan = await self.reasoning.aplan(
            prompt=self.step_prompt,
            obs=obs
      )
      self.apply_plan(plan)
```

### Manual Parallel Control

```python
from mesa_llm.parallel_stepping import step_agents_parallel_sync

class CustomModel(Model):
   def step(self):
      # Manual parallel execution
      citizen_agents = [a for a in self.agents if isinstance(a, Citizen)]
      cop_agents = [a for a in self.agents if isinstance(a, Cop)]

      # Execute different agent types in parallel
      step_agents_parallel_sync(citizen_agents)
      step_agents_parallel_sync(cop_agents)
```

### Mode Selection

```python
from mesa_llm.parallel_stepping import enable_automatic_parallel_stepping

# For I/O-bound LLM operations (recommended)
enable_automatic_parallel_stepping(mode="asyncio")

# For CPU-bound operations or compatibility
enable_automatic_parallel_stepping(mode="threading")
```

## API Reference

```{eval-rst}
.. automodule:: mesa_llm.parallel_stepping
   :members:
   :undoc-members:
   :show-inheritance:
```
