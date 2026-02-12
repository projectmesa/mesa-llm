# Memory Module

The memory system in Mesa-LLM provides different types of memory implementations that enable agents to store and retrieve past events (conversations, observations, actions, messages, plans, etc.). Memory serves as the foundation for creating agents with persistent, contextual awareness that enhances their decision-making capabilities. The memory module contains two classes.

## Usage in Mesa Simulations

```python
from mesa_llm.llm_agent import LLMAgent
from mesa_llm.memory.st_lt_memory import STLTMemory

class MyAgent(LLMAgent):
   def __init__(self, model, api_key, reasoning, **kwargs):
      super().__init__(model, api_key, reasoning, **kwargs)

      # Override default memory with custom configuration
      self.memory = STLTMemory(
            agent=self,
            short_term_capacity=10,    # Store 10 recent experiences
            consolidation_capacity=3, # Consolidate when 13 total entries
            api_key=api_key,
            llm_model="openai/gpt-4o-mini",
            display=True            # Display the memory entries in the console when they are added to the memory
      )
```

## Core memory interfaces

```{eval-rst}
.. automodule:: mesa_llm.memory.memory
   :members:
   :undoc-members:
   :show-inheritance:
```

## Memory implementations

```{eval-rst}
.. automodule:: mesa_llm.memory.st_lt_memory
   :members:
   :undoc-members:
   :show-inheritance:

.. image:: st_lt_consolidation_explained.png
   :alt: ST-LT Memory Consolidation Explained
   :align: center

.. automodule:: mesa_llm.memory.st_memory
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: mesa_llm.memory.lt_memory
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: mesa_llm.memory.episodic_memory
   :members:
   :undoc-members:
   :show-inheritance:
```
