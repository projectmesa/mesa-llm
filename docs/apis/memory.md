# Memory Module

The memory system in Mesa-LLM provides different types of memory implementations that enable agents to store and retrieve past events (conversations, observations, actions, messages, plans, etc.). Memory serves as the foundation for creating agents with persistent, contextual awareness that enhances their decision-making capabilities. The memory module contains two classes.

## Usage in Mesa Simulations

```python
from mesa_llm.llm_agent import LLMAgent
from mesa_llm.memory.st_lt_memory import STLTMemory

class MyAgent(LLMAgent):
   def __init__(self, model, reasoning, **kwargs):
      super().__init__(model, reasoning, **kwargs)

      # Override default memory with custom configuration
      self.memory = STLTMemory(
            agent=self,
            short_term_capacity=10,    # Store 10 recent experiences
            consolidation_capacity=3, # Consolidate when 13 total entries
            llm_model="openai/gpt-4o-mini",
            display=True,             # Display the memory entries in the console when they are added to the memory
            api_base=None,            # Set to a custom URL for self-hosted LLMs
      )
```

### Episodic memory with relevance scoring

`EpisodicMemory` retrieves the most useful memories by combining **importance**,
**recency**, and—when an `embedding_model` is configured—**relevance** (the
cosine similarity between a focal query and each memory's embedding), following
the Generative Agents retrieval design. Without an `embedding_model` the
relevance term is skipped and retrieval falls back to importance + recency.

```python
from mesa_llm.memory.episodic_memory import EpisodicMemory

self.memory = EpisodicMemory(
      agent=self,
      llm_model="openai/gpt-4o-mini",          # used to grade event importance
      embedding_model="openai/text-embedding-3-small",  # enables relevance
      relevance_weight=1.0,                     # weight of the relevance term
      importance_weight=1.0,
      recency_weight=1.0,
)

# Retrieval can be steered with an explicit focal query; when omitted, the most
# recent entry is used as the query.
self.memory.get_prompt_ready(query="where is the nearest food source?")
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
