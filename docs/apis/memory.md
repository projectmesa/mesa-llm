# Memory module

The memory system in Mesa-LLM provides different types of memory implementations that enable agents to store and retrieve past events (conversations, observations, actions, messages, plans, etc.). Memory serves as the foundation for creating agents with persistent, contextual awareness that enhances their decision-making capabilities. The memory module contains two classes.

## Memory Base Classes

```{eval-rst}
.. automodule:: mesa_llm.memory.memory
   :members:
   :undoc-members:
   :show-inheritance:
```

## Short-Term Memory

```{eval-rst}
.. automodule:: mesa_llm.memory.st_memory
   :members:
   :show-inheritance:
```

## Long-Term Memory

```{eval-rst}
.. automodule:: mesa_llm.memory.lt_memory
   :members:
   :show-inheritance:
```

## Short-Term/Long-Term (STLT) Memory

```{eval-rst}
.. automodule:: mesa_llm.memory.st_lt_memory
   :members:
   :show-inheritance:
```

## Episodic Memory

```{eval-rst}
.. automodule:: mesa_llm.memory.episodic_memory
   :members:
   :show-inheritance:
```