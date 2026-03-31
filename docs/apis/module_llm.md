# ModuleLLM

ModuleLLM provides a unified interface for integrating Large Language Models from multiple providers into Mesa-LLM agents. It abstracts away provider-specific implementation details while offering both synchronous and asynchronous generation capabilities with support for function calling, structured outputs, and automatic retry logic.

## Basic LLM Setup

In your .env file, set the API key for the LLM provider, then in your python file, call the ModuleLLM class with the desired model and system prompt.

```
# .env
OPENAI_API_KEY=your-api-key
ANTHROPIC_API_KEY=your-api-key
```

```python
# my_agent.py
from mesa_llm.module_llm import ModuleLLM

# Initialize with specific provider and model
llm = ModuleLLM(
   llm_model="openai/gpt-4o",
   system_prompt="You are a helpful simulation agent."
)

# Generate response
response = llm.generate("What should I do next in this situation?")
```

## Custom API Endpoints

If you are using a self-hosted or remote LLM server (e.g., Ollama on another machine, vLLM, LM Studio), you can specify a custom API endpoint using the `api_base` parameter:

```python
from mesa_llm.module_llm import ModuleLLM

# Connect to a remote Ollama instance
llm = ModuleLLM(
   llm_model="ollama_chat/llama3.2",
   system_prompt="You are a helpful agent.",
   api_base="http://192.168.1.100:11434",
)

# Connect to a local LM Studio server
llm = ModuleLLM(
   llm_model="openai/my-local-model",
   system_prompt="You are a helpful agent.",
   api_base="http://localhost:1234/v1",
)
```

> **Note:** For standard Ollama running locally on the default port, `api_base` is optional — it defaults to `http://localhost:11434`. For cloud providers (OpenAI, Anthropic, Google, Groq, etc.), `api_base` is not needed as the URLs are automatically resolved by the LiteLLM library.

The `api_base` parameter is supported across all layers of Mesa-LLM: `ModuleLLM`, `LLMAgent`, and all `Memory` subclasses (`STLTMemory`, `EpisodicMemory`, `LongTermMemory`).

## Tool Integration

```python
from mesa_llm.tools.tool_manager import ToolManager

tool_manager = ToolManager()
llm = ModuleLLM(llm_model="openai/gpt-4o")

# Generate with tool calling
response = llm.generate(
   prompt="Move to a better location",
   tool_schema=tool_manager.get_all_tools_schema(),
   tool_choice="auto"
)

# Handle tool calls
if response.choices[0].message.tool_calls:
   tool_manager.call_tools(agent=agent, llm_response=response.choices[0].message)
```

## Asynchronous Usage

```python
async def generate_plan(llm, prompt, tools):
   """Generate response asynchronously for parallel processing"""
   response = await llm.agenerate(
      prompt=prompt,
      tool_schema=tools,
      tool_choice="required"
   )
   return response

# Use in parallel agent execution
responses = await asyncio.gather(*[
   generate_plan(agent.llm, prompt, tools) for agent in agents
])
```

## Structured Output

```python
from pydantic import BaseModel

class AgentDecision(BaseModel):
   reasoning: str
   action: str
   confidence: float

response = llm.generate(
   prompt="Analyze the situation and decide your action",
   response_format=AgentDecision
)

# Parse structured response
decision = AgentDecision.parse_raw(response.choices[0].message.content)
```

## Integration with LLMAgent

```python
class MyAgent(LLMAgent):
   def __init__(self, model, **kwargs):
      super().__init__(
            model=model,
            llm_model="anthropic/claude-3-sonnet",  # Automatically creates ModuleLLM
            system_prompt="Custom agent behavior instructions",
            **kwargs
      )

   def step(self):
      # Access LLM through self.llm
      response = self.llm.generate("What should I do?")
```

## API Reference

```{eval-rst}
.. automodule:: mesa_llm.module_llm
   :members:
   :undoc-members:
   :show-inheritance:
```
