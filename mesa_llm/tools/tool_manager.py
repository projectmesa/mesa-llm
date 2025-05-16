from typing import Callable, Dict, Any
import inspect
import json

class ToolManager:
    def __init__(self):
        self.tools: Dict[str, Callable] = {}

    def register(self, fn: Callable):
        #Register a tool function by name
        name = fn.__name__
        self.tools[name] = fn

    def get_schema(self) -> list[dict]:
        #Return schema (have to make it liteLLM compatible)
        schema = []
        for name, fn in self.tools.items():
            sig = inspect.signature(fn)
            properties = {}
            required = []
            for param in sig.parameters.values():
                properties[param.name] = {
                    "type": "string",  #I have to use type-checking here
                    "description": f"{param.name} parameter"
                }
                if param.default == inspect.Parameter.empty:
                    required.append(param.name)
            schema.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": fn.__doc__ or "",
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required
                    }
                }
            })
        return schema

    def call(self, name: str, arguments: dict) -> str:
        """Call a registered tool with validated args"""
        if name not in self.tools:
            raise ValueError(f"Tool '{name}' not found")
        return self.tools[name](**arguments)

    def has_tool(self, name: str) -> bool:
        return name in self.tools
    
'''Example Usage:

tool_manager = ToolManager()

@tool_manager.register
def get_current_weather(location, unit="fahrenheit"):
    """Get the current weather in a given location."""
    return json.dumps({"location": location, "temperature": "22", "unit": unit})

'''