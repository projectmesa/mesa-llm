import datetime

import mesa_llm.tools.inbuilt_tools  # noqa: F401, to register inbuilt tools

from .parallel_stepping import (
    enable_automatic_parallel_stepping,
    step_agents_parallel,
    step_agents_parallel_sync,
)
from .reasoning.reasoning import Observation, Plan
from .recording.record_model import record_model
from .tools.tool_manager import ToolManager

# Enable automatic parallel stepping when mesa_llm is imported
enable_automatic_parallel_stepping()

__all__ = [
    "Observation",
    "Plan",
    "ToolManager",
    "enable_automatic_parallel_stepping",
    "record_model",
    "step_agents_parallel",
    "step_agents_parallel",
    "step_agents_parallel_sync",
    "step_agents_parallel_sync",
]

__title__ = "Mesa-LLM"
__version__ = "0.2.0"
__license__ = "MIT"
_this_year = datetime.datetime.now(tz=datetime.UTC).date().year
__copyright__ = f"Copyright {_this_year} Project Mesa Team"
