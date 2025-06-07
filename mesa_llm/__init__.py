import datetime

import mesa_llm.tools.inbuilt_tools  # noqa: F401, to register inbuilt tools

from .reasoning import Observation, Plan
from .tools import ToolManager

__all__ = [
    "Observation",
    "Plan",
    "ToolManager",
]

__title__ = "Mesa-LLM"
__version__ = "0.0.2"
__license__ = "MIT"
_this_year = datetime.datetime.now(tz=datetime.UTC).date().year
__copyright__ = f"Copyright {_this_year} Project Mesa Team"
