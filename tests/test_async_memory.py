import asyncio
import logging
import os
import time
import unittest
from unittest.mock import MagicMock, patch

from mesa_llm.llm_agent import LLMAgent
from mesa_llm.parallel_stepping import step_agents_parallel
from mesa_llm.reasoning.reasoning import Reasoning
from tests.conftest import build_llm_response

logger = logging.getLogger(__name__)

os.environ["OPENAI_API_KEY"] = "test"


class MockModel:
    def __init__(self):
        self.steps = 1
        self.agents = []

    def register_agent(self, agent):
        self.agents.append(agent)


class MockReasoning(Reasoning):
    def plan(self, prompt, obs=None, ttl=1, selected_tools=None, tool_calls="auto"):
        return MagicMock()

    async def aplan(
        self, prompt, obs=None, ttl=1, selected_tools=None, tool_calls="auto"
    ):
        return MagicMock()


class TestAsyncMemoryFix(unittest.IsolatedAsyncioTestCase):
    @patch("mesa_llm.module_llm.ModuleLLM.agenerate")
    async def test_parallel_memory_consolidation(self, mock_agenerate):
        """
        Verifies that long-term memory consolidation runs concurrently when agents
        are stepped using the async parallel stepping pipeline.

        We patch ModuleLLM.agenerate() with an artificial 1-second async delay.
        Three agents are configured so that each triggers memory consolidation
        during its async post-step.

        Expected behavior:
        - If consolidation is truly async and non-blocking, total runtime should
          be close to ~1 second (all three consolidations overlap).
        - If consolidation is blocking or serial, runtime would be ~3 seconds.

        This test asserts the total wall-clock time is below 2 seconds, providing
        behavioral proof that async memory consolidation works correctly.
        """

        async def slow_response(*args, **kwargs):
            await asyncio.sleep(1)
            return build_llm_response(content="summary")

        mock_agenerate.side_effect = slow_response

        model = MockModel()

        agents = []
        for _ in range(3):
            agent = LLMAgent(
                model=model,
                reasoning=MockReasoning,
                llm_model="openai/fake-model",
            )

            # Force very small capacity
            agent.memory.capacity = 1
            agent.memory.consolidation_capacity = 1

            # Seed memory so next post_step triggers consolidation
            agent.memory.add_to_memory("observation", {"x": 1})
            agent.memory.add_to_memory("observation", {"x": 2})

            agents.append(agent)

        start = time.perf_counter()
        await step_agents_parallel(agents)
        duration = time.perf_counter() - start

        logger.info("Total time: %.4fs", duration)

        # Serial ≈ 3s, Parallel ≈ 1s
        self.assertLess(duration, 2.0)
