# tests/test_reasoning/test_reasoning.py

from unittest.mock import Mock

from mesa_llm.reasoning.reasoning import (
    Observation,
    Plan,
)


class TestObservation:
    """Test the Observation dataclass."""

    def test_observation_creation(self):
        """Test creating an Observation with valid data."""
        obs = Observation(
            step=1,
            self_state={"position": (0, 0), "health": 100},
            local_state={"Agent_1": {"position": (1, 1), "health": 90}},
        )

        assert obs.step == 1
        assert obs.self_state["position"] == (0, 0)
        assert obs.self_state["health"] == 100
        assert "Agent_1" in obs.local_state
        assert obs.local_state["Agent_1"]["position"] == (1, 1)


class TestPlan:
    """Test the Plan dataclass."""

    def test_plan_creation(self):
        """Test creating a Plan with valid data."""
        mock_llm_response = Mock()
        mock_llm_response.content = "Test plan content"

        plan = Plan(step=1, llm_plan=mock_llm_response, ttl=3)

        assert plan.step == 1
        assert plan.llm_plan == mock_llm_response
        assert plan.ttl == 3
