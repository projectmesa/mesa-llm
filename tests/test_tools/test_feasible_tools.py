"""Tests for feasible/annotated tool filtering (issue #148).

Verifies that ToolManager.get_feasible_tools_schema and
get_annotated_tools_schema correctly filter and annotate tools
based on agent-state predicates.
"""

from unittest.mock import Mock

import pytest

from mesa_llm.tools.inbuilt_tools import _has_grid_and_position, _has_other_agents
from mesa_llm.tools.tool_decorator import _GLOBAL_TOOL_REGISTRY, tool
from mesa_llm.tools.tool_manager import ToolManager


def _always_true(agent) -> bool:
    return True


def _always_false(agent) -> bool:
    return False


def _has_energy(agent) -> bool:
    return getattr(agent, "energy", 0) > 0


def _register_test_tools():
    """Register test tools into the global registry and return them."""

    @tool(requires=_always_true)
    def always_available(agent, x: int) -> int:
        """Always available.

        Args:
            agent: Provided automatically.
            x: Input.

        Returns:
            Output.
        """
        return x

    @tool(requires=_always_false)
    def never_available(agent, x: int) -> int:
        """Never available.

        Args:
            agent: Provided automatically.
            x: Input.

        Returns:
            Output.
        """
        return x

    @tool(requires=_has_energy)
    def energy_tool(agent, x: int) -> int:
        """Requires energy > 0.

        Args:
            agent: Provided automatically.
            x: Input.

        Returns:
            Output.
        """
        return x

    @tool
    def no_requires_tool(agent, x: int) -> int:
        """No requires predicate — always feasible.

        Args:
            agent: Provided automatically.
            x: Input.

        Returns:
            Output.
        """
        return x

    return {
        "always_available": always_available,
        "never_available": never_available,
        "energy_tool": energy_tool,
        "no_requires_tool": no_requires_tool,
    }


@pytest.fixture(autouse=True)
def _clean_registry():
    """Start each test with a clean global registry and fresh tools."""
    saved = dict(_GLOBAL_TOOL_REGISTRY)
    saved_instances = list(ToolManager.instances)
    _GLOBAL_TOOL_REGISTRY.clear()
    ToolManager.instances.clear()
    # Re-register our test tools into the now-clean registry
    _register_test_tools()
    yield
    _GLOBAL_TOOL_REGISTRY.clear()
    _GLOBAL_TOOL_REGISTRY.update(saved)
    ToolManager.instances.clear()
    ToolManager.instances.extend(saved_instances)


# ===================================================================
# get_feasible_tools_schema tests
# ===================================================================
class TestGetFeasibleToolsSchema:
    def test_filters_out_infeasible_tools(self):
        manager = ToolManager()
        agent = Mock()
        schemas = manager.get_feasible_tools_schema(agent)
        names = {s["function"]["name"] for s in schemas}
        assert "always_available" in names
        assert "never_available" not in names

    def test_tools_without_requires_always_included(self):
        manager = ToolManager()
        agent = Mock()
        schemas = manager.get_feasible_tools_schema(agent)
        names = {s["function"]["name"] for s in schemas}
        assert "no_requires_tool" in names

    def test_predicate_receives_agent_state(self):
        manager = ToolManager()

        agent_with_energy = Mock()
        agent_with_energy.energy = 10
        schemas = manager.get_feasible_tools_schema(agent_with_energy)
        names = {s["function"]["name"] for s in schemas}
        assert "energy_tool" in names

        agent_no_energy = Mock()
        agent_no_energy.energy = 0
        schemas = manager.get_feasible_tools_schema(agent_no_energy)
        names = {s["function"]["name"] for s in schemas}
        assert "energy_tool" not in names

    def test_with_selected_tools(self):
        manager = ToolManager()
        agent = Mock()
        schemas = manager.get_feasible_tools_schema(
            agent, selected_tools=["always_available", "never_available"]
        )
        names = {s["function"]["name"] for s in schemas}
        assert names == {"always_available"}

    def test_predicate_exception_treated_as_infeasible(self):
        def broken_predicate(agent):
            raise RuntimeError("oops")

        @tool(requires=broken_predicate)
        def broken_tool(agent, x: int) -> int:
            """Broken predicate.

            Args:
                agent: Provided automatically.
                x: Input.

            Returns:
                Output.
            """
            return x

        manager = ToolManager()
        agent = Mock()
        schemas = manager.get_feasible_tools_schema(agent)
        names = {s["function"]["name"] for s in schemas}
        assert "broken_tool" not in names


# ===================================================================
# get_annotated_tools_schema tests
# ===================================================================
class TestGetAnnotatedToolsSchema:
    def test_all_tools_returned_with_annotations(self):
        manager = ToolManager()
        agent = Mock()
        schemas = manager.get_annotated_tools_schema(agent)
        names = {s["function"]["name"] for s in schemas}
        assert "always_available" in names
        assert "never_available" in names
        assert "no_requires_tool" in names

    def test_available_annotation_correct(self):
        manager = ToolManager()
        agent = Mock()
        schemas = manager.get_annotated_tools_schema(agent)
        by_name = {s["function"]["name"]: s for s in schemas}

        assert by_name["always_available"]["available"] is True
        assert by_name["always_available"]["unavailable_reason"] is None

        assert by_name["never_available"]["available"] is False
        assert by_name["never_available"]["unavailable_reason"] is not None

        assert by_name["no_requires_tool"]["available"] is True
        assert by_name["no_requires_tool"]["unavailable_reason"] is None

    def test_annotated_with_selected_tools(self):
        manager = ToolManager()
        agent = Mock()
        schemas = manager.get_annotated_tools_schema(
            agent, selected_tools=["never_available"]
        )
        assert len(schemas) == 1
        assert schemas[0]["available"] is False


# ===================================================================
# Built-in predicate tests (_has_grid_and_position, _has_other_agents)
# ===================================================================
class TestBuiltInPredicates:
    def test_has_grid_and_position_no_grid_no_space(self):
        agent = Mock()
        agent.model.grid = None
        agent.model.space = None
        assert _has_grid_and_position(agent) is False

    def test_has_grid_and_position_with_grid_and_pos(self):
        agent = Mock()
        agent.model.grid = Mock()
        agent.model.space = None
        agent.cell = None
        agent.pos = (1, 2)
        assert _has_grid_and_position(agent) is True

    def test_has_grid_and_position_with_cell_coordinate(self):
        agent = Mock()
        agent.model.grid = Mock()
        agent.model.space = None
        agent.cell = Mock()
        agent.cell.coordinate = (3, 4)
        assert _has_grid_and_position(agent) is True

    def test_has_grid_and_position_no_position(self):
        """Grid exists but agent has no position — should return False."""
        agent = Mock()
        agent.model.grid = Mock()
        agent.model.space = None
        agent.cell = None
        agent.pos = None
        agent.position = None
        assert _has_grid_and_position(agent) is False

    def test_has_other_agents_true(self):
        other = Mock()
        other.unique_id = 2
        agent = Mock()
        agent.unique_id = 1
        agent.model.agents = [agent, other]
        assert _has_other_agents(agent) is True

    def test_has_other_agents_false_alone(self):
        agent = Mock()
        agent.unique_id = 1
        agent.model.agents = [agent]
        assert _has_other_agents(agent) is False
