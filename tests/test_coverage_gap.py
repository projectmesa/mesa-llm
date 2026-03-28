import asyncio

import pytest
from mesa.discrete_space import OrthogonalMooreGrid
from mesa.space import ContinuousSpace, SingleGrid

from mesa_llm.llm_agent import LLMAgent
from mesa_llm.reasoning.reasoning import Reasoning
from mesa_llm.tools.tool_decorator import _python_to_json_type, tool


class DummyReasoning(Reasoning):
    def plan(self, *args, **kwargs):
        pass

    async def aplan(self, *args, **kwargs):
        pass


class AsyncAgent(LLMAgent):
    def __init__(self, model):
        super().__init__(model, reasoning=DummyReasoning)
        self.step_called = False
        self.astep_called = False

    async def astep(self):
        self.astep_called = True
        return await super().astep()


@pytest.mark.asyncio
async def test_llm_agent_async_features(mocker):
    model = mocker.Mock()
    model.steps = 1
    # Mocking memory methods to avoid actual LLM calls
    mocker.patch(
        "mesa_llm.memory.st_lt_memory.STLTMemory.aadd_to_memory",
        side_effect=lambda **kwargs: None,
    )
    mocker.patch(
        "mesa_llm.memory.st_lt_memory.STLTMemory.aprocess_step",
        side_effect=lambda **kwargs: None,
    )

    agent = AsyncAgent(model)

    # Test astep wrapper
    await agent.astep()
    assert agent.astep_called is True

    # Test apre_step and apost_step directly
    await agent.apre_step()
    await agent.apost_step()

    # Test asend_message
    recipients = [mocker.Mock(memory=mocker.Mock(aadd_to_memory=mocker.AsyncMock()))]
    await agent.asend_message("hello", recipients)
    for r in recipients:
        r.memory.aadd_to_memory.assert_called()


def test_llm_agent_pos_none_clears_cell(mocker):
    grid = OrthogonalMooreGrid(dimensions=(5, 5), torus=False)
    model = mocker.Mock(grid=grid)
    agent = LLMAgent(model, reasoning=DummyReasoning)

    # Place in a cell
    cell = grid._cells[(2, 2)]
    agent.cell = cell
    assert agent.pos == (2, 2)

    # Set pos to None
    agent.pos = None
    assert agent.pos is None
    assert agent.cell is None


def test_llm_agent_move_to_all_spaces(mocker):
    # Test Orthogonal Grid
    grid = OrthogonalMooreGrid(dimensions=(5, 5), torus=False)
    model = mocker.Mock(grid=grid, space=None)
    agent = LLMAgent(model, reasoning=DummyReasoning)
    agent.cell = grid._cells[(0, 0)]

    agent.move_to((1, 1))
    assert agent.pos == (1, 1)
    assert agent.cell is grid._cells[(1, 1)]

    # Test Continuous Space
    space = ContinuousSpace(10, 10, False)
    model = mocker.Mock(space=space, grid=None)
    agent = LLMAgent(model, reasoning=DummyReasoning)
    space.place_agent(agent, (1, 1))

    agent.move_to((5.5, 6.6))
    assert agent.pos == (5.5, 6.6)

    # Test Unsupported
    model = mocker.Mock(grid=None, space=None)
    agent = LLMAgent(model, reasoning=DummyReasoning)
    with pytest.raises(ValueError, match="Unsupported environment"):
        agent.move_to((1, 1))

    # Test SingleGrid (Line 112)
    grid = SingleGrid(5, 5, False)
    model = mocker.Mock(grid=grid, space=None)
    agent = LLMAgent(model, reasoning=DummyReasoning)
    grid.place_agent(agent, (0, 0))
    agent.move_to((1, 1))
    assert agent.pos == (1, 1)


def test_llm_agent_step_wrapper(mocker):
    # Test sync step wrapper (Lines 381-390)
    class SyncSubAgent(LLMAgent):
        def __init__(self, model):
            super().__init__(model, reasoning=DummyReasoning)
            self.step_called = False

        def step(self):
            self.step_called = True

    model = mocker.Mock()
    model.steps = 1
    mocker.patch(
        "mesa_llm.memory.st_lt_memory.STLTMemory.process_step",
        side_effect=lambda **kwargs: None,
    )

    agent = SyncSubAgent(model)
    agent.step()
    assert agent.step_called is True

    # Test astep fallback to step (Line 366)
    mocker.patch(
        "mesa_llm.memory.st_lt_memory.STLTMemory.aprocess_step",
        side_effect=lambda **kwargs: None,
    )

    async def run_astep():
        await agent.astep()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(run_astep())
    assert agent.step_called is True


def test_tool_decorator_string_annotations():
    # Test _python_to_json_type with string literals
    assert _python_to_json_type("int") == {"type": "integer"}
    assert _python_to_json_type("list[int]") == {
        "type": "array",
        "items": {"type": "integer"},
    }
    assert _python_to_json_type("list[str]") == {
        "type": "array",
        "items": {"type": "string"},
    }
    assert _python_to_json_type("dict[str, int]") == {
        "type": "object",
        "additionalProperties": {"type": "integer"},
    }

    # Test invalid string fallback
    assert _python_to_json_type("SomethingUnknown") == {"type": "string"}

    # Test None type
    assert _python_to_json_type(type(None)) == {"type": "null"}


def test_optional_union_edge_cases():
    # Test Union with more than 2 types
    schema = _python_to_json_type(int | str | float)
    assert "anyOf" in schema
    assert len(schema["anyOf"]) == 3

    # Test Optional with None only (unusual but possible)
    schema = _python_to_json_type(type(None) | type(None))
    assert schema == {"type": "null"}


@tool
def string_annotated_tool(agent, data: "list[int]") -> str:
    """A tool with string annotations.
    Args:
        data: describe data
    """
    return str(data)


def test_string_annotated_tool_schema():
    schema = string_annotated_tool.__tool_schema__
    props = schema["function"]["parameters"]["properties"]
    assert props["data"]["type"] == "array"
    assert props["data"]["items"]["type"] == "integer"
