import asyncio
import json
from unittest.mock import AsyncMock, Mock

import pytest

from mesa_llm.reasoning.decision import (
    DecisionOption,
    DecisionOutput,
    DecisionReasoning,
)
from mesa_llm.reasoning.reasoning import Observation, Plan


class TestDecisionModels:
    def test_decision_option_creation(self):
        option = DecisionOption(
            name="move_to_market",
            description="Move toward the market cell.",
            tradeoffs=["Consumes one turn", "May expose the agent"],
            score=0.82,
        )

        assert option.name == "move_to_market"
        assert option.tradeoffs == ["Consumes one turn", "May expose the agent"]
        assert option.score == 0.82

    def test_decision_output_creation(self):
        output = DecisionOutput(
            goal="Reach a safer location",
            constraints=["Can only move once"],
            known_facts=["An exit is visible to the north"],
            unknowns=["Whether another agent will block the path"],
            assumptions=["The northern cell remains open this turn"],
            options=[
                DecisionOption(
                    name="move_north",
                    description="Move one cell north.",
                    tradeoffs=["Fast", "Could enter conflict"],
                    score=0.9,
                )
            ],
            chosen_option="move_north",
            rationale="It advances the goal with the best visible route.",
            confidence=0.76,
            risks=["The path could become blocked"],
            next_action="move_north",
        )

        assert output.goal == "Reach a safer location"
        assert output.chosen_option == "move_north"
        assert output.confidence == 0.76
        assert output.next_action == "move_north"


class TestDecisionReasoning:
    def test_decision_reasoning_initialization(self, mock_agent):
        reasoning = DecisionReasoning(mock_agent)

        assert reasoning.agent == mock_agent

    def test_get_decision_system_prompt(self, mock_agent):
        reasoning = DecisionReasoning(mock_agent)

        prompt = reasoning.get_decision_system_prompt()

        assert "strict JSON object" in prompt
        assert "known_facts" in prompt
        assert "next_action" in prompt

    def test_get_decision_prompt_with_observation(self, mock_agent):
        mock_agent.memory = Mock()
        mock_agent.memory.get_prompt_ready.return_value = "memory1\n\nmemory2"
        mock_agent.memory.get_communication_history.return_value = "communication"

        reasoning = DecisionReasoning(mock_agent)
        obs = Observation(step=1, self_state={}, local_state={})
        prompt_list = reasoning.get_decision_prompt(obs)

        assert len(prompt_list) >= 2
        assert "last communication" in prompt_list[-2]
        assert "current observation" in prompt_list[-1]

    def test_plan_with_prompt(self, llm_response_factory, mock_agent):
        mock_agent.memory = Mock()
        mock_agent.memory.get_prompt_ready.return_value = "memory1"
        mock_agent.memory.get_communication_history.return_value = ""
        mock_agent.memory.add_to_memory = Mock()
        mock_agent.llm = Mock()
        mock_agent.tool_manager = Mock()
        mock_agent.tool_manager.get_all_tools_schema.return_value = {}
        mock_agent._step_display_data = {}

        mock_agent.llm.generate.return_value = llm_response_factory(
            content=json.dumps(
                {
                    "goal": "Reach food",
                    "constraints": ["One movement per turn"],
                    "known_facts": ["Food is visible nearby"],
                    "unknowns": ["Whether the route stays open"],
                    "assumptions": ["The route remains open this step"],
                    "options": [
                        {
                            "name": "move_to_food",
                            "description": "Move toward visible food",
                            "tradeoffs": ["Fast", "May be contested"],
                            "score": 0.88,
                        }
                    ],
                    "chosen_option": "move_to_food",
                    "rationale": "It best advances the immediate goal.",
                    "confidence": 0.78,
                    "risks": ["Another agent may reach it first"],
                    "next_action": "move_to_food",
                }
            )
        )

        mock_plan = Plan(step=1, llm_plan=Mock())
        reasoning = DecisionReasoning(mock_agent)
        reasoning.execute_tool_call = Mock(return_value=mock_plan)

        obs = Observation(step=1, self_state={}, local_state={})
        result = reasoning.plan(obs=obs, prompt="Custom prompt")

        assert result == mock_plan
        mock_agent.memory.add_to_memory.assert_called_once()
        reasoning.execute_tool_call.assert_called_once_with(
            "move_to_food",
            selected_tools=None,
            ttl=1,
        )
        assert mock_agent._step_display_data["plan_content"] == (
            "It best advances the immediate goal."
        )

    def test_plan_with_selected_tools(self, llm_response_factory, mock_agent):
        mock_agent.step_prompt = "Default step prompt"
        mock_agent.memory = Mock()
        mock_agent.memory.get_prompt_ready.return_value = "memory1"
        mock_agent.memory.get_communication_history.return_value = ""
        mock_agent.memory.add_to_memory = Mock()
        mock_agent.llm = Mock()
        mock_agent.tool_manager = Mock()
        mock_agent.tool_manager.get_all_tools_schema.return_value = {}
        mock_agent._step_display_data = {}

        mock_agent.llm.generate.return_value = llm_response_factory(
            content=json.dumps(
                {
                    "goal": "Hold position",
                    "constraints": ["No safe exit visible"],
                    "known_facts": ["A threat is adjacent"],
                    "unknowns": ["Threat intent"],
                    "assumptions": ["Staying still is safer than moving blindly"],
                    "options": [
                        {
                            "name": "wait",
                            "description": "Hold current position",
                            "tradeoffs": ["No progress", "Reduces exposure"],
                            "score": 0.61,
                        }
                    ],
                    "chosen_option": "wait",
                    "rationale": "It minimizes immediate danger.",
                    "confidence": 0.53,
                    "risks": ["Threat may approach anyway"],
                    "next_action": "wait",
                }
            )
        )

        mock_plan = Plan(step=1, llm_plan=Mock())
        reasoning = DecisionReasoning(mock_agent)
        reasoning.execute_tool_call = Mock(return_value=mock_plan)

        obs = Observation(step=1, self_state={}, local_state={})
        selected_tools = ["tool1", "tool2"]
        result = reasoning.plan(obs=obs, ttl=3, selected_tools=selected_tools)

        assert result == mock_plan
        mock_agent.tool_manager.get_all_tools_schema.assert_called_with(selected_tools)
        reasoning.execute_tool_call.assert_called_once_with(
            "wait",
            selected_tools=selected_tools,
            ttl=3,
        )

    def test_get_decision_prompt_without_optional_memory_methods(self, mock_agent):
        mock_agent.memory = Mock(spec=[])

        reasoning = DecisionReasoning(mock_agent)
        obs = Observation(step=1, self_state={}, local_state={})

        prompt_list = reasoning.get_decision_prompt(obs)

        assert prompt_list == ["current observation: \n" + str(obs)]

    def test_plan_uses_structured_response_when_available(
        self, llm_response_factory, mock_agent
    ):
        mock_agent.memory = Mock()
        mock_agent.memory.get_prompt_ready.return_value = "memory1"
        mock_agent.memory.get_communication_history.return_value = ""
        mock_agent.memory.add_to_memory = Mock()
        mock_agent.llm = Mock()
        mock_agent.tool_manager = Mock()
        mock_agent.tool_manager.get_all_tools_schema.return_value = {}
        mock_agent._step_display_data = {}

        parsed_output = DecisionOutput(
            goal="Reach food",
            constraints=["One movement per turn"],
            known_facts=["Food is visible nearby"],
            unknowns=["Whether the route stays open"],
            assumptions=["The route remains open this step"],
            options=[
                DecisionOption(
                    name="move_to_food",
                    description="Move toward visible food",
                    tradeoffs=["Fast", "May be contested"],
                    score=0.88,
                )
            ],
            chosen_option="move_to_food",
            rationale="It best advances the immediate goal.",
            confidence=0.78,
            risks=["Another agent may reach it first"],
            next_action="move_to_food",
        )
        rsp = llm_response_factory(content="ignored")
        rsp.choices[0].message.parsed = parsed_output
        mock_agent.llm.generate.return_value = rsp

        mock_plan = Plan(step=1, llm_plan=Mock())
        reasoning = DecisionReasoning(mock_agent)
        reasoning.execute_tool_call = Mock(return_value=mock_plan)

        obs = Observation(step=1, self_state={}, local_state={})
        result = reasoning.plan(obs=obs, prompt="Custom prompt")

        assert result == mock_plan
        mock_agent.memory.add_to_memory.assert_called_once_with(
            type="decision", content=parsed_output.model_dump()
        )

    def test_plan_no_prompt_error(self, mock_agent):
        mock_agent.step_prompt = None
        mock_agent.memory = Mock()
        mock_agent.memory.get_prompt_ready.return_value = "memory1"
        mock_agent.memory.get_communication_history.return_value = ""

        reasoning = DecisionReasoning(mock_agent)
        obs = Observation(step=1, self_state={}, local_state={})

        with pytest.raises(
            ValueError, match=r"No prompt provided and agent.step_prompt is None"
        ):
            reasoning.plan(obs=obs)

    def test_aplan_async_version(self, llm_response_factory, mock_agent):
        mock_agent.step_prompt = "Default step prompt"
        mock_agent.memory = Mock()
        mock_agent.memory.get_prompt_ready.return_value = "memory1"
        mock_agent.memory.get_communication_history.return_value = ""
        mock_agent.memory.aadd_to_memory = AsyncMock()
        mock_agent.llm = Mock()
        mock_agent.tool_manager = Mock()
        mock_agent.tool_manager.get_all_tools_schema.return_value = {}
        mock_agent._step_display_data = {}

        mock_agent.llm.agenerate = AsyncMock(
            return_value=llm_response_factory(
                content=json.dumps(
                    {
                        "goal": "Move closer to ally",
                        "constraints": ["One step per turn"],
                        "known_facts": ["An ally is east of the agent"],
                        "unknowns": ["Whether the east cell is contested"],
                        "assumptions": ["The ally remains in place this step"],
                        "options": [
                            {
                                "name": "move_east",
                                "description": "Move one cell east",
                                "tradeoffs": ["Improves coordination", "May increase exposure"],
                                "score": 0.74,
                            }
                        ],
                        "chosen_option": "move_east",
                        "rationale": "It improves coordination with acceptable risk.",
                        "confidence": 0.69,
                        "risks": ["The east cell may be occupied"],
                        "next_action": "move_east",
                    }
                )
            )
        )

        mock_plan = Plan(step=1, llm_plan=Mock())
        reasoning = DecisionReasoning(mock_agent)
        reasoning.aexecute_tool_call = AsyncMock(return_value=mock_plan)

        obs = Observation(step=1, self_state={}, local_state={})
        result = asyncio.run(reasoning.aplan(obs=obs, ttl=4))

        assert result == mock_plan
        mock_agent.llm.agenerate.assert_called_once()
        reasoning.aexecute_tool_call.assert_awaited_once_with(
            "move_east",
            selected_tools=None,
            ttl=4,
        )
        assert mock_agent._step_display_data["plan_content"] == (
            "It improves coordination with acceptable risk."
        )

    def test_aplan_no_prompt_error(self, mock_agent):
        mock_agent.step_prompt = None
        mock_agent.memory = Mock()
        mock_agent.memory.get_prompt_ready.return_value = "memory1"
        mock_agent.memory.get_communication_history.return_value = ""

        reasoning = DecisionReasoning(mock_agent)
        obs = Observation(step=1, self_state={}, local_state={})

        with pytest.raises(
            ValueError, match=r"No prompt provided and agent.step_prompt is None"
        ):
            asyncio.run(reasoning.aplan(obs=obs))
