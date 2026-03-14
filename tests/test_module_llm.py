import os
from unittest.mock import Mock, patch

import pytest
from pydantic import BaseModel

from mesa_llm.module_llm import ModuleLLM


class TestModuleLLM:
    """Test ModuleLLM class"""

    def test_missing_provider_prefix(self):
        """ModuleLLM should raise ValueError when llm_model has no provider prefix."""
        with pytest.raises(ValueError, match="Invalid model format"):
            ModuleLLM(llm_model="gpt-4o")

    def test_module_llm_initialization(self, mock_environment):
        # Test initialization with default values
        llm = ModuleLLM(llm_model="openai/gpt-4o")
        assert llm.api_key == "test_openai_key"
        assert llm.api_base is None
        assert llm.llm_model == "openai/gpt-4o"
        assert llm.system_prompt is None

        # Test initialization with ollama provider
        llm = ModuleLLM(llm_model="ollama/llama2")
        assert llm.api_base == "http://localhost:11434"
        assert llm.llm_model == "ollama/llama2"
        assert llm.system_prompt is None

        # Test initialization with ollama provider + custom api_base
        llm = ModuleLLM(llm_model="ollama/llama2", api_base="http://localhost:99999")
        assert llm.api_base == "http://localhost:99999"
        assert llm.llm_model == "ollama/llama2"
        assert llm.system_prompt is None

        # Test init without api_key in dotenv
        with patch.dict(os.environ, {}, clear=True), pytest.raises(ValueError):
            ModuleLLM(llm_model="openai/gpt-4o")

    def test_build_messages(self):
        # Test _build_messages with string prompt
        llm = ModuleLLM(llm_model="openai/gpt-4o")
        messages = llm._build_messages("Hello, how are you?")
        assert messages == [
            {"role": "system", "content": ""},
            {"role": "user", "content": "Hello, how are you?"},
        ]

        # Test _build_messages with list of prompts
        messages = llm._build_messages(
            ["Hello, how are you?", "What is the weather in Tokyo?"]
        )
        assert messages == [
            {"role": "system", "content": ""},
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "user", "content": "What is the weather in Tokyo?"},
        ]

        # Test _build_messages with system prompt
        llm = ModuleLLM(
            llm_model="openai/gpt-4o", system_prompt="You are a helpful assistant."
        )
        messages = llm._build_messages("Hello, how are you?")
        assert messages == [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"},
        ]

        # Test _build_messages with system prompt and list of prompts
        messages = llm._build_messages(
            ["Hello, how are you?", "What is the weather in Tokyo?"]
        )
        assert messages == [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "user", "content": "What is the weather in Tokyo?"},
        ]

        # Test _build_messages no system prompt and no prompt
        llm = ModuleLLM(llm_model="openai/gpt-4o")
        messages = llm._build_messages(prompt=None)
        assert messages == [{"role": "system", "content": ""}]

        messages = llm._build_messages(
            "Hello, how are you?", system_prompt="Per-call prompt"
        )
        assert messages == [
            {"role": "system", "content": "Per-call prompt"},
            {"role": "user", "content": "Hello, how are you?"},
        ]

    def test_parse_structured_output(self):
        class DummyOutput(BaseModel):
            answer: str

        llm = ModuleLLM(llm_model="openai/gpt-4o")

        response = Mock()
        response.choices = [Mock()]
        response.choices[0].message = Mock()
        response.choices[0].message.parsed = DummyOutput(answer="parsed")
        parsed = llm.parse_structured_output(response, DummyOutput)
        assert parsed.answer == "parsed"

        response.choices[0].message.parsed = {"answer": "dict"}
        parsed = llm.parse_structured_output(response, DummyOutput)
        assert parsed.answer == "dict"

        response.choices[0].message.parsed = None
        response.choices[0].message.content = '{"answer":"json"}'
        parsed = llm.parse_structured_output(response, DummyOutput)
        assert parsed.answer == "json"

    def test_generate(self, monkeypatch, llm_response_factory):
        monkeypatch.setattr(
            "mesa_llm.module_llm.completion", lambda **kwargs: llm_response_factory()
        )
        # Test generate with string prompt
        llm = ModuleLLM(llm_model="openai/gpt-4o")
        response = llm.generate(prompt="Hello, how are you?")
        assert response is not None

        # Test generate with list of prompts
        response = llm.generate(
            prompt=["Hello, how are you?", "What is the weather in Tokyo?"]
        )
        assert response is not None

        # Test generate with string prompt for Ollama
        llm = ModuleLLM(llm_model="ollama/llama2")
        response = llm.generate(prompt="Hello, how are you?")
        assert response is not None

        # Test generate with list of prompts
        response = llm.generate(
            prompt=["Hello, how are you?", "What is the weather in Tokyo?"]
        )
        assert response is not None

    @pytest.mark.asyncio
    async def test_agenerate(self, monkeypatch, llm_response_factory):
        async def _dummy_acompletion(**kwargs):
            return llm_response_factory()

        monkeypatch.setattr("mesa_llm.module_llm.acompletion", _dummy_acompletion)
        # Test agenerate with string prompt
        llm = ModuleLLM(llm_model="openai/gpt-4o")
        response = await llm.agenerate(prompt="Hello, how are you?")
        assert response is not None

        # Test agenerate with list of prompts
        response = await llm.agenerate(
            prompt=["Hello, how are you?", "What is the weather in Tokyo?"]
        )
        assert response is not None
