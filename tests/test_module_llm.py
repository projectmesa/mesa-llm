import os
from unittest.mock import patch

import pytest

from mesa_llm.module_llm import ModuleLLM


# Dummy responses to stub out external LLM calls during tests
class _DummyResponse(dict):
    pass


def _dummy_completion(**kwargs):
    return _DummyResponse({"choices": [{"message": {"content": "ok"}}]})


async def _dummy_acompletion(**kwargs):
    return _DummyResponse({"choices": [{"message": {"content": "ok"}}]})


@pytest.fixture
def mock_api_key():
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}, clear=True):
        yield


class TestModuleLLM:
    """Test ModuleLLM class"""

    def test_module_llm_initialization(self, mock_api_key):
        # Test initialization with default values
        llm = ModuleLLM(llm_model="openai/gpt-4o")
        assert llm.api_key == "test_key"
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

    def test_get_messages(self):
        # Test get_messages with string prompt
        llm = ModuleLLM(llm_model="openai/gpt-4o")
        messages = llm.get_messages("Hello, how are you?")
        assert messages == [
            {"role": "system", "content": ""},
            {"role": "user", "content": "Hello, how are you?"},
        ]

        # Test get_messages with list of prompts
        messages = llm.get_messages(
            ["Hello, how are you?", "What is the weather in Tokyo?"]
        )
        assert messages == [
            {"role": "system", "content": ""},
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "user", "content": "What is the weather in Tokyo?"},
        ]

        # Test get_messages with system prompt
        llm = ModuleLLM(
            llm_model="openai/gpt-4o", system_prompt="You are a helpful assistant."
        )
        messages = llm.get_messages("Hello, how are you?")
        assert messages == [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"},
        ]

        # Test get_messages with system prompt and list of prompts
        messages = llm.get_messages(
            ["Hello, how are you?", "What is the weather in Tokyo?"]
        )
        assert messages == [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "user", "content": "What is the weather in Tokyo?"},
        ]

        # Test get_messages no system prompt and no prompt
        llm = ModuleLLM(llm_model="openai/gpt-4o")
        messages = llm.get_messages(prompt=None)
        assert messages == [{"role": "system", "content": ""}]

    def test_generate(self, monkeypatch):
        # Prevent network calls by stubbing litellm completion
        monkeypatch.setattr("mesa_llm.module_llm.completion", _dummy_completion)
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
    async def test_agenerate(self, monkeypatch):
        # Prevent network calls by stubbing litellm acompletion
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
