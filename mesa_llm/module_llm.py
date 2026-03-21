import logging
import os
from typing import Any

from dotenv import load_dotenv
from litellm import acompletion, completion, litellm
from litellm.exceptions import (
    APIConnectionError,
    RateLimitError,
    Timeout,
)
from tenacity import AsyncRetrying, retry, retry_if_exception_type, wait_exponential

RETRYABLE_EXCEPTIONS = (
    APIConnectionError,
    Timeout,
    RateLimitError,
)

load_dotenv()
logger = logging.getLogger(__name__)


class ModuleLLM:
    """
    A module that provides a simple interface for using LLMs.

    Note: Currently supports OpenAI, Anthropic, xAI, Huggingface,
    Ollama, OpenRouter, NovitaAI, Gemini.
    """

    def __init__(
        self,
        llm_model: str,
        api_base: str | None = None,
        system_prompt: str | None = None,
    ) -> None:
        """
        Initialize the LLM module.

        Args:
            llm_model (str): The model to use in the format "{provider}/{model}"
                (for example, "openai/gpt-4o").
            api_base (str | None): The API base URL. Required for Ollama providers.
            system_prompt (str | None): The system prompt to use for the LLM.

        Raises:
            ValueError: If llm_model is not in the expected "{provider}/{model}"
                format, or if the provider API key is missing.
        """
        self.api_base: str | None = api_base
        self.llm_model: str = llm_model
        self.system_prompt: str | None = system_prompt

        if "/" not in llm_model:
            raise ValueError(
                f"Invalid model format '{llm_model}'. "
                "Expected '{provider}/{model}', e.g. 'openai/gpt-4o'."
            )

        provider = self.llm_model.split("/")[0].upper()

        if provider in ["OLLAMA", "OLLAMA_CHAT"]:
            if self.api_base is None:
                self.api_base = "http://localhost:11434"
                logger.warning(
                    "Using default Ollama API base: %s. If inference is not working, "
                    "you may need to set the API base to the correct URL.",
                    self.api_base,
                )
        else:
            try:
                self.api_key: str = os.environ[f"{provider}_API_KEY"]
            except KeyError as err:
                raise ValueError(
                    f"No API key found for {provider}. Please set the "
                    f"{provider}_API_KEY environment variable (e.g., in your .env file)."
                ) from err

        if not litellm.supports_function_calling(model=self.llm_model):
            logger.warning(
                "%s does not support function calling. This model may not be able "
                "to use tools. Please check the model documentation at "
                "https://docs.litellm.ai/docs/providers for more information.",
                self.llm_model,
            )

    def _build_messages(
        self, prompt: str | list[str] | None = None
    ) -> list[dict[str, str]]:
        """
        Format the prompt messages for the LLM.

        Args:
            prompt (str | list[str] | None): The prompt to generate a response for.

        Returns:
            list[dict[str, str]]: Messages in {"role": ..., "content": ...} format.
        """
        messages: list[dict[str, str]] = []

        system_content = self.system_prompt if self.system_prompt else ""
        messages.append({"role": "system", "content": system_content})

        if prompt:
            if isinstance(prompt, str):
                messages.append({"role": "user", "content": prompt})
            elif isinstance(prompt, list):
                messages.extend([{"role": "user", "content": p} for p in prompt])

        return messages

    @retry(
        wait=wait_exponential(multiplier=1, min=1, max=60),
        retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
        reraise=True,
    )
    def generate(
        self,
        prompt: str | list[str] | None = None,
        tool_schema: list[dict[str, Any]] | None = None,
        tool_choice: str = "auto",
        response_format: dict[str, Any] | object | None = None,
    ) -> Any:
        """
        Generate a response from the LLM using litellm.

        Args:
            prompt (str | list[str] | None): The prompt to generate a response for.
            tool_schema (list[dict[str, Any]] | None): Schema of tools available.
            tool_choice (str): Tool selection strategy. Defaults to "auto".
            response_format (dict[str, Any] | object | None): Desired response format.

        Returns:
            Any: The raw litellm response object.
        """
        messages = self._build_messages(prompt)

        completion_kwargs: dict[str, Any] = {
            "model": self.llm_model,
            "messages": messages,
            "tools": tool_schema,
            "tool_choice": tool_choice if tool_schema else None,
            "response_format": response_format,
        }
        if self.api_base:
            completion_kwargs["api_base"] = self.api_base

        return completion(**completion_kwargs)

    async def agenerate(
        self,
        prompt: str | list[str] | None = None,
        tool_schema: list[dict[str, Any]] | None = None,
        tool_choice: str = "auto",
        response_format: dict[str, Any] | object | None = None,
    ) -> Any:
        """
        Asynchronous version of generate() for parallel LLM calls.

        Args:
            prompt (str | list[str] | None): The prompt to generate a response for.
            tool_schema (list[dict[str, Any]] | None): Schema of tools available.
            tool_choice (str): Tool selection strategy. Defaults to "auto".
            response_format (dict[str, Any] | object | None): Desired response format.

        Returns:
            Any: The raw litellm response object.
        """
        messages = self._build_messages(prompt)
        response: Any = None
        async for attempt in AsyncRetrying(
            wait=wait_exponential(multiplier=1, min=1, max=60),
            retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
            reraise=True,
        ):
            with attempt:
                completion_kwargs: dict[str, Any] = {
                    "model": self.llm_model,
                    "messages": messages,
                    "tools": tool_schema,
                    "tool_choice": tool_choice if tool_schema else None,
                    "response_format": response_format,
                }
                if self.api_base:
                    completion_kwargs["api_base"] = self.api_base

                response = await acompletion(**completion_kwargs)
        return response
