import logging
import os

from dotenv import load_dotenv
from litellm import acompletion, completion, litellm
from litellm.exceptions import (
    APIConnectionError,
    RateLimitError,
    Timeout,
)
from pydantic import BaseModel
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
    A module that provides a simple interface for using LLMs

    Note : Currently supports OpenAI, Anthropic, xAI, Huggingface, Ollama, OpenRouter, NovitaAI, Gemini
    """

    def __init__(
        self,
        llm_model: str,
        api_base: str | None = None,
        system_prompt: str | None = None,
    ):
        """
        Initialize the LLM module

        Args:
            llm_model: The model to use for the LLM in the format
                "{provider}/{model}" (for example, "openai/gpt-4o").
            api_base: The API base to use if the LLM provider is Ollama
            system_prompt: The system prompt to use for the LLM

        Raises:
            ValueError: If llm_model is not in the expected "{provider}/{model}"
                format, or if the provider API key is missing.
        """
        self.api_base = api_base
        self.llm_model = llm_model
        self.system_prompt = system_prompt

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
                    "Using default Ollama API base: %s. If inference is not working, you may need to set the API base to the correct URL.",
                    self.api_base,
                )
        else:
            try:
                self.api_key = os.environ[f"{provider}_API_KEY"]
            except KeyError as err:
                raise ValueError(
                    f"No API key found for {provider}. Please set the {provider}_API_KEY environment variable (e.g., in your .env file)."
                ) from err

        if not litellm.supports_function_calling(model=self.llm_model):
            logger.warning(
                "%s does not support function calling. This model may not be able to use tools. Please check the model documentation at https://docs.litellm.ai/docs/providers for more information.",
                self.llm_model,
            )

    def _build_messages(
        self,
        prompt: str | list[str] | None = None,
        system_prompt: str | None = None,
    ) -> list[dict]:
        """
        Format the prompt messages for the LLM of the form : {"role": ..., "content": ...}

        Args:
            prompt: The prompt to generate a response for (str, list of strings, or None)

        Returns:
            The messages for the LLM
        """
        messages = []

        # Always include a system message. Default to empty string if no system prompt to support Ollama
        system_content = (
            system_prompt if system_prompt is not None else self.system_prompt
        )
        system_content = system_content if system_content else ""
        messages.append({"role": "system", "content": system_content})

        if prompt:
            if isinstance(prompt, str):
                messages.append({"role": "user", "content": prompt})
            elif isinstance(prompt, list):
                # Use extend to add all prompts from the list
                messages.extend([{"role": "user", "content": p} for p in prompt])

        return messages

    def parse_structured_output(
        self,
        response,
        response_model: type[BaseModel],
    ) -> BaseModel:
        """Normalize structured LLM output into the requested pydantic model."""
        message = response.choices[0].message
        parsed = getattr(message, "parsed", None)

        if isinstance(parsed, response_model):
            return parsed

        if parsed is not None:
            return response_model.model_validate(parsed)

        return response_model.model_validate_json(message.content)

    @retry(
        wait=wait_exponential(multiplier=1, min=1, max=60),
        retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
        reraise=True,
    )
    def generate(
        self,
        prompt: str | list[str] | None = None,
        tool_schema: list[dict] | None = None,
        tool_choice: str = "auto",
        response_format: dict | object | None = None,
        system_prompt: str | None = None,
    ) -> str:
        """
        Generate a response from the LLM using litellm based on the prompt

        Args:
            prompt: The prompt to generate a response for (str, list of strings, or None)
            tool_schema: The schema of the tools to use
            tool_choice: The choice of tool to use
            response_format: The format of the response

        Returns:
            The response from the LLM
        """

        messages = self._build_messages(prompt, system_prompt=system_prompt)

        completion_kwargs = {
            "model": self.llm_model,
            "messages": messages,
            "tools": tool_schema,
            "tool_choice": tool_choice if tool_schema else None,
            "response_format": response_format,
        }
        if self.api_base:
            completion_kwargs["api_base"] = self.api_base

        response = completion(**completion_kwargs)

        return response

    async def agenerate(
        self,
        prompt: str | list[str] | None = None,
        tool_schema: list[dict] | None = None,
        tool_choice: str = "auto",
        response_format: dict | object | None = None,
        system_prompt: str | None = None,
    ) -> str:
        """
        Asynchronous version of generate() method for parallel LLM calls.
        """
        messages = self._build_messages(prompt, system_prompt=system_prompt)
        async for attempt in AsyncRetrying(
            wait=wait_exponential(multiplier=1, min=1, max=60),
            retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
            reraise=True,
        ):
            with attempt:
                completion_kwargs = {
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
