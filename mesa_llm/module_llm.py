import asyncio
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

try:
    from groq import AsyncGroq as AsyncGroqClient
    from groq import Groq as GroqClient
except ImportError:  # pragma: no cover - optional dependency in local dev envs
    AsyncGroqClient = None
    GroqClient = None

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

    Note : Currently supports OpenAI, Anthropic, xAI, Huggingface, Ollama,
    OpenRouter, NovitaAI, Gemini, and Groq.
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
        self.api_key: str | None = None
        self._groq_client: Any | None = None
        self._agroq_client: Any | None = None

        if "/" not in llm_model:
            raise ValueError(
                f"Invalid model format '{llm_model}'. "
                "Expected '{provider}/{model}', e.g. 'openai/gpt-4o'."
            )

        provider = self.llm_model.split("/", 1)[0].upper()

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

        if provider == "GROQ":
            if "/" not in self.llm_model or not self.llm_model.split("/", 1)[1]:
                raise ValueError(
                    "Invalid model format for Groq. Expected 'groq/{model_name}'."
                )
            if GroqClient is None:
                raise ValueError(
                    "Groq provider selected but Groq SDK is not installed. Please install 'groq'."
                )
            self._groq_client = GroqClient(api_key=self.api_key)
            if AsyncGroqClient is not None:
                self._agroq_client = AsyncGroqClient(api_key=self.api_key)

        if not litellm.supports_function_calling(model=self.llm_model):
            logger.warning(
                "%s does not support function calling. This model may not be able to use tools. Please check the model documentation at https://docs.litellm.ai/docs/providers for more information.",
                self.llm_model,
            )

    def _build_messages(self, prompt: str | list[str] | None = None) -> list[dict]:
        """
        Format the prompt messages for the LLM of the form : {"role": ..., "content": ...}

        Args:
            prompt: The prompt to generate a response for (str, list of strings, or None)

        Returns:
            The messages for the LLM
        """
        messages = []

        # Always include a system message. Default to empty string if no system prompt to support Ollama
        system_content = self.system_prompt if self.system_prompt else ""
        messages.append({"role": "system", "content": system_content})

        if prompt:
            if isinstance(prompt, str):
                messages.append({"role": "user", "content": prompt})
            elif isinstance(prompt, list):
                if not all(isinstance(p, str) for p in prompt):
                    raise TypeError("prompt list must contain only strings")
                # Use extend to add all prompts from the list
                messages.extend([{"role": "user", "content": p} for p in prompt])
            else:
                raise TypeError("prompt must be a string, list[str], or None")

        return messages

    def _is_groq_provider(self) -> bool:
        """Return whether the configured provider is Groq."""
        return self.llm_model.split("/", 1)[0].lower() == "groq"

    def _groq_model_name(self) -> str:
        """Return model name portion expected by the Groq SDK."""
        return self.llm_model.split("/", 1)[1]

    def _groq_create_kwargs(
        self,
        messages: list[dict],
        tool_schema: list[dict] | None,
        tool_choice: str,
        response_format: dict | object | None,
    ) -> dict[str, Any]:
        """Build request kwargs for Groq chat completions."""
        kwargs: dict[str, Any] = {
            "model": self._groq_model_name(),
            "messages": messages,
        }
        if tool_schema:
            kwargs["tools"] = tool_schema
            kwargs["tool_choice"] = tool_choice
        if isinstance(response_format, dict):
            kwargs["response_format"] = response_format
        return kwargs

    def _raise_groq_error(self, error: Exception) -> None:
        """Raise normalized Groq errors with actionable context."""
        message = str(error)
        lowered = message.lower()
        if "model" in lowered and (
            "not found" in lowered or "invalid" in lowered or "unknown" in lowered
        ):
            raise ValueError(
                f"Invalid Groq model '{self.llm_model}'. Please verify the model name and provider prefix."
            ) from error
        if "timeout" in lowered:
            raise TimeoutError(
                f"Groq request timed out for model '{self.llm_model}'. Please retry."
            ) from error
        raise RuntimeError(
            f"Groq request failed for model '{self.llm_model}': {message}"
        ) from error

    def _build_rate_limit_error(self, error: RateLimitError) -> RateLimitError:
        provider = self.llm_model.split("/", 1)[0].lower()
        docs_url = {
            "anthropic": "https://platform.claude.com/docs/en/api/rate-limits",
            "gemini": "https://ai.google.dev/gemini-api/docs/rate-limits",
            "groq": "https://console.groq.com/docs/rate-limits",
            "novita": "https://novita.ai/docs/guides/llm-rate-limits",
            "openai": "https://developers.openai.com/api/docs/guides/rate-limits",
            "openrouter": "https://openrouter.ai/docs/api/reference/limits",
            "xai": "https://docs.x.ai/developers/rate-limits",
        }.get(provider)

        detail = error.message.removeprefix("litellm.RateLimitError: ").strip()
        message_parts = [f"Rate limit exceeded for model '{self.llm_model}'."]
        if detail:
            message_parts.append(detail)
        message_parts.append(
            "Please wait a few minutes and try again, or switch to a different model."
        )
        if docs_url:
            message_parts.append(f"To check your quota visit: {docs_url}")

        message = " ".join(message_parts)
        return RateLimitError(
            message=message,
            llm_provider=error.llm_provider,
            model=error.model,
            response=error.response,
            litellm_debug_info=error.litellm_debug_info,
            max_retries=error.max_retries,
            num_retries=error.num_retries,
        )

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
    ) -> Any:
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

        messages = self._build_messages(prompt)

        if self._is_groq_provider():
            if self._groq_client is None:
                raise ValueError(
                    "Groq client is not initialized. Ensure GROQ_API_KEY is set and the 'groq' package is installed."
                )
            try:
                return self._groq_client.chat.completions.create(
                    **self._groq_create_kwargs(
                        messages=messages,
                        tool_schema=tool_schema,
                        tool_choice=tool_choice,
                        response_format=response_format,
                    )
                )
            except Exception as error:
                self._raise_groq_error(error)

        completion_kwargs = {
            "model": self.llm_model,
            "messages": messages,
            "tools": tool_schema,
            "tool_choice": tool_choice if tool_schema else None,
            "response_format": response_format,
        }
        if self.api_base:
            completion_kwargs["api_base"] = self.api_base

        try:
            response = completion(**completion_kwargs)
        except RateLimitError as error:
            raise self._build_rate_limit_error(error) from error

        return response

    async def agenerate(
        self,
        prompt: str | list[str] | None = None,
        tool_schema: list[dict] | None = None,
        tool_choice: str = "auto",
        response_format: dict | object | None = None,
    ) -> Any:
        """
        Asynchronous version of generate() method for parallel LLM calls.
        """
        messages = self._build_messages(prompt)
        if self._is_groq_provider():
            if self._agroq_client is not None:
                try:
                    return await self._agroq_client.chat.completions.create(
                        **self._groq_create_kwargs(
                            messages=messages,
                            tool_schema=tool_schema,
                            tool_choice=tool_choice,
                            response_format=response_format,
                        )
                    )
                except Exception as error:
                    self._raise_groq_error(error)
            if self._groq_client is None:
                raise ValueError(
                    "Groq client is not initialized. Ensure GROQ_API_KEY is set and the 'groq' package is installed."
                )
            try:
                return await asyncio.to_thread(
                    self._groq_client.chat.completions.create,
                    **self._groq_create_kwargs(
                        messages=messages,
                        tool_schema=tool_schema,
                        tool_choice=tool_choice,
                        response_format=response_format,
                    ),
                )
            except Exception as error:
                self._raise_groq_error(error)

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

                try:
                    response = await acompletion(**completion_kwargs)
                except RateLimitError as error:
                    raise self._build_rate_limit_error(error) from error
        return response
