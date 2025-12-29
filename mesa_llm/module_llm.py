import os

from dotenv import load_dotenv
from litellm import acompletion, completion, litellm
from litellm.exceptions import (
    APIConnectionError,
    RateLimitError,
    Timeout,
)
from rich.console import Console
from tenacity import AsyncRetrying, retry, retry_if_exception_type, wait_exponential

RETRYABLE_EXCEPTIONS = (
    APIConnectionError,
    Timeout,
    RateLimitError,
)

load_dotenv()
console = Console()


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
            llm_model: The model to use for the LLM in the format of {provider}/{LLM}
            api_base: The API base to use if the LLM provider is Ollama
            system_prompt: The system prompt to use for the LLM
        """
        self.api_base = api_base
        self.llm_model = llm_model
        self.system_prompt = system_prompt
        provider = self.llm_model.split("/")[0].upper()

        if provider in ["OLLAMA", "OLLAMA_CHAT"]:
            if self.api_base is None:
                self.api_base = "http://localhost:11434"
                console.print(
                    f"[yellow][Warning] Using default Ollama API base: {self.api_base}. If inference is not working, you may need to set the API base to the correct URL.[/yellow]"
                )
        else:
            try:
                self.api_key = os.environ[f"{provider}_API_KEY"]
            except KeyError as err:
                raise ValueError(
                    f"No API key found for {provider}. Please set the API key in the dotenv file."
                ) from err

        if not litellm.supports_function_calling(model=self.llm_model):
            console.print(
                f"[yellow][Warning]: {self.llm_model} does not support function calling. This model may not be able to use tools. Please check the model documentation at https://docs.litellm.ai/docs/providers for more information.[/yellow]"
            )

    def get_messages(self, prompt: str | list[str]) -> list[dict]:
        """
        Format the prompt messages for the LLM of the form : {"role": ..., "content": ...}

        Args:
            prompt: The prompt to generate a response for

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
                # Use extend to add all prompts from the list
                messages.extend([{"role": "user", "content": p} for p in prompt])

        return messages

    @retry(
        wait=wait_exponential(multiplier=1, min=1, max=60),
        retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
        reraise=True,
    )
    def generate(
        self,
        prompt: str | list[str],
        tool_schema: list[dict] | None = None,
        tool_choice: str = "auto",
        response_format: dict | object | None = None,
    ) -> str:
        """
        Generate a response from the LLM using litellm based on the prompt

        Args:
            prompt: The prompt to generate a response for
            tool_schema: The schema of the tools to use
            tool_choice: The choice of tool to use
            response_format: The format of the response

        Returns:
            The response from the LLM
        """

        messages = self.get_messages(prompt)

        # If api_base is provided, use it to override the default API base
        if self.api_base:
            response = completion(
                model=self.llm_model,
                messages=messages,
                api_base=self.api_base,
                tools=tool_schema,
                tool_choice=tool_choice if tool_schema else None,
                response_format=response_format,
            )

        # Otherwise, use the default API base
        else:
            response = completion(
                model=self.llm_model,
                messages=messages,
                tools=tool_schema,
                tool_choice=tool_choice if tool_schema else None,
                response_format=response_format,
            )

        return response

    async def agenerate(
        self,
        prompt: str | list[str],
        tool_schema: list[dict] | None = None,
        tool_choice: str = "auto",
        response_format: dict | object | None = None,
    ) -> str:
        """
        Asynchronous version of generate() method for parallel LLM calls.
        """
        messages = self.get_messages(prompt)
        async for attempt in AsyncRetrying(
            wait=wait_exponential(multiplier=1, min=1, max=60),
            retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
            reraise=True,
        ):
            with attempt:
                response = await acompletion(
                    model=self.llm_model,
                    messages=messages,
                    tools=tool_schema,
                    tool_choice=tool_choice if tool_schema else None,
                    response_format=response_format,
                )
        return response
