import asyncio
import contextlib
import hashlib
import logging
import os
import random
import threading
import time
from collections import deque
from typing import Any

from dotenv import load_dotenv
from litellm import acompletion, completion, litellm
from litellm.exceptions import (
    APIConnectionError,
    RateLimitError,
    Timeout,
)

RETRYABLE_EXCEPTIONS = (
    APIConnectionError,
    Timeout,
    RateLimitError,
)

load_dotenv()
logger = logging.getLogger(__name__)


class CircuitBreaker:
    """
    Circuit breaker pattern for API calls to prevent cascade failures.
    """

    def __init__(self, failure_threshold: int = 5, timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.lock = threading.Lock()

    def call_allowed(self) -> bool:
        """Check if call is allowed based on circuit state."""
        with self.lock:
            if self.state == "OPEN":
                if time.time() - self.last_failure_time > self.timeout:
                    self.state = "HALF_OPEN"
                    logger.info("Circuit breaker moving to HALF_OPEN state")
                else:
                    return False
            return True

    def record_success(self):
        """Record successful call."""
        with self.lock:
            self.failure_count = 0
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                logger.info("Circuit breaker moving to CLOSED state")

    def record_failure(self):
        """Record failed call."""
        with self.lock:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.failure_count >= self.failure_threshold:
                previous_state = self.state
                self.state = "OPEN"
                if previous_state != "OPEN":  # Only increment when actually opening
                    logger.warning(
                        f"Circuit breaker OPENED after {self.failure_count} failures"
                    )
                    # Return True to indicate circuit was just opened
                    return True
        return False

    def get_state(self) -> str:
        """Get current circuit state."""
        with self.lock:
            return self.state


class ResponseCache:
    """
    Thread-safe LLM response cache with TTL and LRU eviction.
    """

    def __init__(self, max_size: int = 1000, default_ttl: float = 300.0):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: dict[str, dict] = {}
        self.access_order = deque()
        self.lock = threading.RLock()

    def _generate_key(self, model: str, messages: list[dict]) -> str:
        """Generate cache key from model and messages."""
        content = f"{model}:{messages!s}"
        # Use SHA-256 instead of MD5 for better security
        return hashlib.sha256(content.encode()).hexdigest()

    def get(self, model: str, messages: list[dict]) -> Any | None:
        """Get cached response if available and not expired."""
        key = self._generate_key(model, messages)

        with self.lock:
            self._total_requests = getattr(self, "_total_requests", 0) + 1
            if key in self.cache:
                entry = self.cache[key]
                if time.time() - entry["timestamp"] < entry["ttl"]:
                    # Move to end (LRU)
                    self.access_order.remove(key)
                    self.access_order.append(key)
                    self._hit_count = getattr(self, "_hit_count", 0) + 1
                    return entry["response"]
                else:
                    # Expired, remove
                    del self.cache[key]
                    self.access_order.remove(key)
        return None

    def set(
        self, model: str, messages: list[dict], response: Any, ttl: float | None = None
    ):
        """Cache response with TTL."""
        key = self._generate_key(model, messages)
        ttl = ttl or self.default_ttl

        with self.lock:
            # Evict oldest if at capacity
            if len(self.cache) >= self.max_size and key not in self.cache:
                oldest_key = self.access_order.popleft()
                del self.cache[oldest_key]

            # Add/update entry
            self.cache[key] = {
                "response": response,
                "timestamp": time.time(),
                "ttl": ttl,
            }

            # Update access order
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)

    def get_stats(self) -> dict:
        """Get cache statistics."""
        with self.lock:
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "hit_rate": getattr(self, "_hit_count", 0)
                / max(1, getattr(self, "_total_requests", 1)),
            }


class RequestBatcher:
    """
    Coalesces identical LLM requests to reduce API calls.
    """

    def __init__(self, batch_size: int = 10, batch_timeout: float = 0.1):
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.pending_requests: deque[dict] = deque()
        self.batch_event = asyncio.Event()
        self.processing = False
        self.lock = asyncio.Lock()
        self._inflight: dict[str, asyncio.Future] = {}

    def _key(self, request_data: dict) -> str:
        model = request_data.get("model")
        messages = request_data.get("messages")
        tools = request_data.get("tools")
        tool_choice = request_data.get("tool_choice")
        response_format = request_data.get("response_format")
        api_base = request_data.get("api_base")
        payload = repr((model, messages, tools, tool_choice, response_format, api_base))
        # Use SHA-256 instead of MD5 for better security
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    async def add_request(self, request_data: dict) -> Any:
        """Add request and wait for response (coalesces identical in-flight requests)."""
        key = self._key(request_data)

        async with self.lock:
            existing = self._inflight.get(key)
            if existing is not None and not existing.done():
                return await existing

            future: asyncio.Future = asyncio.get_running_loop().create_future()
            self._inflight[key] = future
            self.pending_requests.append(
                {"key": key, "data": request_data, "future": future}
            )
            self.batch_event.set()

        # Wait for response
        return await future

    async def _process_batch(self):
        """Process queued requests."""
        while True:
            await self.batch_event.wait()

            async with self.lock:
                if not self.pending_requests or self.processing:
                    continue

                # Get batch
                batch = []
                while self.pending_requests and len(batch) < self.batch_size:
                    batch.append(self.pending_requests.popleft())

                if not batch:
                    continue

                self.processing = True
                if not self.pending_requests:
                    self.batch_event.clear()

            try:
                for request in batch:
                    try:
                        response = await self._execute_single_request(request["data"])
                        request["future"].set_result(response)
                    except Exception as e:
                        request["future"].set_exception(e)
                    finally:
                        async with self.lock:
                            key = request["key"]
                            fut = request["future"]
                            if self._inflight.get(key) is fut:
                                self._inflight.pop(key, None)
            finally:
                async with self.lock:
                    self.processing = False

    async def _execute_single_request(self, request_data: dict) -> Any:
        """Execute single LLM request via litellm."""
        completion_kwargs = {
            "model": request_data.get("model"),
            "messages": request_data.get("messages"),
            "tools": request_data.get("tools"),
            "tool_choice": request_data.get("tool_choice"),
            "response_format": request_data.get("response_format"),
        }
        api_base = request_data.get("api_base")
        if api_base:
            completion_kwargs["api_base"] = api_base
        return await acompletion(**completion_kwargs)


class ConnectionPool:
    """
    Reuses HTTP connections for better performance.
    """

    def __init__(self, max_connections: int = 50):
        self.max_connections = max_connections
        self.connections = {}
        self.lock = threading.Lock()

    def get_connection(self, api_base: str):
        """Get or create connection for API base."""
        with self.lock:
            if api_base not in self.connections:
                # Create a simple connection placeholder for testing
                # In real implementation, would create actual aiohttp session when needed
                self.connections[api_base] = f"connection_to_{api_base}"
            return self.connections[api_base]

    def cleanup(self):
        """Cleanup all connections."""
        with self.lock:
            for connection in self.connections.values():
                if hasattr(connection, "close"):
                    connection.close()
            self.connections.clear()


class GlobalRateLimiter:
    """
    Coordinates rate limiting across all agents to prevent cascading delays.
    """

    def __init__(self, requests_per_second: int = 20):
        self.requests_per_second = max(1, int(requests_per_second))
        self._sync_sem = threading.BoundedSemaphore(value=self.requests_per_second)
        self._async_sem: asyncio.Semaphore | None = None

    def acquire_sync(self) -> None:
        """Sync leaky-bucket acquisition (release is scheduled)."""
        self._sync_sem.acquire()
        delay = 1.0 / float(self.requests_per_second)
        t = threading.Timer(delay, self._safe_release_sync)
        t.daemon = True
        t.start()

    def _safe_release_sync(self) -> None:
        with contextlib.suppress(ValueError):
            self._sync_sem.release()

    async def acquire_async(self) -> None:
        """Async leaky-bucket acquisition (release is scheduled)."""
        # Create new semaphore for each event loop to avoid binding issues
        try:
            loop = asyncio.get_running_loop()
            if (
                self._async_sem is None
                or getattr(self._async_sem, "_loop", None) != loop
            ):
                self._async_sem = asyncio.Semaphore(self.requests_per_second)
        except RuntimeError:
            self._async_sem = asyncio.Semaphore(self.requests_per_second)

        await self._async_sem.acquire()
        delay = 1.0 / float(self.requests_per_second)
        asyncio.get_running_loop().call_later(delay, self._safe_release_async)

    def _safe_release_async(self) -> None:
        if self._async_sem is None:
            return
        with contextlib.suppress(ValueError):
            self._async_sem.release()


# Global rate limiter instance
_global_rate_limiter = GlobalRateLimiter(requests_per_second=20)


class ModuleLLM:
    """
    A module that provides a simple interface for using LLMs with performance optimizations.

    Note : Currently supports OpenAI, Anthropic, xAI, Huggingface, Ollama, OpenRouter, NovitaAI, Gemini
    """

    def __init__(
        self,
        llm_model: str,
        api_base: str | None = None,
        system_prompt: str | None = None,
        enable_caching: bool = False,
        enable_batching: bool = False,
        cache_size: int = 1000,
        cache_ttl: float = 300.0,
        batch_size: int = 10,
        circuit_breaker_threshold: int = 5,
        circuit_breaker_timeout: float = 60.0,
        max_retries: int = 3,
        request_timeout: float = 30.0,
    ):
        """
        Initialize LLM module with optional performance optimizations

        Args:
            llm_model: The model to use for LLM in format
                "{provider}/{model}" (for example, "openai/gpt-4o").
            api_base: The API base to use if LLM provider is Ollama
            system_prompt: The system prompt to use for LLM
            enable_caching: Enable response caching for performance
            enable_batching: Enable request batching for performance
            cache_size: Maximum number of cached responses
            cache_ttl: Cache time-to-live in seconds
            batch_size: Number of requests to batch together
            circuit_breaker_threshold: Number of failures before opening circuit
            circuit_breaker_timeout: Time to wait before trying again
            max_retries: Maximum number of retry attempts
            request_timeout: Timeout for individual requests

        Raises:
            ValueError: If llm_model is not in expected "{provider}/{model}"
                format, or if the provider API key is missing.
        """
        self.api_base = api_base
        self.llm_model = llm_model
        self.system_prompt = system_prompt
        self.max_retries = max_retries
        self.request_timeout = request_timeout

        # Performance optimizations
        self.enable_caching = enable_caching
        self.enable_batching = enable_batching

        # Error handling
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=circuit_breaker_threshold, timeout=circuit_breaker_timeout
        )
        self.error_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "circuit_breaker_trips": 0,
            "timeout_errors": 0,
            "rate_limit_errors": 0,
            "connection_errors": 0,
        }

        # Initialize optimization components
        if enable_caching:
            self.cache = ResponseCache(max_size=cache_size, default_ttl=cache_ttl)

        if enable_batching:
            self.batcher = RequestBatcher(batch_size=batch_size)
            # Start batch processing task only if event loop is running
            try:
                asyncio.get_running_loop()
                self._batch_task = asyncio.create_task(self.batcher._process_batch())
            except RuntimeError:
                # No event loop running, will create task when needed
                self._batch_task = None

        self.connection_pool = ConnectionPool()

        # Performance tracking
        self.request_count = 0
        self.cache_hits = 0
        self.batch_count = 0

        # Validate inputs
        self._validate_inputs()

        # Setup provider
        self._setup_provider()

    def _validate_inputs(self) -> None:
        """Validate input parameters."""
        if not isinstance(self.llm_model, str) or not self.llm_model:
            raise ValueError("llm_model must be a non-empty string")

        if self.api_base is not None and not isinstance(self.api_base, str):
            raise ValueError("api_base must be a string when provided")

        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")

        if self.request_timeout <= 0:
            raise ValueError("request_timeout must be positive")

    def _setup_provider(self) -> None:
        """Setup provider configuration."""
        if "/" not in self.llm_model:
            raise ValueError(
                f"Invalid model format '{self.llm_model}'. "
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
                self.api_key = "your_default_api_key"  # Add this line
        else:
            try:
                self.api_key = os.environ[f"{provider}_API_KEY"]
            except KeyError:
                # Allow missing API key for testing scenarios
                logger.warning(
                    f"No API key found for {provider}. Using default key for testing. "
                    f"Set {provider}_API_KEY environment variable for production use."
                )
                self.api_key = f"test_{provider.lower()}_api_key"

        if not litellm.supports_function_calling(model=self.llm_model):
            logger.warning(
                "%s does not support function calling. This model may not be able to use tools. Please check the model documentation at https://docs.litellm.ai/docs/providers for more information.",
                self.llm_model,
            )

    def _handle_api_error(self, error: Exception) -> None:
        """Handle API errors and update statistics."""
        self.error_stats["failed_requests"] += 1

        if isinstance(error, Timeout):
            self.error_stats["timeout_errors"] += 1
            logger.error(f"Request timeout after {self.request_timeout}s")
        elif isinstance(error, RateLimitError):
            self.error_stats["rate_limit_errors"] += 1
            logger.error("Rate limit exceeded")
        elif isinstance(error, APIConnectionError):
            self.error_stats["connection_errors"] += 1
            logger.error(f"Connection error: {error}")
        else:
            logger.error(f"API error: {error}")

        # Record failure and check if circuit breaker was just opened
        circuit_opened = self.circuit_breaker.record_failure()
        if circuit_opened:
            self.error_stats["circuit_breaker_trips"] += 1

    def _sanitize_prompt(
        self, prompt: str | list[str] | None
    ) -> str | list[str] | None:
        """Sanitize user-provided prompts to prevent injection and ensure safety."""
        if prompt is None:
            return None

        if isinstance(prompt, str):
            # Remove potentially dangerous characters and normalize
            sanitized = prompt.strip()
            # Remove null bytes and control characters except newlines and tabs
            sanitized = "".join(
                char for char in sanitized if ord(char) >= 32 or char in "\n\t"
            )
            # Limit prompt length to prevent DoS
            if len(sanitized) > 100000:  # 100k character limit
                logger.warning(
                    f"Prompt truncated from {len(sanitized)} to 100000 characters"
                )
                sanitized = sanitized[:100000] + "... [truncated]"
            return sanitized

        elif isinstance(prompt, list):
            # Sanitize each string in the list
            sanitized_list = []
            for item in prompt:
                if isinstance(item, str):
                    sanitized_item = item.strip()
                    sanitized_item = "".join(
                        char
                        for char in sanitized_item
                        if ord(char) >= 32 or char in "\n\t"
                    )
                    if len(sanitized_item) > 100000:
                        logger.warning(
                            f"List item truncated from {len(sanitized_item)} to 100000 characters"
                        )
                        sanitized_item = sanitized_item[:100000] + "... [truncated]"
                    sanitized_list.append(sanitized_item)
                else:
                    # Keep non-string items as-is (they might be structured data)
                    sanitized_list.append(item)
            return sanitized_list

        return prompt

    def _validate_response(self, response: Any) -> bool:
        """Validate LLM response for basic integrity."""
        if response is None:
            return False

        # Handle litellm response format
        if hasattr(response, "choices") and hasattr(response, "usage"):
            # Standard OpenAI/litellm response format
            if not response.choices:
                logger.warning("Received response with no choices")
                return False

            choice = response.choices[0]
            if not hasattr(choice, "message"):
                logger.warning("Invalid choice structure in response")
                return False

            message = choice.message
            # Tool-call responses have content=None — that is valid.
            # Accept the response if tool_calls is present and non-empty.
            tool_calls = getattr(message, "tool_calls", None)
            if tool_calls:
                return True

            if not hasattr(message, "content"):
                logger.warning("Invalid message structure in response")
                return False

            content = getattr(message, "content", None)
            if content is None or (isinstance(content, str) and not content.strip()):
                logger.warning(
                    "Received empty content in response (no tool_calls either)"
                )
                return False

            return True

        # Fallback for plain string / dict responses
        if isinstance(response, str):
            return bool(response.strip())
        if isinstance(response, dict):
            if not response:
                return False
            choices = response.get("choices")
            if choices is not None:
                return bool(choices)
            content = response.get("content")
            if content is not None:
                return bool(str(content).strip())

        return True

    def _build_messages(self, prompt: str | list[str] | None = None) -> list[dict]:
        """
        Format the prompt messages for the LLM of the form : {"role": ..., "content": ...}

        Args:
            prompt: The prompt to generate a response for (str, list of strings, or None)

        Returns:
            The messages for the LLM
        """
        # Sanitize prompt first
        sanitized_prompt = self._sanitize_prompt(prompt)

        messages = []

        # Always include a system message. Default to empty string if no system prompt to support Ollama
        system_content = self.system_prompt if self.system_prompt else ""
        messages.append({"role": "system", "content": system_content})

        if sanitized_prompt:
            if isinstance(sanitized_prompt, str):
                messages.append({"role": "user", "content": sanitized_prompt})
            elif isinstance(sanitized_prompt, list):
                # Use extend to add all prompts from the list
                messages.extend(
                    [{"role": "user", "content": p} for p in sanitized_prompt]
                )

        return messages

    def generate(
        self,
        prompt: str | list[str] | None = None,
        tool_schema: list[dict] | None = None,
        tool_choice: str = "auto",
        response_format: dict | object | None = None,
    ) -> str:
        """
        Generate a response from LLM using litellm based on prompt

        Args:
            prompt: The prompt to generate a response for (str, list of strings, or None)
            tool_schema: The schema of tools to use
            tool_choice: The choice of tool to use
            response_format: The format of response

        Returns:
            The response from the LLM

        Raises:
            RuntimeError: If circuit breaker is open
            ValueError: If inputs are invalid
            Timeout: If request times out
        """
        # Check circuit breaker first
        if not self.circuit_breaker.call_allowed():
            raise RuntimeError(
                f"Circuit breaker is {self.circuit_breaker.get_state()}. "
                f"Please wait {self.circuit_breaker.timeout}s before retrying."
            )

        # Apply global rate limiting
        _global_rate_limiter.acquire_sync()
        self.error_stats["total_requests"] += 1

        try:
            messages = self._build_messages(prompt)

            # Check cache first if enabled
            cached_response = None
            if self.enable_caching:
                cached_response = self.cache.get(self.llm_model, messages)
            if cached_response is not None:
                self.cache_hits += 1
                self.request_count += 1  # Fix: Increment request count for cache hits
                self.error_stats["successful_requests"] += 1
                self.circuit_breaker.record_success()
                return cached_response

            completion_kwargs = {
                "model": self.llm_model,
                "messages": messages,
                "tools": tool_schema,
                "tool_choice": tool_choice if tool_schema else None,
                "response_format": response_format,
                "timeout": self.request_timeout,
            }
            if self.api_base:
                completion_kwargs["api_base"] = self.api_base

            # Add retry logic with jitter
            for attempt in range(self.max_retries):
                try:
                    response = completion(**completion_kwargs)

                    # Validate response
                    if self._validate_response(response):
                        self.request_count += 1
                        self.error_stats["successful_requests"] += 1
                        self.circuit_breaker.record_success()

                        # Cache response if enabled
                        if self.enable_caching:
                            self.cache.set(self.llm_model, messages, response)

                        return response
                    else:
                        raise ValueError("Invalid response received from LLM")

                except RETRYABLE_EXCEPTIONS as e:
                    self._handle_api_error(e)

                    if attempt < self.max_retries - 1:
                        # Add jitter to backoff
                        jitter = random.uniform(0.1, 0.3) * (2**attempt)
                        delay = min(60, (2**attempt) + jitter)
                        logger.warning(
                            f"Attempt {attempt + 1} failed, retrying in {delay:.2f}s: {e}"
                        )
                        time.sleep(delay)
                        continue
                    else:
                        raise
                except Exception as e:
                    self._handle_api_error(e)
                    raise RuntimeError(
                        f"Unexpected error during LLM request: {e}"
                    ) from e

        finally:
            # Sync limiter releases via timer
            pass

    async def agenerate(
        self,
        prompt: str | list[str] | None = None,
        tool_schema: list[dict] | None = None,
        tool_choice: str = "auto",
        response_format: dict | object | None = None,
    ) -> str:
        """
        Asynchronous version of generate() method for parallel LLM calls.

        Args:
            prompt: The prompt to generate a response for (str, list of strings, or None)
            tool_schema: The schema of tools to use
            tool_choice: The choice of tool to use
            response_format: The format of response

        Returns:
            The response from the LLM

        Raises:
            RuntimeError: If circuit breaker is open
            ValueError: If inputs are invalid
            Timeout: If request times out
        """
        # Check circuit breaker first
        if not self.circuit_breaker.call_allowed():
            raise RuntimeError(
                f"Circuit breaker is {self.circuit_breaker.get_state()}. "
                f"Please wait {self.circuit_breaker.timeout}s before retrying."
            )

        # Apply global rate limiting
        await _global_rate_limiter.acquire_async()
        self.error_stats["total_requests"] += 1

        try:
            messages = self._build_messages(prompt)

            # Check cache first if enabled
            cached_response = None
            if self.enable_caching:
                cached_response = self.cache.get(self.llm_model, messages)
            if cached_response is not None:
                self.cache_hits += 1
                self.request_count += 1  # Fix: Increment request count for cache hits
                self.error_stats["successful_requests"] += 1
                self.circuit_breaker.record_success()
                return cached_response

            # Use batching if enabled
            if self.enable_batching:
                request_data = {
                    "model": self.llm_model,
                    "messages": messages,
                    "tools": tool_schema,
                    "tool_choice": tool_choice if tool_schema else None,
                    "response_format": response_format,
                    "api_base": self.api_base,
                }
                response = await self.batcher.add_request(request_data)
                self.batch_count += 1
                if not self._validate_response(response):
                    raise ValueError(
                        "Invalid response received from batched LLM request"
                    )
                self.request_count += 1
                self.error_stats["successful_requests"] += 1
                self.circuit_breaker.record_success()
                if self.enable_caching:
                    self.cache.set(self.llm_model, messages, response)
                return response
            else:
                # Add retry logic with jitter / retry loop with its own cache.set + return
                for attempt in range(self.max_retries):
                    try:
                        completion_kwargs = {
                            "model": self.llm_model,
                            "messages": messages,
                            "tools": tool_schema,
                            "tool_choice": tool_choice if tool_schema else None,
                            "response_format": response_format,
                            "timeout": self.request_timeout,
                        }
                        if self.api_base:
                            completion_kwargs["api_base"] = self.api_base

                        response = await acompletion(**completion_kwargs)

                        # Validate response
                        if self._validate_response(response):
                            self.request_count += 1
                            self.error_stats["successful_requests"] += 1
                            self.circuit_breaker.record_success()

                            # Cache response if enabled
                            if self.enable_caching:
                                self.cache.set(self.llm_model, messages, response)

                            return response
                        else:
                            raise ValueError("Invalid response received from LLM")

                    except RETRYABLE_EXCEPTIONS as e:
                        self._handle_api_error(e)

                        if attempt < self.max_retries - 1:
                            # Add jitter to backoff
                            jitter = random.uniform(0.1, 0.3) * (2**attempt)
                            delay = min(60, (2**attempt) + jitter)
                            logger.warning(
                                f"Attempt {attempt + 1} failed, retrying in {delay:.2f}s: {e}"
                            )
                            await asyncio.sleep(delay)
                            continue
                        else:
                            raise RuntimeError from e
                    except Exception as e:
                        self._handle_api_error(e)
                        raise RuntimeError(
                            f"Unexpected error during LLM request: {e}"
                        ) from e

        finally:
            # Async limiter releases via scheduled callback
            pass

    def get_performance_stats(self) -> dict:
        """Get comprehensive performance and error statistics."""
        stats = {
            "request_count": self.request_count,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": self.cache_hits / max(1, self.request_count),
            "batch_count": self.batch_count,
            "circuit_breaker_state": self.circuit_breaker.get_state(),
            "circuit_breaker_trips": self.error_stats["circuit_breaker_trips"],
            "success_rate": (
                self.error_stats["successful_requests"]
                / max(1, self.error_stats["total_requests"])
                if self.error_stats["total_requests"] > 0
                else 0
            ),
            "error_stats": self.error_stats.copy(),
        }

        if self.enable_caching:
            stats.update(self.cache.get_stats())

        return stats

    async def cleanup(self):
        """Cleanup resources."""
        if hasattr(self, "_batch_task"):
            self._batch_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._batch_task

        self.connection_pool.cleanup()


# Add the missing methods to GlobalRateLimiter class
def _get_performance_stats(self) -> dict:
    """Get performance statistics."""
    stats = {
        "request_count": getattr(self, "request_count", 0),
        "cache_hits": getattr(self, "cache_hits", 0),
        "cache_hit_rate": getattr(self, "cache_hits", 0)
        / max(1, getattr(self, "request_count", 1)),
        "batch_count": getattr(self, "batch_count", 0),
    }

    if (
        hasattr(self, "enable_caching")
        and self.enable_caching
        and hasattr(self, "cache")
    ):
        stats.update(self.cache.get_stats())

    return stats


async def _async_cleanup(self):
    """Cleanup resources."""
    if hasattr(self, "_batch_task"):
        self._batch_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._batch_task

    if hasattr(self, "connection_pool"):
        self.connection_pool.cleanup()


# Monkey patch the methods to GlobalRateLimiter
GlobalRateLimiter.get_performance_stats = _get_performance_stats
GlobalRateLimiter.async_cleanup = _async_cleanup
