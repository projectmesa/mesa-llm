import asyncio
import hashlib
import logging
import os
import threading
import time
from collections import deque
from typing import Any, Dict, List, Optional

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


class ResponseCache:
    """
    Thread-safe LLM response cache with TTL and LRU eviction.
    """
    
    def __init__(self, max_size: int = 1000, default_ttl: float = 300.0):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: Dict[str, Dict] = {}
        self.access_order = deque()
        self.lock = threading.RLock()
    
    def _generate_key(self, model: str, messages: List[Dict]) -> str:
        """Generate cache key from model and messages."""
        content = f"{model}:{str(messages)}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(self, model: str, messages: List[Dict]) -> Optional[Any]:
        """Get cached response if available and not expired."""
        key = self._generate_key(model, messages)
        
        with self.lock:
            self._total_requests = getattr(self, "_total_requests", 0) + 1
            if key in self.cache:
                entry = self.cache[key]
                if time.time() - entry['timestamp'] < entry['ttl']:
                    # Move to end (LRU)
                    self.access_order.remove(key)
                    self.access_order.append(key)
                    self._hit_count = getattr(self, "_hit_count", 0) + 1
                    return entry['response']
                else:
                    # Expired, remove
                    del self.cache[key]
                    self.access_order.remove(key)
        return None
    
    def set(self, model: str, messages: List[Dict], response: Any, ttl: Optional[float] = None):
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
                'response': response,
                'timestamp': time.time(),
                'ttl': ttl
            }
            
            # Update access order
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)
    
    def get_stats(self) -> Dict:
        """Get cache statistics."""
        with self.lock:
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'hit_rate': getattr(self, '_hit_count', 0) / max(1, getattr(self, '_total_requests', 1))
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
    
    def _key(self, request_data: Dict) -> str:
        model = request_data.get("model")
        messages = request_data.get("messages")
        tools = request_data.get("tools")
        tool_choice = request_data.get("tool_choice")
        response_format = request_data.get("response_format")
        api_base = request_data.get("api_base")
        payload = repr((model, messages, tools, tool_choice, response_format, api_base))
        return hashlib.md5(payload.encode("utf-8")).hexdigest()
    
    async def add_request(self, request_data: Dict) -> Any:
        """Add request and wait for response (coalesces identical in-flight requests)."""
        key = self._key(request_data)
        
        async with self.lock:
            existing = self._inflight.get(key)
            if existing is not None and not existing.done():
                return await existing

            future: asyncio.Future = asyncio.get_running_loop().create_future()
            self._inflight[key] = future
            self.pending_requests.append({"key": key, "data": request_data, "future": future})
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
    
    async def _execute_single_request(self, request_data: Dict) -> Any:
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
                if hasattr(connection, 'close'):
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
        try:
            self._sync_sem.release()
        except ValueError:
            pass

    async def acquire_async(self) -> None:
        """Async leaky-bucket acquisition (release is scheduled)."""
        if self._async_sem is None:
            self._async_sem = asyncio.Semaphore(self.requests_per_second)
        await self._async_sem.acquire()
        delay = 1.0 / float(self.requests_per_second)
        asyncio.get_running_loop().call_later(delay, self._safe_release_async)

    def _safe_release_async(self) -> None:
        if self._async_sem is None:
            return
        try:
            self._async_sem.release()
        except ValueError:
            pass


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

        Raises:
            ValueError: If llm_model is not in the expected "{provider}/{model}"
                format, or if the provider API key is missing.
        """
        self.api_base = api_base
        self.llm_model = llm_model
        self.system_prompt = system_prompt
        
        # Performance optimizations
        self.enable_caching = enable_caching
        self.enable_batching = enable_batching
        
        # Initialize optimization components
        if enable_caching:
            self.cache = ResponseCache(max_size=cache_size, default_ttl=cache_ttl)
        
        if enable_batching:
            self.batcher = RequestBatcher(batch_size=batch_size)
            # Start batch processing task only if event loop is running
            try:
                loop = asyncio.get_running_loop()
                self._batch_task = asyncio.create_task(self.batcher._process_batch())
            except RuntimeError:
                # No event loop running, will create task when needed
                self._batch_task = None
        
        self.connection_pool = ConnectionPool()
        
        # Performance tracking
        self.request_count = 0
        self.cache_hits = 0
        self.batch_count = 0

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
                # Use extend to add all prompts from the list
                messages.extend([{"role": "user", "content": p} for p in prompt])

        return messages

    @retry(
        wait=wait_exponential(multiplier=1.1, min=1, max=5),  # Gentler backoff
        retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
        reraise=True,
    )
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
        """
        # Apply global rate limiting
        _global_rate_limiter.acquire_sync()
        try:
            self.request_count += 1
            messages = self._build_messages(prompt)

            # Check cache first if enabled
            cached_response = None
            if self.enable_caching:
                cached_response = self.cache.get(self.llm_model, messages)
            if cached_response is not None:
                self.cache_hits += 1
                return cached_response

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

            # Cache response if enabled
            if self.enable_caching:
                self.cache.set(self.llm_model, messages, response)

            return response
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
        """
        # Apply global rate limiting
        await _global_rate_limiter.acquire_async()
        try:
            self.request_count += 1
            messages = self._build_messages(prompt)

            # Check cache first if enabled
            cached_response = None
            if self.enable_caching:
                cached_response = self.cache.get(self.llm_model, messages)
            if cached_response is not None:
                self.cache_hits += 1
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
            else:
                async for attempt in AsyncRetrying(
                    wait=wait_exponential(multiplier=1.1, min=1, max=5),  # Gentler backoff
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

            # Cache response if enabled
            if self.enable_caching:
                self.cache.set(self.llm_model, messages, response)

            return response
        finally:
            # Async limiter releases via scheduled callback
            pass

    def get_performance_stats(self) -> Dict:
        """Get performance statistics."""
        stats = {
            'request_count': self.request_count,
            'cache_hits': self.cache_hits,
            'cache_hit_rate': self.cache_hits / max(1, self.request_count),
            'batch_count': self.batch_count,
        }
        
        if self.enable_caching:
            stats.update(self.cache.get_stats())
        
        return stats

    async def cleanup(self):
        """Cleanup resources."""
        if hasattr(self, '_batch_task'):
            self._batch_task.cancel()
            try:
                await self._batch_task
            except asyncio.CancelledError:
                pass
        
        self.connection_pool.cleanup()
