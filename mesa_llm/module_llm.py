import asyncio
import contextlib
import hashlib
import logging
import threading
import time
from collections import deque
from typing import Any

from dotenv import load_dotenv
from litellm import acompletion
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
        if self._async_sem is None:
            self._async_sem = asyncio.Semaphore(self.requests_per_second)
        await self._async_sem.acquire()
        delay = 1.0 / float(self.requests_per_second)
        asyncio.get_running_loop().call_later(delay, self._safe_release_async)

    def _safe_release_async(self) -> None:
        if self._async_sem is None:
            return
        with contextlib.suppress(ValueError):
            self._async_sem.release()

    # ... (rest of the code remains the same)

    def get_performance_stats(self) -> dict:
        """Get performance statistics."""
        stats = {
            "request_count": self.request_count,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": self.cache_hits / max(1, self.request_count),
            "batch_count": self.batch_count,
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
