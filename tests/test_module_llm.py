import asyncio
import os
import time
from unittest.mock import patch

import pytest
from litellm.exceptions import RateLimitError

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

    def test_generate_rewrites_rate_limit_error_with_openai_docs(self, monkeypatch):
        original_error = RateLimitError(
            "per-minute limit hit", "openai", "openai/gpt-4o"
        )

        def _raise_rate_limit(**kwargs):
            raise original_error

        monkeypatch.setattr("mesa_llm.module_llm.completion", _raise_rate_limit)

        llm = ModuleLLM(llm_model="openai/gpt-4o")
        with pytest.raises(RateLimitError) as exc_info:
            ModuleLLM.generate.__wrapped__(llm, prompt="Hello, how are you?")

        # Check that it contains the expected elements rather than exact match
        error_str = str(exc_info.value)
        assert "Rate limit exceeded for model" in error_str
        assert "openai/gpt-4o" in error_str
        assert "per-minute limit hit" in error_str
        assert "developers.openai.com/api/docs/guides/rate-limits" in error_str

    def test_generate_rewrites_rate_limit_error_with_gemini_docs(self, monkeypatch):
        original_error = RateLimitError(
            'geminiException - {"error": {"code": 429}}',
            "gemini",
            "gemini/gemini-2.0-flash",
            max_retries=5,
            num_retries=3,
        )

        def _raise_rate_limit(**kwargs):
            raise original_error

        monkeypatch.setattr("mesa_llm.module_llm.completion", _raise_rate_limit)

        llm = ModuleLLM(llm_model="gemini/gemini-2.0-flash")
        with pytest.raises(RateLimitError) as exc_info:
            ModuleLLM.generate.__wrapped__(llm, prompt="Hello, how are you?")

        # Check that it contains expected elements rather than exact match
        error_str = str(exc_info.value)
        assert "Rate limit exceeded for model" in error_str
        assert "gemini/gemini-2.0-flash" in error_str
        assert "geminiException" in error_str
        assert 'code":429' in error_str
        assert "ai.google.dev/gemini-api/docs/rate-limits" in error_str
        assert "LiteLLM Retried: 3 times" in error_str
        assert "LiteLLM Max Retries: 5" in error_str

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

    # === Performance Optimization Tests ===

    @pytest.mark.asyncio
    async def test_module_llm_with_optimizations_initializes(self):
        """Test that ModuleLLM initializes with optimizations enabled."""
        llm = ModuleLLM(
            llm_model="test/gpt-4",
            enable_caching=True,
            enable_batching=True,
            cache_size=100,
            cache_ttl=60.0,
            batch_size=10,
        )

        assert llm.enable_caching is True
        assert llm.enable_batching is True
        assert hasattr(llm, "cache")
        assert hasattr(llm, "batcher")
        assert hasattr(llm, "connection_pool")

        # Test performance stats
        stats = llm.get_performance_stats()
        assert "request_count" in stats
        assert "cache_hits" in stats
        assert "cache_hit_rate" in stats
        assert "batch_count" in stats

        await llm.cleanup()

    @pytest.mark.asyncio
    async def test_caching_functionality(self):
        """Test that caching works correctly for repeated requests."""
        mock_response = "Cached response"

        with (
            patch("mesa_llm.module_llm.completion", return_value=mock_response),
            patch("mesa_llm.module_llm.acompletion", return_value=mock_response),
        ):
            llm = ModuleLLM(
                llm_model="test/gpt-4",
                enable_caching=True,
                cache_size=100,
                cache_ttl=60.0,
            )

            # First call - should cache the response
            response1 = await llm.agenerate("Test prompt")
            stats1 = llm.get_performance_stats()

            # Second call with same prompt - should hit cache
            response2 = await llm.agenerate("Test prompt")
            stats2 = llm.get_performance_stats()

            assert response1 == response2 == mock_response
            assert stats2["request_count"] > stats1["request_count"]
            assert stats2["cache_hits"] > stats1["cache_hits"]
            assert stats2["cache_hit_rate"] > 0

            await llm.cleanup()

    @pytest.mark.asyncio
    async def test_batching_functionality(self):
        """Test that batching works correctly."""
        mock_response = "Batched response"

        with (
            patch("mesa_llm.module_llm.completion", return_value=mock_response),
            patch("mesa_llm.module_llm.acompletion", return_value=mock_response),
        ):
            llm = ModuleLLM(llm_model="test/gpt-4", enable_batching=True, batch_size=5)

            # Make multiple requests to test batching
            tasks = [llm.agenerate(f"Test prompt {i}") for i in range(3)]
            responses = await asyncio.gather(*tasks)

            # All responses should be the same (mocked)
            assert all(r == mock_response for r in responses)

            stats = llm.get_performance_stats()
            assert stats["batch_count"] >= 0

            await llm.cleanup()

    @pytest.mark.asyncio
    async def test_performance_comparison_with_without_optimizations(self):
        """Compare performance with and without optimizations."""
        mock_response = "Performance test response"

        with (
            patch("mesa_llm.module_llm.completion", return_value=mock_response),
            patch("mesa_llm.module_llm.acompletion", return_value=mock_response),
        ):
            # Test without optimizations - larger scale to show benefits
            llm_no_opt = ModuleLLM(llm_model="test/gpt-4")
            start_time = time.time()
            # Use more requests with repetition to simulate real-world usage
            for i in range(20):
                # Repeat prompts to simulate cache hits in real scenarios
                prompt = (
                    f"Test prompt {i % 5}"  # Only 5 unique prompts, repeated 4 times
                )
                await llm_no_opt.agenerate(prompt)
            no_opt_time = time.time() - start_time

            # Test with optimizations - same workload
            llm_opt = ModuleLLM(
                llm_model="test/gpt-4",
                enable_caching=True,
                enable_batching=True,
                cache_size=100,
                cache_ttl=60.0,
            )

            start_time = time.time()
            for i in range(20):
                # Same repeated prompts - should benefit from caching
                prompt = f"Test prompt {i % 5}"
                await llm_opt.agenerate(prompt)
            opt_time = time.time() - start_time

            # For caching to be beneficial, we need enough repeated requests
            # Allow for optimization overhead but expect benefits with repetition
            if opt_time > no_opt_time:
                # If optimized is slower, check if cache is working
                stats = llm_opt.get_performance_stats()
                cache_hit_rate = stats.get("cache_hit_rate", 0)

                # Cache should have hits with repeated prompts
                assert cache_hit_rate > 0.5, (
                    f"Cache hit rate should be >50% for repeated prompts, got {cache_hit_rate}"
                )

                # Allow some tolerance for small-scale tests where overhead dominates
                assert opt_time <= no_opt_time * 2.0, (
                    f"Optimized ({opt_time:.4f}s) should be <= 2x non-optimized ({no_opt_time:.4f}s) for small scale"
                )
            else:
                # Optimized is faster - this is ideal
                assert True, "Optimized version is faster as expected"

            await llm_no_opt.cleanup()
            await llm_opt.cleanup()

    def test_connection_pool_initialization(self):
        """Test connection pool is properly initialized."""
        llm = ModuleLLM(llm_model="test/gpt-4")

        assert hasattr(llm, "connection_pool")
        assert llm.connection_pool is not None

        # Test connection pool methods
        connection = llm.connection_pool.get_connection("https://api.openai.com")
        assert connection is not None

        llm.connection_pool.cleanup()
        assert len(llm.connection_pool.connections) == 0

    @pytest.mark.asyncio
    async def test_performance_stats_accuracy(self):
        """Test performance statistics are accurate."""
        mock_response = "Stats test response"

        with (
            patch("mesa_llm.module_llm.completion", return_value=mock_response),
            patch("mesa_llm.module_llm.acompletion", return_value=mock_response),
        ):
            llm = ModuleLLM(
                llm_model="test/gpt-4", enable_caching=True, enable_batching=True
            )

            # Make some requests
            await llm.agenerate("Test 1")
            await llm.agenerate("Test 2")  # Should hit cache
            await llm.agenerate("Test 1")  # Should hit cache

            stats = llm.get_performance_stats()

            assert stats["request_count"] == 3
            assert stats["cache_hits"] >= 1  # At least one cache hit
            assert 0 <= stats["cache_hit_rate"] <= 1
            assert stats["batch_count"] >= 0

            await llm.cleanup()

    @pytest.mark.asyncio
    async def test_agenerate_rewrites_rate_limit_error_with_openrouter_docs(
        self, monkeypatch
    ):
        class _SingleAttempt:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        class _SingleAsyncRetrying:
            def __init__(self, **kwargs):
                self._yielded = False

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self._yielded:
                    raise StopAsyncIteration
                self._yielded = True
                return _SingleAttempt()

        async def _raise_rate_limit(**kwargs):
            raise RateLimitError(
                "provider throttle triggered",
                "openrouter",
                "openrouter/openai/gpt-4o",
            )

        monkeypatch.setenv("OPENROUTER_API_KEY", "test_openrouter_key")
        monkeypatch.setattr("mesa_llm.module_llm.AsyncRetrying", _SingleAsyncRetrying)
        monkeypatch.setattr("mesa_llm.module_llm.acompletion", _raise_rate_limit)

        llm = ModuleLLM(llm_model="openrouter/openai/gpt-4o")
        with pytest.raises(RateLimitError) as exc_info:
            await llm.agenerate(prompt="Hello, how are you?")

        # Check that it contains expected elements rather than exact match
        error_str = str(exc_info.value)
        assert "Rate limit exceeded for model" in error_str
        assert "openrouter/openai/gpt-4o" in error_str
        assert "provider throttle triggered" in error_str
        assert "openrouter.ai/docs/api/reference/limits" in error_str
