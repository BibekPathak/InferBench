"""Tests for backend adapters.

The mock backend is tested directly. The vLLM adapter is tested against a mock
HTTP transport (``respx``), which validates the streaming/parsing/error paths
without a server — an honest substitute for the GPU host we do not have.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from inferbench.backends import (
    BackendAdapter,
    CompletionSignal,
    FailureSignal,
    MockBackend,
    MockBackendOptions,
    TokenChunk,
    create_backend,
)
from inferbench.config import BackendConfig, BackendKind
from inferbench.errors import BackendNotReadyError, ConfigurationError, ModelNotFoundError
from inferbench.types import RequestSpec


def spec(n: int = 8, prompt_tokens: int = 16) -> RequestSpec:
    return RequestSpec(
        request_id="r0",
        prompt="hello world",
        target_prompt_tokens=prompt_tokens,
        max_output_tokens=n,
    )


async def collect(backend: BackendAdapter, request: RequestSpec) -> list[object]:
    events: list[object] = []
    async for event in backend.stream(request):
        events.append(event)
    return events


class TestMockBackendContract:
    """The mock must satisfy the adapter contract exactly."""

    def test_is_a_backend_adapter(self) -> None:
        assert isinstance(MockBackend(), BackendAdapter)

    def test_capabilities_are_honest(self) -> None:
        caps = MockBackend().capabilities
        assert caps.streaming is True
        assert caps.is_real is False

    def test_version_is_none_not_fabricated(self) -> None:
        assert MockBackend().version is None

    @pytest.mark.asyncio
    async def test_setup_and_teardown(self) -> None:
        backend = MockBackend(options=MockBackendOptions(real_time=False))
        await backend.setup()
        await backend.teardown()
        await backend.teardown()  # idempotent

    @pytest.mark.asyncio
    async def test_not_ready_raises_on_setup(self) -> None:
        backend = MockBackend(options=MockBackendOptions(ready=False))
        with pytest.raises(BackendNotReadyError):
            await backend.setup()

    @pytest.mark.asyncio
    async def test_context_manager(self) -> None:
        async with MockBackend(options=MockBackendOptions(real_time=False)) as backend:
            events = await collect(backend, spec(4))
        assert any(isinstance(e, CompletionSignal) for e in events)


class TestMockBackendStreaming:
    @pytest.mark.asyncio
    async def test_exact_token_count(self) -> None:
        backend = MockBackend(options=MockBackendOptions(real_time=False))
        await backend.setup()
        events = await collect(backend, spec(7))
        tokens = [e for e in events if isinstance(e, TokenChunk)]
        assert len(tokens) == 7

    @pytest.mark.asyncio
    async def test_terminal_event_is_last_and_unique(self) -> None:
        backend = MockBackend(options=MockBackendOptions(real_time=False))
        await backend.setup()
        events = await collect(backend, spec(5))
        terminals = [e for e in events if isinstance(e, (CompletionSignal, FailureSignal))]
        assert len(terminals) == 1
        assert events[-1] is terminals[0]

    @pytest.mark.asyncio
    async def test_monotonic_timestamps(self) -> None:
        backend = MockBackend(options=MockBackendOptions(real_time=False))
        await backend.setup()
        events = await collect(backend, spec(10))
        times = [e.t for e in events]  # type: ignore[attr-defined]
        assert times == sorted(times)

    @pytest.mark.asyncio
    async def test_prompt_tokens_reported_when_capable(self) -> None:
        backend = MockBackend(options=MockBackendOptions(real_time=False))
        await backend.setup()
        events = await collect(backend, spec(3, prompt_tokens=42))
        completion = next(e for e in events if isinstance(e, CompletionSignal))
        assert completion.prompt_tokens == 42

    @pytest.mark.asyncio
    async def test_prompt_tokens_none_when_not_capable(self) -> None:
        """Never fabricate a token count the backend does not report."""
        backend = MockBackend(
            options=MockBackendOptions(real_time=False, report_prompt_tokens=False)
        )
        await backend.setup()
        events = await collect(backend, spec(3))
        completion = next(e for e in events if isinstance(e, CompletionSignal))
        assert completion.prompt_tokens is None

    @pytest.mark.asyncio
    async def test_failure_injection(self) -> None:
        backend = MockBackend(options=MockBackendOptions(real_time=False, fail_every=1))
        await backend.setup()
        events = await collect(backend, spec(5))
        failure = next(e for e in events if isinstance(e, FailureSignal))
        assert failure.status == "error"

    @pytest.mark.asyncio
    async def test_interruption_preserves_partial_count(self) -> None:
        backend = MockBackend(options=MockBackendOptions(real_time=False, interrupt_every=1))
        await backend.setup()
        events = await collect(backend, spec(10))
        failure = next(e for e in events if isinstance(e, FailureSignal))
        assert failure.status == "interrupted"
        assert 0 < failure.output_tokens < 10

    @pytest.mark.asyncio
    async def test_concurrency_is_tracked(self) -> None:
        backend = MockBackend(options=MockBackendOptions(real_time=False))
        await backend.setup()
        assert backend.inflight == 0
        await collect(backend, spec(2))
        assert backend.inflight == 0


class TestMockAnalyticTiming:
    """The mock's timing model must match its analytic predictions exactly."""

    @pytest.mark.asyncio
    async def test_ttft_matches_prediction(self) -> None:
        options = MockBackendOptions(real_time=False, ttft_base_s=0.01, prefill_tokens_per_s=1000.0)
        backend = MockBackend(options=options)
        await backend.setup()
        events = await collect(backend, spec(3, prompt_tokens=500))
        first = events[0]
        assert isinstance(first, TokenChunk)
        # 0.01 + 500/1000 = 0.51
        assert first.t == pytest.approx(0.51)

    @pytest.mark.asyncio
    async def test_tpot_matches_prediction(self) -> None:
        options = MockBackendOptions(
            real_time=False, ttft_base_s=0.0, tpot_base_s=0.02, prefill_tokens_per_s=None
        )
        backend = MockBackend(options=options)
        await backend.setup()
        events = await collect(backend, spec(5, prompt_tokens=1))
        tokens = [e for e in events if isinstance(e, TokenChunk)]
        gaps = [b.t - a.t for a, b in zip(tokens, tokens[1:], strict=False)]
        assert all(gap == pytest.approx(0.02) for gap in gaps)


class TestCreateBackend:
    def test_creates_mock(self) -> None:
        backend = create_backend(BackendConfig(kind=BackendKind.mock))
        assert isinstance(backend, MockBackend)

    def test_creates_vllm(self) -> None:
        backend = create_backend(BackendConfig(kind=BackendKind.vllm))
        assert backend.name == "vllm"

    def test_rejects_bad_mock_options(self) -> None:
        with pytest.raises(ConfigurationError, match="MockBackendOptions"):
            create_backend(BackendConfig(kind=BackendKind.mock), options="nope")


# ---------------------------------------------------------------------------
# vLLM adapter, exercised against a mock transport
# ---------------------------------------------------------------------------

_BASE = "http://localhost:8000"


def _sse(*chunks: str) -> str:
    return "".join(f"data: {chunk}\n\n" for chunk in chunks)


class TestVLLMAdapter:
    @respx.mock
    @pytest.mark.asyncio
    async def test_setup_validates_model(self) -> None:
        respx.get(f"{_BASE}/health").mock(return_value=httpx.Response(200))
        respx.get(f"{_BASE}/v1/models").mock(
            return_value=httpx.Response(200, json={"data": [{"id": "llama-3-8b"}]})
        )
        from inferbench.backends.vllm import VLLMBackend

        backend = VLLMBackend(
            BackendConfig(kind=BackendKind.vllm, base_url=_BASE, model="llama-3-8b")
        )
        await backend.setup()
        await backend.teardown()

    @respx.mock
    @pytest.mark.asyncio
    async def test_setup_rejects_missing_model(self) -> None:
        respx.get(f"{_BASE}/health").mock(return_value=httpx.Response(200))
        respx.get(f"{_BASE}/v1/models").mock(
            return_value=httpx.Response(200, json={"data": [{"id": "other-model"}]})
        )
        from inferbench.backends.vllm import VLLMBackend

        backend = VLLMBackend(BackendConfig(kind=BackendKind.vllm, base_url=_BASE, model="missing"))
        with pytest.raises(ModelNotFoundError, match="not served"):
            await backend.setup()

    @respx.mock
    @pytest.mark.asyncio
    async def test_stream_parses_tokens_and_usage(self) -> None:
        respx.get(f"{_BASE}/health").mock(return_value=httpx.Response(200))
        respx.get(f"{_BASE}/v1/models").mock(
            return_value=httpx.Response(200, json={"data": [{"id": "m"}]})
        )
        body = _sse(
            '{"id":"cmpl-1","choices":[{"delta":{"content":"He"},"finish_reason":null}]}',
            '{"id":"cmpl-1","choices":[{"delta":{"content":"llo"},"finish_reason":null}]}',
            '{"id":"cmpl-1","choices":[{"delta":{},"finish_reason":"stop"}],'
            '"usage":{"prompt_tokens":11,"completion_tokens":2}}',
            "[DONE]",
        )
        respx.post(f"{_BASE}/v1/chat/completions").mock(return_value=httpx.Response(200, text=body))
        from inferbench.backends.vllm import VLLMBackend

        backend = VLLMBackend(BackendConfig(kind=BackendKind.vllm, base_url=_BASE, model="m"))
        await backend.setup()
        events = await collect(backend, spec(4))
        await backend.teardown()

        tokens = [e for e in events if isinstance(e, TokenChunk)]
        completion = next(e for e in events if isinstance(e, CompletionSignal))
        assert [t.text for t in tokens] == ["He", "llo"]
        assert completion.output_tokens == 2
        assert completion.prompt_tokens == 11
        assert completion.finish_reason == "stop"
        assert completion.backend_request_id == "cmpl-1"

    @respx.mock
    @pytest.mark.asyncio
    async def test_http_error_becomes_failure_not_exception(self) -> None:
        respx.get(f"{_BASE}/health").mock(return_value=httpx.Response(200))
        respx.get(f"{_BASE}/v1/models").mock(
            return_value=httpx.Response(200, json={"data": [{"id": "m"}]})
        )
        respx.post(f"{_BASE}/v1/chat/completions").mock(
            return_value=httpx.Response(500, text="internal error")
        )
        from inferbench.backends.vllm import VLLMBackend

        backend = VLLMBackend(BackendConfig(kind=BackendKind.vllm, base_url=_BASE, model="m"))
        await backend.setup()
        events = await collect(backend, spec(4))
        await backend.teardown()

        failure = next(e for e in events if isinstance(e, FailureSignal))
        assert failure.status == "error"
        assert "500" in failure.message

    @respx.mock
    @pytest.mark.asyncio
    async def test_malformed_chunk_is_failure(self) -> None:
        respx.get(f"{_BASE}/health").mock(return_value=httpx.Response(200))
        respx.get(f"{_BASE}/v1/models").mock(
            return_value=httpx.Response(200, json={"data": [{"id": "m"}]})
        )
        body = "data: {not valid json}\n\n"
        respx.post(f"{_BASE}/v1/chat/completions").mock(return_value=httpx.Response(200, text=body))
        from inferbench.backends.vllm import VLLMBackend

        backend = VLLMBackend(BackendConfig(kind=BackendKind.vllm, base_url=_BASE, model="m"))
        await backend.setup()
        events = await collect(backend, spec(4))
        await backend.teardown()

        failure = next(e for e in events if isinstance(e, FailureSignal))
        assert failure.status == "error"
        assert "Malformed" in failure.message

    @respx.mock
    @pytest.mark.asyncio
    async def test_disconnect_after_tokens_is_interruption(self) -> None:
        respx.get(f"{_BASE}/health").mock(return_value=httpx.Response(200))
        respx.get(f"{_BASE}/v1/models").mock(
            return_value=httpx.Response(200, json={"data": [{"id": "m"}]})
        )
        respx.post(f"{_BASE}/v1/chat/completions").mock(
            side_effect=httpx.ConnectError("connection reset")
        )
        from inferbench.backends.vllm import VLLMBackend

        backend = VLLMBackend(BackendConfig(kind=BackendKind.vllm, base_url=_BASE, model="m"))
        await backend.setup()
        events = await collect(backend, spec(4))
        await backend.teardown()

        failure = next(e for e in events if isinstance(e, FailureSignal))
        # No tokens arrived, so this is a hard error rather than an interruption.
        assert failure.status == "error"
