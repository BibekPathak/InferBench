"""vLLM backend adapter, targeting the OpenAI-compatible HTTP API.

vLLM's server exposes ``/v1/models`` and ``/v1/chat/completions`` with
Server-Sent Events streaming, matching the OpenAI schema. This adapter speaks
that protocol using ``httpx`` directly (rather than the ``openai`` SDK) so we
control the socket-level timing used for TTFT and inter-token latency, and so
the adapter has no dependency beyond the core install.

Status: **written to the supported API but not yet validated against a live
vLLM server**, because the development host has no GPU. It is deliberately
isolated behind :class:`~inferbench.backends.base.BackendAdapter` so this is a
contained risk. The failure/streaming paths are unit-tested against a mock
transport.

Timing note: token timestamps are taken as each SSE ``data:`` line is parsed on
the client, which measures *client-observed* inter-token latency. This is the
metric that actually affects a user, but it includes network and client
scheduling effects. It is not the server's internal decode time, and reports
must not conflate them.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx

from inferbench.backends.base import (
    BackendAdapter,
    BackendCapabilities,
    CompletionSignal,
    FailureSignal,
    StreamEvent,
    TokenChunk,
)
from inferbench.config import BackendConfig
from inferbench.errors import (
    BackendError,
    BackendNotReadyError,
    ModelNotFoundError,
)
from inferbench.logging_utils import get_logger, log_event
from inferbench.types import RequestSpec
from inferbench.version_info import http_user_agent

__all__ = ["VLLMBackend"]

_log = get_logger("backends.vllm")


class VLLMBackend(BackendAdapter):
    """Streaming adapter for a vLLM OpenAI-compatible server."""

    def __init__(self, config: BackendConfig) -> None:
        super().__init__(config)
        self._client: httpx.AsyncClient | None = None
        self._server_version: str | None = None
        self._served_models: tuple[str, ...] = ()

    @property
    def name(self) -> str:
        return "vllm"

    @property
    def version(self) -> str | None:
        return self._server_version

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            streaming=True,
            # vLLM reports usage in the final chunk when requested.
            reports_prompt_tokens=True,
            reports_request_id=True,
            reports_logprobs=True,
            is_real=True,
        )

    async def setup(self) -> None:
        """Wait for readiness, then confirm the requested model is served."""
        headers = self._auth_headers()
        timeout = httpx.Timeout(
            connect=10.0,
            read=self.config.readiness_timeout_s,
            write=10.0,
            pool=10.0,
        )
        self._client = httpx.AsyncClient(
            base_url=self.config.base_url,
            headers=headers,
            timeout=timeout,
        )

        await self._await_ready()
        await self._resolve_model()
        log_event(
            _log,
            logging.INFO,
            "vLLM backend ready",
            base_url=self.config.base_url,
            model=self.config.model,
        )

    async def teardown(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _auth_headers(self) -> dict[str, str]:
        headers = {"User-Agent": http_user_agent()}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers

    async def _await_ready(self) -> None:
        assert self._client is not None
        deadline = time.monotonic() + self.config.readiness_timeout_s
        last_error: str = "unknown"

        while True:
            try:
                response = await self._client.get("/health")
                if response.status_code < 400:
                    return
                last_error = f"HTTP {response.status_code}"
            except httpx.HTTPError as exc:
                last_error = f"{type(exc).__name__}: {exc}"

            if time.monotonic() >= deadline:
                raise BackendNotReadyError(
                    f"Backend at {self.config.base_url} not ready after "
                    f"{self.config.readiness_timeout_s}s (last error: {last_error})"
                )
            await asyncio.sleep(self.config.readiness_poll_s)

    async def _resolve_model(self) -> None:
        assert self._client is not None
        try:
            response = await self._client.get("/v1/models")
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise BackendError(f"Could not list models from {self.config.base_url}: {exc}") from exc

        payload = response.json()
        models = payload.get("data", []) if isinstance(payload, dict) else []
        ids = tuple(
            str(entry.get("id")) for entry in models if isinstance(entry, dict) and entry.get("id")
        )
        self._served_models = ids

        if ids and self.config.model not in ids:
            raise ModelNotFoundError(
                f"Model {self.config.model!r} is not served. Available models: {', '.join(ids)}"
            )

    async def describe(self) -> dict[str, Any]:
        details: dict[str, Any] = {
            "adapter": "vllm",
            "base_url": self.config.base_url,
            "served_models": list(self._served_models),
        }
        # Version endpoint is optional across vLLM releases; never invent one.
        if self._client is not None:
            try:
                response = await self._client.get("/version")
                if response.status_code < 400:
                    data = response.json()
                    if isinstance(data, dict) and "version" in data:
                        version = str(data["version"])
                        self._server_version = version
                        details["server_version"] = version
            except (httpx.HTTPError, ValueError):
                details["server_version"] = None
        return details

    def _payload(self, spec: RequestSpec) -> dict[str, Any]:
        return {
            "model": self.config.model,
            "messages": [{"role": "user", "content": spec.prompt}],
            "max_tokens": spec.max_output_tokens,
            "stream": True,
            # Ask vLLM to include token usage so prompt token counts are real
            # rather than estimated.
            "stream_options": {"include_usage": True},
            "temperature": 0.0,
        }

    async def stream(self, spec: RequestSpec) -> AsyncIterator[StreamEvent]:
        if self._client is None:
            raise BackendError("Backend not set up; call setup() first")

        start = time.perf_counter()
        output_tokens = 0
        prompt_tokens: int | None = None
        backend_request_id: str | None = None
        finish_reason: str | None = None

        timeout = httpx.Timeout(
            connect=10.0,
            read=self.config.request_timeout_s,
            write=30.0,
            pool=self.config.request_timeout_s,
        )

        try:
            async with self._client.stream(
                "POST",
                "/v1/chat/completions",
                json=self._payload(spec),
                timeout=timeout,
            ) as response:
                if response.status_code >= 400:
                    body = await _safe_read(response)
                    yield FailureSignal(
                        t=time.perf_counter() - start,
                        status="error",
                        message=f"HTTP {response.status_code}: {body[:300]}",
                        output_tokens=0,
                    )
                    return

                async for line in response.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[len("data:") :].strip()
                    if data == "[DONE]":
                        break

                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        # A malformed chunk is a protocol problem, not a
                        # measurement; record it rather than crashing.
                        yield FailureSignal(
                            t=time.perf_counter() - start,
                            status="error",
                            message=f"Malformed stream chunk: {data[:200]}",
                            output_tokens=output_tokens,
                            backend_request_id=backend_request_id,
                        )
                        return

                    chunk_id = chunk.get("id")
                    if isinstance(chunk_id, str):
                        backend_request_id = chunk_id

                    usage = chunk.get("usage")
                    if isinstance(usage, dict):
                        pt = usage.get("prompt_tokens")
                        if isinstance(pt, int):
                            prompt_tokens = pt

                    for choice in chunk.get("choices") or []:
                        if not isinstance(choice, dict):
                            continue
                        reason = choice.get("finish_reason")
                        if isinstance(reason, str):
                            finish_reason = reason
                        delta = choice.get("delta") or {}
                        content = delta.get("content")
                        if isinstance(content, str) and content:
                            output_tokens += 1
                            yield TokenChunk(
                                t=time.perf_counter() - start,
                                text=content,
                            )

            yield CompletionSignal(
                t=time.perf_counter() - start,
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
                finish_reason=finish_reason,
                backend_request_id=backend_request_id,
            )

        except httpx.TimeoutException as exc:
            yield FailureSignal(
                t=time.perf_counter() - start,
                status="timeout",
                message=f"Request exceeded {self.config.request_timeout_s}s: {exc}",
                output_tokens=output_tokens,
                backend_request_id=backend_request_id,
            )
        except httpx.HTTPError as exc:
            # A transport failure after tokens arrived is an interrupted
            # generation, not a hard error; the partial count is preserved.
            status = "interrupted" if output_tokens > 0 else "error"
            yield FailureSignal(
                t=time.perf_counter() - start,
                status=status,
                message=f"{type(exc).__name__}: {exc}",
                output_tokens=output_tokens,
                backend_request_id=backend_request_id,
            )


async def _safe_read(response: httpx.Response) -> str:
    try:
        return (await response.aread()).decode("utf-8", errors="replace")
    except httpx.HTTPError:
        return "<unreadable response body>"
