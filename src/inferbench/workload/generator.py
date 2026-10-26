"""Deterministic workload generation.

The generator turns a :class:`~inferbench.config.WorkloadConfig` into a list of
:class:`~inferbench.types.RequestSpec`. Determinism is a hard requirement: the
same config and seed must produce byte-identical prompts, or comparisons across
runs are meaningless.

Token counting
--------------
Real token counts require the backend's tokenizer, which InferBench does not
assume. Two modes exist:

- ``tokenizer`` — a callable was supplied (e.g. a HuggingFace tokenizer); prompt
  text is built and measured with it, so ``target_prompt_tokens`` is accurate.
- ``heuristic`` — the default; a deterministic word-like vocabulary is used and
  token counts are approximated by whitespace splitting. The *mode* is recorded
  in run metadata and shown in reports, so a heuristic run is never presented as
  if measured with a real tokenizer.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from inferbench.config import WorkloadConfig, WorkloadKind
from inferbench.types import RequestSpec

__all__ = [
    "WorkloadPlan",
    "generate_workload",
    "TokenCounter",
    "HISTORICAL_SPECIAL_REQUEST_ID",
]

#: Reserved id for the request issued by the runner to verify the backend can
#: actually serve, before the measured workload starts.
HISTORICAL_SPECIAL_REQUEST_ID = "__inferbench_probe__"

#: A fixed vocabulary keeps prompt generation deterministic across runs and
#: processes (Python's hash randomisation does not affect this list).
_VOCAB: tuple[str, ...] = tuple(f"w{i}" for i in range(512))

TokenCounter = Callable[[str], int]

#: Class labels for the mixed workload, ordered short→long.
_MIX_LABELS = ("short", "medium", "long")


@dataclass(frozen=True, slots=True)
class WorkloadPlan:
    """A fully materialized workload.

    ``token_count_mode`` records whether token counts came from a real
    tokenizer or the heuristic, which determines how strongly results may be
    compared.
    """

    specs: tuple[RequestSpec, ...]
    token_count_mode: str
    #: Distribution summary, recorded in metadata and used for fairness gating.
    prompt_token_summary: dict[str, float]
    output_token_summary: dict[str, float]


def _heuristic_count(text: str) -> int:
    """Approximate token count by whitespace-delimited pieces."""
    return len(text.split())


def _build_prompt(rng: random.Random, tokens: int) -> str:
    """Build a deterministic pseudo-prompt of approximately ``tokens`` pieces."""
    return " ".join(rng.choice(_VOCAB) for _ in range(max(1, tokens)))


def _build_prompt_with_tokenizer(
    rng: random.Random, tokens: int, decode: Callable[[Sequence[int]], str]
) -> str:
    """Build a prompt whose real token count is exactly ``tokens`` when possible."""
    # Sample token ids from a stable range, then decode. This is the standard
    # trick for producing exact-length prompts for a specific tokenizer.
    ids = [rng.randrange(0, 30_000) for _ in range(max(1, tokens))]
    try:
        text = decode(ids)
    except Exception:  # pragma: no cover - tokenizer-specific
        return _build_prompt(rng, tokens)
    return text


def _summary(values: Sequence[int]) -> dict[str, float]:
    """Small, JSON-safe summary of a length distribution."""
    if not values:
        return {"count": 0, "min": 0, "max": 0, "mean": 0.0}
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "min": float(ordered[0]),
        "max": float(ordered[-1]),
        "mean": round(sum(ordered) / len(ordered), 3),
    }


def generate_workload(
    config: WorkloadConfig,
    *,
    encode: TokenCounter | None = None,
    decode: Callable[[Sequence[int]], str] | None = None,
) -> WorkloadPlan:
    """Materialize a workload from ``config``.

    Args:
        config: Workload definition (lengths, counts, seed, mix weights).
        encode: Optional tokenizer encode function, used to verify lengths.
        decode: Optional tokenizer decode function, used to build exact prompts.

    Returns:
        A :class:`WorkloadPlan` with one spec per measured request. Warm-up and
        repetition assignment happen in the runner, which owns load semantics.
    """
    rng = random.Random(config.seed)
    token_count_mode = "tokenizer" if decode is not None else "heuristic"

    lengths: list[int] = []
    labels: list[str] = []

    if config.kind is WorkloadKind.fixed:
        lengths = [config.prompt_tokens] * config.num_requests
        labels = ["fixed"] * config.num_requests
    elif config.kind is WorkloadKind.randomized:
        assert config.prompt_tokens_min is not None and config.prompt_tokens_max is not None
        lengths = [
            rng.randint(config.prompt_tokens_min, config.prompt_tokens_max)
            for _ in range(config.num_requests)
        ]
        labels = ["random"] * config.num_requests
    else:  # mixed
        assert config.prompt_tokens_min is not None and config.prompt_tokens_max is not None
        lo, hi = config.prompt_tokens_min, config.prompt_tokens_max
        span = max(1, (hi - lo) // 3)
        bucket_bounds = (
            (lo, lo + span),
            (lo + span + 1, lo + 2 * span),
            (lo + 2 * span + 1, max(hi, lo + 2 * span + 1)),
        )
        for _ in range(config.num_requests):
            bucket = rng.choices(
                population=(0, 1, 2),
                weights=config.mix_weights,
                k=1,
            )[0]
            bucket_lo, bucket_hi = bucket_bounds[bucket]
            lengths.append(rng.randint(bucket_lo, bucket_hi))
            labels.append(_MIX_LABELS[bucket])

    output_lengths = _resolve_output_lengths(config, rng)

    specs: list[RequestSpec] = []
    for index in range(config.num_requests):
        target_prompt = lengths[index]
        if decode is not None:
            prompt = _build_prompt_with_tokenizer(rng, target_prompt, decode)
        else:
            prompt = _build_prompt(rng, target_prompt)

        specs.append(
            RequestSpec(
                request_id=f"req-{index:05d}",
                prompt=prompt,
                target_prompt_tokens=target_prompt,
                max_output_tokens=output_lengths[index],
                class_label=labels[index],
            )
        )

    return WorkloadPlan(
        specs=tuple(specs),
        token_count_mode=token_count_mode,
        prompt_token_summary=_summary(lengths),
        output_token_summary=_summary(output_lengths),
    )


def _resolve_output_lengths(config: WorkloadConfig, rng: random.Random) -> list[int]:
    """Per-request ``max_tokens`` values, mirroring the prompt length logic."""
    if config.kind is WorkloadKind.fixed:
        return [config.output_tokens] * config.num_requests

    lo = config.output_tokens_min
    hi = config.output_tokens_max
    if lo is None or hi is None:
        # Only the prompt-length distribution was randomized; keep output fixed.
        return [config.output_tokens] * config.num_requests

    if config.kind is WorkloadKind.randomized:
        return [rng.randint(lo, hi) for _ in range(config.num_requests)]

    # mixed: line output lengths up with the same short/medium/long buckets,
    # so a "long" request is long in both prompt and output.
    span = max(1, (hi - lo) // 3)
    buckets = (
        (lo, lo + span),
        (lo + span + 1, lo + 2 * span),
        (lo + 2 * span + 1, max(hi, lo + 2 * span + 1)),
    )
    return [
        rng.randint(*buckets[rng.choices((0, 1, 2), weights=config.mix_weights, k=1)[0]])
        for _ in range(config.num_requests)
    ]
