"""Typed configuration for InferBench.

Configuration is expressed as pydantic models so that a malformed config file
fails loudly at load time with a precise error, rather than producing a subtly
wrong benchmark. The same models are used for both YAML experiment configs and
(for future phases) API-supplied configuration.

Design rules:
- No default references a concrete GPU, model path, API key, or cloud provider.
- Every field that affects reproducibility is explicit and recorded in run
  metadata.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from inferbench.errors import ConfigurationError

__all__ = [
    "BackendKind",
    "WorkloadKind",
    "ArrivalPattern",
    "BackendConfig",
    "WorkloadConfig",
    "LoadConfig",
    "RunConfig",
    "load_config",
]


class BackendKind(StrEnum):
    """Supported inference backend families."""

    mock = "mock"
    vllm = "vllm"


class WorkloadKind(StrEnum):
    """Prompt/output length distribution shapes."""

    fixed = "fixed"
    randomized = "randomized"
    mixed = "mixed"


class ArrivalPattern(StrEnum):
    """How requests arrive at the backend."""

    #: A fixed pool of concurrent workers, each issuing the next request on
    #: completion. Models sustained concurrency, not a fixed request rate.
    closed_loop = "closed_loop"
    #: Requests are released on a schedule approximating a target rate.
    constant_rate = "constant_rate"
    #: Requests are released in bursts of ``burst_size`` at a target rate.
    bursty = "bursty"


class BackendConfig(BaseModel):
    """Connection and readiness settings for an inference backend."""

    model_config = ConfigDict(extra="forbid")

    kind: BackendKind = Field(
        default=BackendKind.mock,
        description="Which backend adapter to use.",
    )
    base_url: str = Field(
        default="http://localhost:8000",
        description="Base URL of the backend's OpenAI-compatible HTTP server.",
    )
    model: str = Field(
        default="mock-model",
        description="Model identifier as the backend knows it. Never a local path.",
    )
    api_key: str | None = Field(
        default=None,
        description=(
            "Optional API key. Prefer the INFERBENCH_API_KEY environment "
            "variable over embedding secrets in config files."
        ),
    )
    request_timeout_s: float = Field(
        default=60.0,
        gt=0,
        description="Per-request deadline. Exceeded requests are recorded as timeouts.",
    )
    readiness_timeout_s: float = Field(
        default=120.0,
        gt=0,
        description="How long to wait for the backend to become ready before a run.",
    )
    readiness_poll_s: float = Field(
        default=2.0,
        gt=0,
        description="Interval between readiness checks.",
    )

    @field_validator("base_url")
    @classmethod
    def _strip_trailing_slash(cls, value: str) -> str:
        return value.rstrip("/")

    @field_validator("api_key")
    @classmethod
    def _reject_empty_key(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            return None
        return value


class WorkloadConfig(BaseModel):
    """Definition of the requests to generate.

    Token lengths are targets, not guarantees: the actual prompt token count
    depends on the tokenizer in use. The workload generator records which
    tokenizer (or fallback heuristic) produced each request so that runs using
    different distributions are never compared silently.
    """

    model_config = ConfigDict(extra="forbid")

    kind: WorkloadKind = Field(default=WorkloadKind.fixed)
    num_requests: int = Field(
        default=32,
        gt=0,
        description="Requests issued per repetition, excluding warm-up.",
    )
    prompt_tokens: int = Field(
        default=128,
        ge=1,
        description="Target prompt length for fixed workloads.",
    )
    output_tokens: int = Field(
        default=64,
        ge=1,
        description="Requested maximum output length (``max_tokens``).",
    )
    prompt_tokens_min: int | None = Field(default=None, ge=1)
    prompt_tokens_max: int | None = Field(default=None, ge=1)
    output_tokens_min: int | None = Field(default=None, ge=1)
    output_tokens_max: int | None = Field(default=None, ge=1)
    #: For ``mixed`` workloads: weights for (short, medium, long) buckets.
    mix_weights: tuple[float, float, float] = Field(
        default=(0.4, 0.3, 0.3),
        description="Relative weights of short/medium/long request classes.",
    )
    seed: int = Field(
        default=0,
        ge=0,
        description="Seed for deterministic prompt content and length sampling.",
    )

    @model_validator(mode="after")
    def _check_ranges(self) -> WorkloadConfig:
        pairs = (
            (
                "prompt_tokens_min",
                self.prompt_tokens_min,
                "prompt_tokens_max",
                self.prompt_tokens_max,
            ),
            (
                "output_tokens_min",
                self.output_tokens_min,
                "output_tokens_max",
                self.output_tokens_max,
            ),
        )
        for lo_name, lo, hi_name, hi in pairs:
            if lo is not None and hi is not None and lo > hi:
                raise ValueError(f"{lo_name} ({lo}) must be <= {hi_name} ({hi})")

        if self.kind in (WorkloadKind.randomized, WorkloadKind.mixed) and (
            self.prompt_tokens_min is None or self.prompt_tokens_max is None
        ):
            raise ValueError(
                f"workload kind {self.kind.value!r} requires "
                "prompt_tokens_min and prompt_tokens_max"
            )
        if self.kind is WorkloadKind.mixed:
            if any(weight < 0 for weight in self.mix_weights):
                raise ValueError("mix_weights must be non-negative")
            if sum(self.mix_weights) <= 0:
                raise ValueError("mix_weights must sum to a positive value")
        return self


class LoadConfig(BaseModel):
    """Load profile: how many requests are in flight, and how they arrive."""

    model_config = ConfigDict(extra="forbid")

    pattern: ArrivalPattern = Field(default=ArrivalPattern.closed_loop)
    concurrency: int = Field(
        default=1,
        ge=1,
        description="Number of in-flight requests for closed-loop load.",
    )
    rate_rps: float | None = Field(
        default=None,
        gt=0,
        description="Offered request rate for constant/bursty patterns.",
    )
    burst_size: int = Field(
        default=8,
        ge=1,
        description="Requests per burst for the bursty pattern.",
    )
    warmup_requests: int = Field(
        default=4,
        ge=0,
        description=(
            "Requests discarded before measurement. Warms caches, CUDA graphs, "
            "and connection pools so measured statistics are not startup noise."
        ),
    )
    repetitions: int = Field(
        default=1,
        ge=1,
        description="Independent repetitions of the measured workload.",
    )

    @model_validator(mode="after")
    def _check_pattern_requirements(self) -> LoadConfig:
        needs_rate = (ArrivalPattern.constant_rate, ArrivalPattern.bursty)
        if self.pattern in needs_rate and self.rate_rps is None:
            raise ValueError(f"load pattern {self.pattern.value!r} requires rate_rps")
        return self


class RunConfig(BaseModel):
    """The complete definition of one experiment run."""

    model_config = ConfigDict(extra="forbid")

    run_name: str = Field(
        default="unnamed",
        min_length=1,
        description="Human-readable label recorded in metadata and report headers.",
    )
    experiment: str = Field(
        default="baseline",
        min_length=1,
        description="Experiment identifier, e.g. 'A' or 'baseline'.",
    )
    backend: BackendConfig = Field(default_factory=BackendConfig)
    workload: WorkloadConfig = Field(default_factory=WorkloadConfig)
    load: LoadConfig = Field(default_factory=LoadConfig)
    output_dir: Path = Field(
        default=Path("runs"),
        description="Directory for raw JSONL, the SQLite store, and exports.",
    )
    notes: str = Field(default="", description="Free-form operator notes for the record.")


def load_config(path: str | Path) -> RunConfig:
    """Load and validate a run configuration from a YAML or JSON file.

    Args:
        path: Path to a ``.yaml``/``.yml``/``.json`` config file.

    Raises:
        ConfigurationError: The file is missing, unparseable, or fails validation.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise ConfigurationError(f"Config file not found: {config_path}")
    if not config_path.is_file():
        raise ConfigurationError(f"Config path is not a file: {config_path}")

    try:
        raw_text = config_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigurationError(f"Could not read config {config_path}: {exc}") from exc

    try:
        # safe_load handles JSON too, since JSON is a subset of YAML.
        parsed: Any = yaml.safe_load(raw_text)
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"Could not parse config {config_path}: {exc}") from exc

    if parsed is None:
        parsed = {}
    if not isinstance(parsed, dict):
        raise ConfigurationError(
            f"Config {config_path} must be a mapping at the top level, got {type(parsed).__name__}"
        )

    try:
        return RunConfig.model_validate(parsed)
    except Exception as exc:  # pydantic ValidationError
        raise ConfigurationError(f"Invalid config {config_path}: {exc}") from exc
