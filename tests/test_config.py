"""Tests for typed configuration loading and validation."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from pydantic import ValidationError

from inferbench.config import (
    ArrivalPattern,
    BackendKind,
    LoadConfig,
    RunConfig,
    WorkloadConfig,
    WorkloadKind,
    load_config,
)
from inferbench.errors import ConfigurationError


class TestDefaults:
    def test_defaults_are_cpu_safe(self) -> None:
        """A default RunConfig must not require a GPU or a real model path."""
        config = RunConfig()
        assert config.backend.kind is BackendKind.mock
        assert "localhost" in config.backend.base_url
        # No filesystem path is used as a model identifier.
        assert "/" not in config.backend.model

    def test_load_config_accepts_real_baseline(self) -> None:
        """The shipped baseline config must parse cleanly."""
        path = Path(__file__).resolve().parents[1] / "configs" / "baseline.yaml"
        config = load_config(path)
        assert config.run_name == "baseline-mock"
        assert config.backend.kind is BackendKind.mock
        assert config.workload.num_requests == 24


class TestBackendValidation:
    def test_trailing_slash_stripped(self) -> None:
        config = RunConfig.model_validate(
            {"backend": {"kind": "mock", "base_url": "http://localhost:8000/"}}
        )
        assert config.backend.base_url == "http://localhost:8000"

    def test_blank_api_key_becomes_none(self) -> None:
        config = RunConfig.model_validate({"backend": {"api_key": "   "}})
        assert config.backend.api_key is None

    def test_unknown_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RunConfig.model_validate({"backend": {"kind": "mock", "nonsense": 1}})

    def test_nonpositive_timeout_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RunConfig.model_validate({"backend": {"request_timeout_s": 0}})


class TestWorkloadValidation:
    def test_min_max_inverted_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be <="):
            WorkloadConfig(
                kind=WorkloadKind.randomized,
                prompt_tokens_min=256,
                prompt_tokens_max=128,
            )

    def test_randomized_requires_bounds(self) -> None:
        with pytest.raises(ValueError, match="requires"):
            WorkloadConfig(kind=WorkloadKind.randomized)

    def test_mixed_requires_bounds(self) -> None:
        with pytest.raises(ValueError, match="requires"):
            WorkloadConfig(kind=WorkloadKind.mixed)

    def test_mixed_rejects_zero_weight_sum(self) -> None:
        with pytest.raises(ValueError, match="sum to a positive"):
            WorkloadConfig(
                kind=WorkloadKind.mixed,
                prompt_tokens_min=8,
                prompt_tokens_max=64,
                mix_weights=(0.0, 0.0, 0.0),
            )

    def test_mixed_accepts_valid_weights(self) -> None:
        workload = WorkloadConfig(
            kind=WorkloadKind.mixed,
            prompt_tokens_min=8,
            prompt_tokens_max=64,
            mix_weights=(1.0, 1.0, 2.0),
        )
        assert workload.kind is WorkloadKind.mixed

    def test_zero_requests_rejected(self) -> None:
        with pytest.raises(ValidationError):
            WorkloadConfig(num_requests=0)


class TestLoadValidation:
    def test_rate_pattern_requires_rate(self) -> None:
        with pytest.raises(ValueError, match="requires rate_rps"):
            LoadConfig(pattern=ArrivalPattern.constant_rate)

    def test_bursty_requires_rate(self) -> None:
        with pytest.raises(ValueError, match="requires rate_rps"):
            LoadConfig(pattern=ArrivalPattern.bursty)

    def test_closed_loop_needs_no_rate(self) -> None:
        load = LoadConfig(pattern=ArrivalPattern.closed_loop, concurrency=8)
        assert load.rate_rps is None

    def test_constant_rate_accepts_rate(self) -> None:
        load = LoadConfig(pattern=ArrivalPattern.constant_rate, rate_rps=2.5)
        assert load.rate_rps == 2.5


class TestLoadConfigErrors:
    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigurationError, match="not found"):
            load_config(tmp_path / "nope.yaml")

    def test_directory_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigurationError, match="not a file"):
            load_config(tmp_path)

    def test_malformed_yaml(self, tmp_config: Callable[..., Path]) -> None:
        path = tmp_config("key: [unclosed\n")
        with pytest.raises(ConfigurationError, match="Could not parse"):
            load_config(path)

    def test_non_mapping_toplevel(self, tmp_config: Callable[..., Path]) -> None:
        path = tmp_config("- just\n- a list\n")
        with pytest.raises(ConfigurationError, match="must be a mapping"):
            load_config(path)

    def test_empty_file_uses_defaults(self, tmp_config: Callable[..., Path]) -> None:
        path = tmp_config("")
        config = load_config(path)
        assert config.backend.kind is BackendKind.mock

    def test_invalid_value_wrapped(self, tmp_config: Callable[..., Path]) -> None:
        path = tmp_config("workload:\n  num_requests: -5\n")
        with pytest.raises(ConfigurationError, match="Invalid config"):
            load_config(path)

    def test_json_accepted(self, tmp_config: Callable[..., Path]) -> None:
        path = tmp_config('{"run_name": "json-run"}', name="config.json")
        config = load_config(path)
        assert config.run_name == "json-run"
