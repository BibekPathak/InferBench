"""Tests for structured logging."""

from __future__ import annotations

import io
import json
import logging

import pytest

from inferbench.logging_utils import bind_run_context, configure_logging, get_logger


def test_json_formatter_emits_valid_json() -> None:
    stream = io.StringIO()
    configure_logging(level="INFO", fmt="json", stream=stream)
    get_logger("test").info("hello", extra={"run_id": "abc"})

    line = stream.getvalue().strip()
    payload = json.loads(line)
    assert payload["message"] == "hello"
    assert payload["level"] == "INFO"
    assert payload["run_id"] == "abc"
    assert payload["logger"] == "inferbench.test"


def test_text_formatter_is_human_readable() -> None:
    stream = io.StringIO()
    configure_logging(level="INFO", fmt="text", stream=stream)
    get_logger("test").info("hello")

    output = stream.getvalue()
    assert "hello" in output
    assert "INFO" in output


def test_reconfiguration_does_not_duplicate_handlers() -> None:
    stream = io.StringIO()
    configure_logging(level="INFO", fmt="text", stream=stream)
    configure_logging(level="INFO", fmt="text", stream=stream)
    get_logger("test").info("once")

    assert stream.getvalue().count("once") == 1


def test_level_filtering() -> None:
    stream = io.StringIO()
    configure_logging(level="WARNING", fmt="text", stream=stream)
    logger = get_logger("test")
    logger.info("suppressed")
    logger.warning("shown")

    output = stream.getvalue()
    assert "suppressed" not in output
    assert "shown" in output


def test_bind_run_context_adds_run_id() -> None:
    stream = io.StringIO()
    configure_logging(level="INFO", fmt="json", stream=stream)
    adapter = bind_run_context("run-42", phase="warmup")
    adapter.info("working")

    payload = json.loads(stream.getvalue().strip())
    assert payload["run_id"] == "run-42"
    assert payload["phase"] == "warmup"


def test_unknown_level_name_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown log level"):
        configure_logging(level="NOTALEVEL")


def test_logger_does_not_propagate() -> None:
    configure_logging(level="INFO", fmt="text", stream=io.StringIO())
    assert get_logger().propagate is False
    assert logging.getLogger("inferbench").propagate is False
