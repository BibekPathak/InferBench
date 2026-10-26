"""Structured logging configuration for InferBench.

Logging is JSON-lines by default so that experiment runs are machine-parseable
and auditable, with a human-readable mode available for interactive use. We use
only the standard library to avoid a runtime dependency for something this
central.

Design notes:
- Configuration is idempotent: calling :func:`configure_logging` twice will not
  duplicate handlers.
- A ``run_id`` may be bound via :func:`bind_run_context` so every subsequent
  record from this process carries it, which makes multi-run logs greppable.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import IO, Any, Literal

__all__ = [
    "configure_logging",
    "bind_run_context",
    "get_logger",
    "log_event",
    "LogFormat",
]

LogFormat = Literal["json", "text"]

_LOGGER_NAMESPACE = "inferbench"

# Keys that the standard ``LogRecord`` already provides; anything else found on
# the record is treated as structured, user-supplied context.
_RESERVED_RECORD_KEYS = frozenset(logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys()) | {
    "message",
    "asctime",
    "taskName",
}


class _JsonFormatter(logging.Formatter):
    """Render each record as a single JSON object on one line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)

        for key, value in record.__dict__.items():
            if key not in _RESERVED_RECORD_KEYS:
                payload[key] = value

        return json.dumps(payload, default=str, ensure_ascii=False)


class _TextFormatter(logging.Formatter):
    """Compact human-readable formatter with inline structured context."""

    def format(self, record: logging.LogRecord) -> str:
        timestamp = self.formatTime(record, datefmt="%H:%M:%S")
        base = f"{timestamp} {record.levelname:<7} {record.name}: {record.getMessage()}"
        extras = {
            key: value for key, value in record.__dict__.items() if key not in _RESERVED_RECORD_KEYS
        }
        if extras:
            rendered = " ".join(f"{key}={value}" for key, value in extras.items())
            base = f"{base}  [{rendered}]"
        if record.exc_info:
            base = f"{base}\n{self.formatException(record.exc_info)}"
        return base


def configure_logging(
    level: int | str = logging.INFO,
    fmt: LogFormat = "json",
    stream: IO[str] | None = None,
) -> None:
    """Configure the ``inferbench`` logger tree.

    Args:
        level: Logging level as an int or name (e.g. ``"DEBUG"``).
        fmt: ``"json"`` for machine-readable lines, ``"text"`` for humans.
        stream: Destination stream; defaults to ``sys.stderr`` so stdout stays
            clean for real command output.
    """
    if isinstance(level, str):
        level = logging.getLevelName(level.upper())
        if not isinstance(level, int):  # getLevelName returns str on bad input
            raise ValueError(f"Unknown log level: {level!r}")

    logger = logging.getLogger(_LOGGER_NAMESPACE)
    logger.setLevel(level)
    # Do not propagate to the root logger, which may have arbitrary handlers.
    logger.propagate = False

    formatter: logging.Formatter = _JsonFormatter() if fmt == "json" else _TextFormatter()
    target = stream if stream is not None else sys.stderr

    # Replace only our own handlers, leaving foreign handlers untouched.
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    handler = logging.StreamHandler(target)
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a logger inside the ``inferbench`` namespace.

    Args:
        name: Optional dotted suffix, e.g. ``"runner"``.
    """
    if not name:
        return logging.getLogger(_LOGGER_NAMESPACE)
    return logging.getLogger(f"{_LOGGER_NAMESPACE}.{name}")


def bind_run_context(run_id: str, **context: Any) -> logging.LoggerAdapter[logging.Logger]:
    """Return a logger adapter that stamps ``run_id`` and extra context on records."""
    logger = get_logger()
    return logging.LoggerAdapter(logger, {"run_id": run_id, **context})


def log_event(
    logger: logging.Logger,
    level: int,
    message: str,
    /,
    *,
    exc_info: bool = False,
    **context: Any,
) -> None:
    """Emit a log record with structured, typed context.

    Python's ``Logger.debug``/``info``/etc. are typed without the ``extra``
    keyword, so arbitrary structured fields fail strict type checking when
    passed directly. This helper is the single, typed funnel for structured
    logging; the extra fields are rendered by :class:`_JsonFormatter`.

    Args:
        logger: Target logger.
        level: A ``logging`` level constant, e.g. ``logging.INFO``.
        message: Human-readable message.
        exc_info: Whether to attach the current exception traceback.
        **context: Arbitrary structured fields to attach to the record.
    """
    if logger.isEnabledFor(level):
        logger.log(level, message, extra=context, exc_info=exc_info)
