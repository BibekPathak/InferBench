"""Version and user-agent helpers.

Kept in a tiny module so adapters can identify themselves to servers without
importing the CLI or the package root (which would create import cycles).
"""

from __future__ import annotations

from inferbench import __version__

__all__ = ["http_user_agent", "tool_version"]


def tool_version() -> str:
    """InferBench's own version, recorded in all run metadata."""
    return __version__


def http_user_agent() -> str:
    """User-Agent string sent to backend servers."""
    return f"inferbench/{__version__}"
