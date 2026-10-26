"""Shared pytest fixtures for the InferBench test suite."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture()
def fixtures_dir() -> Path:
    """Directory holding static test fixtures."""
    return FIXTURES_DIR


@pytest.fixture()
def tmp_config(tmp_path: Path) -> Callable[..., Path]:
    """Factory writing a YAML config with given content and returning its path."""

    def _write(content: str, name: str = "config.yaml") -> Path:
        path = tmp_path / name
        path.write_text(content, encoding="utf-8")
        return path

    return _write
