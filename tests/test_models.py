"""Tests for model profiles."""

from __future__ import annotations

from pathlib import Path

import pytest

from inferbench.config import BackendKind
from inferbench.errors import ConfigurationError
from inferbench.models import (
    DEFAULT_PROFILES,
    get_profile,
    list_profiles,
    load_profiles,
    resolve_profile,
)


class TestDefaults:
    def test_builtin_profiles_present(self) -> None:
        names = {p.name for p in DEFAULT_PROFILES}
        assert "mock" in names
        assert any(p.requires_gpu for p in DEFAULT_PROFILES)

    def test_mock_profile_needs_no_gpu(self) -> None:
        profile = get_profile("mock")
        assert profile.requires_gpu is False
        assert profile.backend_kind is BackendKind.mock

    def test_no_profile_contains_a_filesystem_path(self) -> None:
        """Model identifiers must be server-side names, never local paths."""
        for profile in DEFAULT_PROFILES:
            assert not profile.model.startswith("/")
            assert "~" not in profile.model

    def test_list_is_sorted(self) -> None:
        names = [p.name for p in list_profiles()]
        assert names == sorted(names)


class TestResolveProfile:
    def test_produces_backend_config(self) -> None:
        config = resolve_profile("mock")
        assert config.kind is BackendKind.mock
        assert config.model == "mock-model"

    def test_overrides_applied(self) -> None:
        config = resolve_profile("mock", base_url="http://example:9000/", request_timeout_s=5.0)
        assert config.base_url == "http://example:9000"
        assert config.request_timeout_s == 5.0

    def test_unknown_profile_lists_known(self) -> None:
        with pytest.raises(ConfigurationError, match="Unknown model profile"):
            get_profile("does-not-exist")

    def test_to_dict_is_serializable(self) -> None:
        import json

        payload = get_profile("mock").to_dict()
        assert json.loads(json.dumps(payload))["backend_kind"] == "mock"


class TestProfileFile:
    def test_user_profiles_merged(self, tmp_path: Path) -> None:
        path = tmp_path / "profiles.yaml"
        path.write_text(
            "my-model:\n"
            "  backend_kind: vllm\n"
            "  model: my-org/my-model\n"
            "  description: custom\n"
            "  parameters_billions: 3.5\n",
            encoding="utf-8",
        )
        profiles = load_profiles(path)
        assert "my-model" in profiles
        assert profiles["my-model"].parameters_billions == 3.5
        # Built-ins survive.
        assert "mock" in profiles

    def test_user_can_override_builtin(self, tmp_path: Path) -> None:
        path = tmp_path / "profiles.yaml"
        path.write_text("mock:\n  backend_kind: mock\n  model: my-mock\n", encoding="utf-8")
        assert load_profiles(path)["mock"].model == "my-mock"

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigurationError, match="not found"):
            load_profiles(tmp_path / "nope.yaml")

    def test_non_mapping_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "profiles.yaml"
        path.write_text("- a\n- b\n", encoding="utf-8")
        with pytest.raises(ConfigurationError, match="must map"):
            load_profiles(path)

    def test_missing_model_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "profiles.yaml"
        path.write_text("bad:\n  dtype: bfloat16\n", encoding="utf-8")
        with pytest.raises(ConfigurationError, match="missing required 'model'"):
            load_profiles(path)

    def test_unknown_field_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "profiles.yaml"
        path.write_text("bad:\n  model: m\n  nonsense: 1\n", encoding="utf-8")
        with pytest.raises(ConfigurationError, match="unknown field"):
            load_profiles(path)

    def test_invalid_backend_kind_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "profiles.yaml"
        path.write_text("bad:\n  model: m\n  backend_kind: tensorrt\n", encoding="utf-8")
        with pytest.raises(ConfigurationError, match="invalid backend_kind"):
            load_profiles(path)

    def test_non_numeric_parameters_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "profiles.yaml"
        path.write_text("bad:\n  model: m\n  parameters_billions: huge\n", encoding="utf-8")
        with pytest.raises(ConfigurationError, match="must be numeric"):
            load_profiles(path)

    def test_empty_file_is_ok(self, tmp_path: Path) -> None:
        path = tmp_path / "profiles.yaml"
        path.write_text("", encoding="utf-8")
        assert "mock" in load_profiles(path)
