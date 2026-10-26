"""Model profile registry."""

from __future__ import annotations

from inferbench.models.profiles import (
    DEFAULT_PROFILES,
    ModelProfile,
    get_profile,
    list_profiles,
    load_profiles,
    resolve_profile,
)

__all__ = [
    "DEFAULT_PROFILES",
    "ModelProfile",
    "get_profile",
    "list_profiles",
    "load_profiles",
    "resolve_profile",
]
