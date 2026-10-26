"""Model profile registry.

A model profile is a named, reusable description of *what to run* — a backend,
a model identifier, and the facts needed to interpret results. Profiles exist so
that configuration files, reports, and comparisons can reference a stable name
instead of repeating (and drifting from) raw backend settings.

Deliberately **not** included:

- Local filesystem paths to weights. Profiles name models as a server knows
  them; where the weights live is a deployment concern.
- API keys or tokens.
- Cloud-provider specifics.

Profiles may be defined in two ways:

1. Built-in defaults (listed below), which are conventional names only.
2. A user profile file (``--profiles path.yaml``) that overrides or adds
   entries, so nothing is hardcoded to one environment.

A profile does not guarantee the model is available: :func:`resolve_profile`
only produces a :class:`~inferbench.config.BackendConfig`. Whether the backend
actually serves the model is validated by the adapter at run time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from inferbench.config import BackendConfig, BackendKind
from inferbench.errors import ConfigurationError

__all__ = [
    "ModelProfile",
    "list_profiles",
    "get_profile",
    "load_profiles",
    "resolve_profile",
    "DEFAULT_PROFILES",
]


@dataclass(frozen=True, slots=True)
class ModelProfile:
    """A named way to benchmark a particular model on a particular backend."""

    name: str
    backend_kind: BackendKind
    model: str
    description: str = ""
    #: Approximate parameter count, for grouping and reporting. ``None`` is
    #: honest; it is never guessed from the model name.
    parameters_billions: float | None = None
    #: Conventional dtype the profile is intended to run at.
    dtype: str | None = None
    #: Quantization method, e.g. ``"awq"``, when the profile targets one.
    quantization: str | None = None
    #: Whether this profile requires GPU hardware to be meaningful.
    requires_gpu: bool = True
    #: Backend-specific hints (e.g. max model length) passed through verbatim.
    backend_hints: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "backend_kind": self.backend_kind.value,
            "model": self.model,
            "description": self.description,
            "parameters_billions": self.parameters_billions,
            "dtype": self.dtype,
            "quantization": self.quantization,
            "requires_gpu": self.requires_gpu,
            "backend_hints": dict(self.backend_hints),
        }

    def to_backend_config(self, **overrides: Any) -> BackendConfig:
        """Build a :class:`BackendConfig` for this profile.

        Args:
            **overrides: Values applied on top of the profile, such as
                ``base_url`` or ``request_timeout_s``.
        """
        payload: dict[str, Any] = {
            "kind": self.backend_kind,
            "model": self.model,
        }
        payload.update(overrides)
        return BackendConfig.model_validate(payload)


#: Built-in profiles. The mock entry is always usable; the others are
#: conventional names that a real deployment maps onto its served models.
DEFAULT_PROFILES: tuple[ModelProfile, ...] = (
    ModelProfile(
        name="mock",
        backend_kind=BackendKind.mock,
        model="mock-model",
        description=(
            "Deterministic synthetic backend. No GPU or model download. "
            "Validates the pipeline; never represents real performance."
        ),
        requires_gpu=False,
    ),
    ModelProfile(
        name="llama-3.1-8b",
        backend_kind=BackendKind.vllm,
        model="meta-llama/Llama-3.1-8B-Instruct",
        description="Common 8B instruct baseline; fits a single 24GB accelerator.",
        parameters_billions=8.0,
        dtype="bfloat16",
    ),
    ModelProfile(
        name="llama-3.1-8b-awq",
        backend_kind=BackendKind.vllm,
        model="meta-llama/Llama-3.1-8B-Instruct-AWQ",
        description="4-bit AWQ variant for quantization comparisons (Experiment E).",
        parameters_billions=8.0,
        dtype="int4",
        quantization="awq",
    ),
    ModelProfile(
        name="qwen2.5-7b",
        backend_kind=BackendKind.vllm,
        model="Qwen/Qwen2.5-7B-Instruct",
        description="Alternative 7B instruct model for cross-model comparisons.",
        parameters_billions=7.0,
        dtype="bfloat16",
    ),
    ModelProfile(
        name="mistral-7b",
        backend_kind=BackendKind.vllm,
        model="mistralai/Mistral-7B-Instruct-v0.3",
        description="Widely used 7B instruct model; useful as a second data point.",
        parameters_billions=7.0,
        dtype="bfloat16",
    ),
)


def _parse_profile(name: str, payload: Any, *, source: Path | str) -> ModelProfile:
    """Validate one profile entry from a profile file."""
    if not isinstance(payload, dict):
        raise ConfigurationError(
            f"Profile {name!r} in {source} must be a mapping, got {type(payload).__name__}"
        )

    allowed = {
        "backend_kind",
        "model",
        "description",
        "parameters_billions",
        "dtype",
        "quantization",
        "requires_gpu",
        "backend_hints",
    }
    unknown = set(payload) - allowed
    if unknown:
        raise ConfigurationError(
            f"Profile {name!r} in {source} has unknown field(s): {sorted(unknown)}. "
            f"Allowed: {sorted(allowed)}"
        )

    if "model" not in payload:
        raise ConfigurationError(f"Profile {name!r} in {source} is missing required 'model'")

    kind_raw = payload.get("backend_kind", BackendKind.vllm.value)
    try:
        kind = BackendKind(kind_raw)
    except ValueError as exc:
        raise ConfigurationError(
            f"Profile {name!r} in {source} has invalid backend_kind {kind_raw!r}; "
            f"expected one of {[k.value for k in BackendKind]}"
        ) from exc

    hints = payload.get("backend_hints", {})
    if not isinstance(hints, dict):
        raise ConfigurationError(f"Profile {name!r} in {source}: backend_hints must be a mapping")

    params = payload.get("parameters_billions")
    if params is not None:
        try:
            params = float(params)
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(
                f"Profile {name!r} in {source}: parameters_billions must be numeric"
            ) from exc

    return ModelProfile(
        name=name,
        backend_kind=kind,
        model=str(payload["model"]),
        description=str(payload.get("description", "")),
        parameters_billions=params,
        dtype=payload.get("dtype"),
        quantization=payload.get("quantization"),
        requires_gpu=bool(payload.get("requires_gpu", kind is BackendKind.vllm)),
        backend_hints=dict(hints),
    )


def load_profiles(path: str | Path | None = None) -> dict[str, ModelProfile]:
    """Return built-in profiles, optionally merged with a user file.

    User entries override built-ins with the same name, so a deployment can
    redefine ``llama-3.1-8b`` to point at its own served model name.

    Raises:
        ConfigurationError: The file is missing, unparseable, or invalid.
    """
    profiles: dict[str, ModelProfile] = {profile.name: profile for profile in DEFAULT_PROFILES}
    if path is None:
        return profiles

    profile_path = Path(path)
    if not profile_path.is_file():
        raise ConfigurationError(f"Profile file not found: {profile_path}")

    try:
        parsed = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"Could not read profile file {profile_path}: {exc}") from exc

    if parsed is None:
        return profiles
    if not isinstance(parsed, dict):
        raise ConfigurationError(
            f"Profile file {profile_path} must map profile names to definitions"
        )

    for name, payload in parsed.items():
        profiles[str(name)] = _parse_profile(str(name), payload, source=profile_path)
    return profiles


def list_profiles(path: str | Path | None = None) -> list[ModelProfile]:
    """Return all known profiles, sorted by name."""
    return sorted(load_profiles(path).values(), key=lambda profile: profile.name)


def get_profile(name: str, path: str | Path | None = None) -> ModelProfile:
    """Look up one profile by name.

    Raises:
        ConfigurationError: No profile with that name exists.
    """
    profiles = load_profiles(path)
    try:
        return profiles[name]
    except KeyError:
        known = ", ".join(sorted(profiles))
        raise ConfigurationError(
            f"Unknown model profile {name!r}. Known profiles: {known}"
        ) from None


def resolve_profile(
    name: str,
    path: str | Path | None = None,
    **overrides: Any,
) -> BackendConfig:
    """Resolve a profile name into a ready-to-use :class:`BackendConfig`."""
    return get_profile(name, path).to_backend_config(**overrides)
