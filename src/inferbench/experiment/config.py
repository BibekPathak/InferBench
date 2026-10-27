"""Experiment definitions: multi-run studies with a declared variable.

A single :class:`~inferbench.config.RunConfig` describes one measurement. An
*experiment* is the next level up: a set of runs that differ along one or more
deliberately varied axes, executed together, so that the effect of those axes
can be studied.

Why this is a separate concept rather than "run the config N times":

- **A declared controlled variable** makes the report's claims checkable. If an
  experiment says it varies ``load.concurrency``, the loader verifies that
  every variant really does keep everything *else* fixed, and refuses if not.
  Without that check, an "experiment" can silently vary two things at once and
  produce a confident, wrong conclusion.
- **A shared base plus deltas** keeps variant definitions short and makes the
  invariant (what is *not* changing) explicit and diffable.
- **Repetition and ordering** live here because they are properties of the
  study, not of any one run.

The expansion of an experiment into ``RunConfig`` objects is deterministic and
side-effect free, so the same experiment file always yields the same plan.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from inferbench.config import RunConfig
from inferbench.errors import ConfigurationError

__all__ = [
    "Variant",
    "ExperimentConfig",
    "load_experiment",
    "expand_experiment",
]


class Variant(BaseModel):
    """One arm of an experiment: a labelled delta applied to the base config.

    ``overrides`` is a nested patch, e.g.::

        {"load": {"concurrency": 8}, "workload": {"prompt_tokens": 512}}

    Only keys present in ``overrides`` change; everything else is inherited from
    the base. This is what keeps the controlled variable verifiable.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, description="Short label, used in reports and run names.")
    description: str = ""
    overrides: dict[str, Any] = Field(
        default_factory=dict,
        description="Nested patch applied to the base run config.",
    )

    @field_validator("overrides")
    @classmethod
    def _reject_unknown_sections(cls, value: dict[str, Any]) -> dict[str, Any]:
        allowed = {"backend", "workload", "load", "run_name", "experiment", "notes"}
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(
                f"override sections must be among {sorted(allowed)}, got {sorted(unknown)}"
            )
        for section in ("backend", "workload", "load"):
            if section in value and not isinstance(value[section], dict):
                raise ValueError(f"override section {section!r} must be a mapping")
        return value


class ExperimentConfig(BaseModel):
    """A complete multi-run experiment definition."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, description="Experiment identifier, e.g. 'A' or 'B'.")
    description: str = ""
    #: Which config path is deliberately varied, e.g. ``"load.concurrency"``.
    #: Used to verify variants only differ where they are supposed to.
    varies: str = Field(
        default="",
        description=(
            "Dotted config path the experiment claims to vary, e.g. 'load.concurrency'. "
            "Empty means no check is performed."
        ),
    )
    base: RunConfig = Field(default_factory=RunConfig)
    variants: list[Variant] = Field(min_length=1)

    @field_validator("variants")
    @classmethod
    def _unique_names(cls, value: list[Variant]) -> list[Variant]:
        names = [variant.name for variant in value]
        duplicates = {name for name in names if names.count(name) > 1}
        if duplicates:
            raise ValueError(f"variant names must be unique; repeated: {sorted(duplicates)}")
        return value


@dataclass(frozen=True, slots=True)
class ExpandedRun:
    """One concrete run produced by expanding an experiment."""

    variant: Variant
    config: RunConfig


def _apply_overrides(base: RunConfig, overrides: dict[str, Any]) -> RunConfig:
    """Return a copy of ``base`` with ``overrides`` applied.

    Sections are merged shallowly (a variant replaces whole fields within a
    section rather than deep-merging sub-mappings), which keeps the semantics
    obvious: an override either sets a field or leaves it inherited.
    """
    payload = base.model_dump(mode="python")
    for section, value in overrides.items():
        if section in ("backend", "workload", "load"):
            merged = dict(payload.get(section) or {})
            merged.update(value)
            payload[section] = merged
        else:
            payload[section] = value
    return RunConfig.model_validate(payload)


def expand_experiment(experiment: ExperimentConfig) -> list[ExpandedRun]:
    """Expand an experiment into concrete run configs, one per variant.

    Each run is named ``<experiment>--<variant>`` so runs from the same study
    sort and group together in the store.

    Raises:
        ConfigurationError: A variant declares an override the base rejects, or
            variants differ outside the declared ``varies`` path.
    """
    expanded: list[ExpandedRun] = []
    for variant in experiment.variants:
        try:
            config = _apply_overrides(experiment.base, variant.overrides)
        except Exception as exc:  # pydantic ValidationError
            raise ConfigurationError(
                f"Variant {variant.name!r} of experiment {experiment.name!r} is invalid: {exc}"
            ) from exc

        # Tag each run deterministically so a study groups up in the store.
        config.run_name = f"{experiment.name}--{variant.name}"
        if not config.experiment:
            config.experiment = experiment.name
        elif config.experiment == "baseline":
            # Inherited default; replace it with the experiment's own name.
            config.experiment = experiment.name
        expanded.append(ExpandedRun(variant=variant, config=config))

    if experiment.varies:
        _verify_controlled_variable(experiment, expanded)
    return expanded


def _verify_controlled_variable(experiment: ExperimentConfig, expanded: list[ExpandedRun]) -> None:
    """Ensure variants differ *only* where the experiment claims they do.

    This is the guard that keeps an experiment honest. Without it, a stray edit
    to one variant's prompt length would look like "the effect of concurrency"
    while actually measuring two things.

    Raises:
        ConfigurationError: Variants differ outside the declared path.
    """
    section, _, field = experiment.varies.partition(".")
    if not section or not field:
        raise ConfigurationError(
            f"experiment.varies must be a dotted path like 'load.concurrency', "
            f"got {experiment.varies!r}"
        )

    varied: list[Any] = []
    for run in expanded:
        section_value = getattr(run.config, section, None)
        if section_value is None:
            raise ConfigurationError(f"experiment.varies names unknown section {section!r}")
        field_value = getattr(section_value, field, None)
        varied.append(field_value)

    if len(set(map(repr, varied))) < 2:
        raise ConfigurationError(
            f"experiment {experiment.name!r} declares varies={experiment.varies!r}, "
            f"but every variant has the same value ({varied[0]!r}); the experiment "
            "would not vary anything"
        )

    # Everything except the declared field must be identical across variants.
    reference = expanded[0].config
    reference_payload = reference.model_dump(mode="python")
    reference_section = dict(reference_payload.get(section) or {})

    for run in expanded[1:]:
        payload = run.config.model_dump(mode="python")
        current_section = dict(payload.get(section) or {})
        for key, value in reference_section.items():
            if key == field:
                continue
            current = current_section.get(key)
            if current != value:
                raise ConfigurationError(
                    f"experiment {experiment.name!r} declares varies={experiment.varies!r}, "
                    f"but variant {run.variant.name!r} also changes {section}.{key} "
                    f"({value!r} -> {current!r}). An experiment must vary one thing "
                    "at a time; split it or adjust `varies`."
                )
        # Other sections must be untouched entirely.
        for other in ("backend", "workload", "load"):
            if other == section:
                continue
            if payload.get(other) != reference_payload.get(other):
                raise ConfigurationError(
                    f"experiment {experiment.name!r} declares varies={experiment.varies!r}, "
                    f"but variant {run.variant.name!r} also changes the {other!r} section"
                )


def load_experiment(path: str | Path) -> ExperimentConfig:
    """Load and validate an experiment definition from YAML/JSON.

    Raises:
        ConfigurationError: The file is missing, unparseable, or invalid.
    """
    experiment_path = Path(path)
    if not experiment_path.is_file():
        raise ConfigurationError(f"Experiment file not found: {experiment_path}")

    try:
        parsed = yaml.safe_load(experiment_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigurationError(
            f"Could not read experiment file {experiment_path}: {exc}"
        ) from exc

    if not isinstance(parsed, dict):
        raise ConfigurationError(
            f"Experiment file {experiment_path} must be a mapping at the top level"
        )

    try:
        return ExperimentConfig.model_validate(parsed)
    except Exception as exc:  # pydantic ValidationError
        raise ConfigurationError(f"Invalid experiment file {experiment_path}: {exc}") from exc
