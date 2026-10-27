"""Tests for experiment definitions and expansion.

The controlled-variable check is the feature that makes an experiment honest, so
it gets the heaviest coverage here: an experiment that claims to vary one thing
but silently varies another must be refused.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from pydantic import ValidationError

from inferbench.config import RunConfig
from inferbench.errors import ConfigurationError
from inferbench.experiment import (
    ExperimentConfig,
    Variant,
    expand_experiment,
    load_experiment,
)


def base(**overrides: object) -> RunConfig:
    defaults: dict[str, object] = {
        "run_name": "exp",
        "experiment": "X",
        "workload": {"kind": "fixed", "num_requests": 4, "prompt_tokens": 128, "output_tokens": 32},
        "load": {"concurrency": 1, "warmup_requests": 0, "repetitions": 1},
    }
    defaults.update(overrides)
    return RunConfig.model_validate(defaults)


class TestExpansion:
    def test_one_config_per_variant(self) -> None:
        experiment = ExperimentConfig(
            name="A",
            varies="load.concurrency",
            base=base(),
            variants=[
                Variant(name="c1", overrides={"load": {"concurrency": 1}}),
                Variant(name="c4", overrides={"load": {"concurrency": 4}}),
            ],
        )
        expanded = expand_experiment(experiment)
        assert len(expanded) == 2
        assert [r.config.load.concurrency for r in expanded] == [1, 4]

    def test_run_names_are_namespaced_by_experiment(self) -> None:
        experiment = ExperimentConfig(
            name="B",
            varies="workload.prompt_tokens",
            base=base(),
            variants=[
                Variant(name="short", overrides={"workload": {"prompt_tokens": 64}}),
                Variant(name="long", overrides={"workload": {"prompt_tokens": 512}}),
            ],
        )
        names = [r.config.run_name for r in expand_experiment(experiment)]
        assert names == ["B--short", "B--long"]

    def test_unlisted_fields_are_inherited(self) -> None:
        experiment = ExperimentConfig(
            name="A",
            varies="load.concurrency",
            base=base(),
            variants=[
                Variant(name="a", overrides={"load": {"concurrency": 1}}),
                Variant(name="b", overrides={"load": {"concurrency": 2}}),
            ],
        )
        expanded = expand_experiment(experiment)
        # Output length was never mentioned, so it must survive untouched.
        assert all(r.config.workload.output_tokens == 32 for r in expanded)

    def test_expansion_is_deterministic(self) -> None:
        experiment = ExperimentConfig(
            name="A",
            varies="load.concurrency",
            base=base(),
            variants=[
                Variant(name="a", overrides={"load": {"concurrency": 1}}),
                Variant(name="b", overrides={"load": {"concurrency": 2}}),
            ],
        )
        first = [r.config.model_dump() for r in expand_experiment(experiment)]
        second = [r.config.model_dump() for r in expand_experiment(experiment)]
        assert first == second

    def test_inherited_baseline_experiment_name_replaced(self) -> None:
        """A base left at the default experiment name takes the experiment's."""
        experiment = ExperimentConfig(
            name="C",
            varies="load.concurrency",
            base=RunConfig.model_validate(
                {"run_name": "x", "workload": {"num_requests": 2}, "load": {"concurrency": 1}}
            ),
            variants=[
                Variant(name="a", overrides={"load": {"concurrency": 1}}),
                Variant(name="b", overrides={"load": {"concurrency": 2}}),
            ],
        )
        assert all(r.config.experiment == "C" for r in expand_experiment(experiment))


class TestVariantValidation:
    def test_duplicate_variant_names_rejected(self) -> None:
        with pytest.raises(ValidationError, match="unique"):
            ExperimentConfig(
                name="A",
                base=base(),
                variants=[Variant(name="dup"), Variant(name="dup")],
            )

    def test_empty_variant_list_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ExperimentConfig(name="A", base=base(), variants=[])

    def test_unknown_override_section_rejected(self) -> None:
        with pytest.raises(ValidationError, match="override sections"):
            Variant(name="a", overrides={"nonsense": {"x": 1}})

    def test_non_mapping_section_rejected(self) -> None:
        with pytest.raises(ValidationError, match="must be a mapping"):
            Variant(name="a", overrides={"load": "not-a-mapping"})

    def test_invalid_override_value_rejected(self) -> None:
        experiment = ExperimentConfig(
            name="A",
            base=base(),
            variants=[Variant(name="bad", overrides={"load": {"concurrency": -5}})],
        )
        with pytest.raises(ConfigurationError, match="invalid"):
            expand_experiment(experiment)


class TestControlledVariableGuard:
    """The guard that keeps an experiment measuring one thing."""

    def test_unvaried_axis_rejected(self) -> None:
        """Declaring an axis while every variant agrees is a mistake."""
        experiment = ExperimentConfig(
            name="A",
            varies="load.concurrency",
            base=base(),
            variants=[
                Variant(name="a", overrides={"load": {"concurrency": 4}}),
                Variant(name="b", overrides={"load": {"concurrency": 4}}),
            ],
        )
        with pytest.raises(ConfigurationError, match="would not vary anything"):
            expand_experiment(experiment)

    def test_second_field_in_same_section_rejected(self) -> None:
        """Changing concurrency *and* warm-up makes the result unattributable."""
        experiment = ExperimentConfig(
            name="A",
            varies="load.concurrency",
            base=base(),
            variants=[
                Variant(name="a", overrides={"load": {"concurrency": 1}}),
                Variant(name="b", overrides={"load": {"concurrency": 8, "warmup_requests": 8}}),
            ],
        )
        with pytest.raises(ConfigurationError, match="also changes load.warmup_requests"):
            expand_experiment(experiment)

    def test_different_section_rejected(self) -> None:
        experiment = ExperimentConfig(
            name="A",
            varies="load.concurrency",
            base=base(),
            variants=[
                Variant(name="a", overrides={"load": {"concurrency": 1}}),
                Variant(
                    name="b",
                    overrides={"load": {"concurrency": 2}, "workload": {"prompt_tokens": 512}},
                ),
            ],
        )
        with pytest.raises(ConfigurationError, match="also changes the 'workload' section"):
            expand_experiment(experiment)

    def test_malformed_varies_path_rejected(self) -> None:
        experiment = ExperimentConfig(
            name="A",
            varies="concurrency",
            base=base(),
            variants=[
                Variant(name="a", overrides={"load": {"concurrency": 1}}),
                Variant(name="b", overrides={"load": {"concurrency": 2}}),
            ],
        )
        with pytest.raises(ConfigurationError, match="dotted path"):
            expand_experiment(experiment)

    def test_unknown_section_in_varies_rejected(self) -> None:
        experiment = ExperimentConfig(
            name="A",
            varies="nonsense.field",
            base=base(),
            variants=[
                Variant(name="a", overrides={"load": {"concurrency": 1}}),
                Variant(name="b", overrides={"load": {"concurrency": 2}}),
            ],
        )
        with pytest.raises(ConfigurationError, match="unknown section"):
            expand_experiment(experiment)

    def test_no_varies_skips_the_check(self) -> None:
        """Without a declared axis, heterogeneous variants are permitted."""
        experiment = ExperimentConfig(
            name="A",
            base=base(),
            variants=[
                Variant(name="a", overrides={"load": {"concurrency": 1}}),
                Variant(
                    name="b",
                    overrides={"load": {"concurrency": 2}, "workload": {"prompt_tokens": 512}},
                ),
            ],
        )
        assert len(expand_experiment(experiment)) == 2


class TestLoadExperimentFile:
    def test_shipped_experiments_load(self) -> None:
        for name in (
            "experiment_a_baseline.yaml",
            "experiment_b1_prompt_length.yaml",
            "experiment_b2_output_length.yaml",
            "experiment_b3_mixed_lengths.yaml",
        ):
            path = Path(__file__).resolve().parents[1] / "configs" / name
            experiment = load_experiment(path)
            expanded = expand_experiment(experiment)
            assert len(expanded) == len(experiment.variants)
            # Every shipped experiment must declare what it varies.
            assert experiment.varies

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigurationError, match="not found"):
            load_experiment(tmp_path / "nope.yaml")

    def test_non_mapping_file(self, tmp_config: Callable[..., Path]) -> None:
        path = tmp_config("- a\n- b\n")
        with pytest.raises(ConfigurationError, match="must be a mapping"):
            load_experiment(path)

    def test_malformed_yaml(self, tmp_config: Callable[..., Path]) -> None:
        path = tmp_config("key: [unclosed\n")
        with pytest.raises(ConfigurationError, match="Could not read"):
            load_experiment(path)

    def test_invalid_experiment_wrapped(self, tmp_config: Callable[..., Path]) -> None:
        path = tmp_config("name: A\nvariants: []\n")
        with pytest.raises(ConfigurationError, match="Invalid experiment"):
            load_experiment(path)
