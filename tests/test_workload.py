"""Tests for deterministic workload generation."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from inferbench.config import WorkloadConfig, WorkloadKind
from inferbench.workload import generate_workload


class TestDeterminism:
    """Same seed must produce byte-identical workloads, or comparisons break."""

    def test_same_seed_same_prompts(self) -> None:
        config = WorkloadConfig(
            kind=WorkloadKind.randomized,
            num_requests=20,
            seed=42,
            prompt_tokens_min=8,
            prompt_tokens_max=64,
        )
        first = generate_workload(config)
        second = generate_workload(config)
        assert [s.prompt for s in first.specs] == [s.prompt for s in second.specs]

    def test_different_seed_different_prompts(self) -> None:
        base: dict[str, object] = {
            "kind": WorkloadKind.randomized,
            "num_requests": 20,
            "prompt_tokens_min": 8,
            "prompt_tokens_max": 64,
        }
        a = generate_workload(WorkloadConfig.model_validate({"seed": 1, **base}))
        b = generate_workload(WorkloadConfig.model_validate({"seed": 2, **base}))
        assert [s.prompt for s in a.specs] != [s.prompt for s in b.specs]

    def test_request_ids_unique_and_ordered(self) -> None:
        plan = generate_workload(WorkloadConfig(num_requests=5))
        ids = [s.request_id for s in plan.specs]
        assert len(set(ids)) == 5
        assert ids == sorted(ids)


class TestFixedWorkload:
    def test_lengths_match_config(self) -> None:
        plan = generate_workload(
            WorkloadConfig(
                kind=WorkloadKind.fixed, num_requests=6, prompt_tokens=100, output_tokens=50
            )
        )
        assert len(plan.specs) == 6
        assert all(s.target_prompt_tokens == 100 for s in plan.specs)
        assert all(s.max_output_tokens == 50 for s in plan.specs)
        assert all(s.class_label == "fixed" for s in plan.specs)

    def test_summary_reports_distribution(self) -> None:
        plan = generate_workload(WorkloadConfig(num_requests=4, prompt_tokens=64))
        assert plan.prompt_token_summary["count"] == 4
        assert plan.prompt_token_summary["min"] == 64
        assert plan.prompt_token_summary["max"] == 64


class TestRandomizedWorkload:
    def test_lengths_within_bounds(self) -> None:
        plan = generate_workload(
            WorkloadConfig(
                kind=WorkloadKind.randomized,
                num_requests=50,
                prompt_tokens_min=10,
                prompt_tokens_max=20,
            )
        )
        lengths = [s.target_prompt_tokens for s in plan.specs]
        assert all(10 <= length <= 20 for length in lengths)
        # A random distribution with 50 draws should not be constant.
        assert len(set(lengths)) > 1

    def test_output_lengths_randomized_when_bounded(self) -> None:
        plan = generate_workload(
            WorkloadConfig(
                kind=WorkloadKind.randomized,
                num_requests=30,
                prompt_tokens_min=10,
                prompt_tokens_max=20,
                output_tokens_min=5,
                output_tokens_max=15,
            )
        )
        outputs = [s.max_output_tokens for s in plan.specs]
        assert all(5 <= value <= 15 for value in outputs)


class TestMixedWorkload:
    def test_produces_multiple_classes(self) -> None:
        plan = generate_workload(
            WorkloadConfig(
                kind=WorkloadKind.mixed,
                num_requests=60,
                prompt_tokens_min=10,
                prompt_tokens_max=90,
                output_tokens_min=10,
                output_tokens_max=90,
            )
        )
        labels = Counter(s.class_label for s in plan.specs)
        # All three buckets should appear with this many draws.
        assert set(labels) == {"short", "medium", "long"}

    def test_long_class_has_longer_prompts_than_short(self) -> None:
        plan = generate_workload(
            WorkloadConfig(
                kind=WorkloadKind.mixed,
                num_requests=120,
                prompt_tokens_min=10,
                prompt_tokens_max=90,
                seed=7,
            )
        )
        short = [s.target_prompt_tokens for s in plan.specs if s.class_label == "short"]
        long = [s.target_prompt_tokens for s in plan.specs if s.class_label == "long"]
        assert short and long
        assert max(short) < min(long)


class TestTokenCountMode:
    def test_heuristic_mode_recorded_when_no_tokenizer(self) -> None:
        plan = generate_workload(WorkloadConfig(num_requests=2))
        assert plan.token_count_mode == "heuristic"

    def test_tokenizer_mode_recorded_and_used(self) -> None:
        # A trivial tokenizer that decodes ids into whitespace-joined letters.
        def decode(ids: Sequence[int]) -> str:
            return " ".join(f"t{i}" for i in ids)

        plan = generate_workload(WorkloadConfig(num_requests=3, prompt_tokens=12), decode=decode)
        assert plan.token_count_mode == "tokenizer"
        # Decoded prompts should contain the requested number of pieces.
        assert all(len(s.prompt.split()) == 12 for s in plan.specs)

    def test_tokenizer_failure_falls_back(self) -> None:
        def bad_decode(ids: Sequence[int]) -> str:
            raise RuntimeError("no such tokenizer")

        plan = generate_workload(WorkloadConfig(num_requests=2, prompt_tokens=8), decode=bad_decode)
        # Mode is still 'tokenizer' (one was supplied) but prompts are usable.
        assert all(s.prompt for s in plan.specs)
