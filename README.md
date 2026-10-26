# InferBench

**Real-world LLM inference benchmarking and optimization platform.**

InferBench measures, compares, and explains the performance of real LLM
inference engines. It is a rigorous benchmarking system — reproducible
workloads, reliable streaming-based latency metrics, automated experiments, and
evidence-backed analysis — not a dashboard or an API wrapper.

> **Status: Phase 0 complete.** The scaffold, environment doctor, and quality
> gates are in place. The benchmark engine itself lands in Phase 1. See
> [`docs/roadmap.md`](docs/roadmap.md).

> **No GPU results are claimed anywhere in this project.** The development host
> has no discrete GPU, so real vLLM/GPU measurements are explicitly pending. Any
> example numbers produced with the mock backend are labelled synthetic.

## What it measures

- **TTFT** — time to first token
- **TPOT** — time per output token, and inter-token latency
- **End-to-end latency** — full request duration
- **Throughput** — requests/sec and output tokens/sec
- **Tail latency** — p50, p95, p99 with a documented aggregation method
- **Failures** — timeouts, cancellations, disconnects, partial generations
- **Hardware telemetry** — GPU utilization and peak memory, when available
- **Cost** — estimated cost per million output tokens from explicit pricing

## Design principles

1. **Measured ≠ simulated ≠ estimated.** These are kept distinct and labelled.
2. **Reproducibility first.** Seeds, versions, hardware, and model revisions are
   recorded; raw per-request data is preserved alongside aggregates.
3. **Graceful degradation.** No GPU is required to install, test, or run the
   mock path.
4. **Fair comparison.** Runs with different token distributions are never
   silently compared.

## Installation

Requires Python 3.11+.

```sh
# from the repository root
uv venv
uv pip install -e ".[dev]"
```

Optional extras:

```sh
uv pip install -e ".[vllm]"      # real backend client dependencies
uv pip install -e ".[analysis]"  # chart rendering for reports
```

## Quick start

```sh
# Inspect the environment (works with or without a GPU)
inferbench doctor

# Machine-readable environment report
inferbench doctor --json
```

`doctor` reports Python, CPU, memory, accelerator availability, disk capacity,
and the versions of tracked packages. On a machine without a GPU it prints the
reason the accelerator is unavailable instead of failing.

## Development

```sh
ruff check .          # lint
ruff format --check . # formatting
mypy src tests        # strict type checking
pytest -q             # tests (CPU-only; GPU tests are marked and excluded)
```

CI runs all of the above on Python 3.11 and 3.12, plus a doctor smoke test.

## Project layout

```
src/inferbench/
  cli.py            command-line interface
  config.py         typed, validated configuration
  env.py            environment and hardware inspection
  errors.py         typed exception hierarchy
  logging_utils.py  structured (JSON-lines) logging
configs/            example experiment configurations
docs/               methodology, roadmap
tests/              unit tests (CPU-only)
```

Modules for the engine (`backends/`, `workload/`, `metrics/`, `runner/`,
`storage/`, `analysis/`) arrive in Phase 1.

## License

Apache-2.0.
