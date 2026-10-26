# InferBench

**Real-world LLM inference benchmarking and optimization platform.**

InferBench measures, compares, and explains the performance of real LLM
inference engines. It is a rigorous benchmarking system — reproducible
workloads, reliable streaming-based latency metrics, automated experiments, and
evidence-backed analysis — not a dashboard or an API wrapper.

> **Status: Phase 1 complete.** The benchmark engine runs end to end on CPU with
> a deterministic mock backend: workload generation → streaming execution →
> metrics → storage → Markdown report. The vLLM adapter is written against the
> supported OpenAI-compatible API but has **not** been validated against a live
> server (no GPU on the development host). See [`docs/roadmap.md`](docs/roadmap.md).

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

### Run a benchmark

```sh
# Runs the mock backend end to end, then writes a Markdown report.
inferbench run --config configs/baseline.yaml

# Report on a stored run.
inferbench report <run_id> --output-dir runs
```

The mock backend needs no GPU or model download. Its report is labelled
**synthetic** throughout — it validates the pipeline, not hardware performance.

To benchmark a real model, point the config at a vLLM server:

```yaml
backend:
  kind: vllm
  base_url: http://localhost:8000
  model: <served-model-name>
```

Each run writes to `runs/<run_id>/`:

| Artifact | Contents |
|----------|----------|
| `raw.jsonl` | One JSON object per request — the raw evidence |
| `metadata.json` | Config, environment, versions, provenance |
| `metrics.json` | Aggregated summary |
| `summary.csv` | Flat summary for spreadsheets |
| `environment.json` | Hardware/software fingerprint |
| `report.md` | Human-readable report |
| `chart_*.png` | Charts (when matplotlib is installed) |

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
  types.py          core data types (requests, records, metadata)
  version_info.py   version and user-agent helpers
  backends/         BackendAdapter interface, mock, vLLM
  workload/         deterministic workload generation
  metrics/          TTFT / TPOT / throughput computation
  runner/           experiment orchestration
  storage/          SQLite index + raw JSONL artifacts
  analysis/         Markdown reports and optional charts
configs/            example experiment configurations
docs/               methodology, roadmap
tests/              unit + integration tests (CPU-only)
```

## License

Apache-2.0.
