# InferBench Roadmap

Phases are ordered by dependency. Each phase ends green on lint, type checks,
and tests before the next begins. Real GPU measurements are deferred until a
GPU host is available; nothing is ever fabricated to fill that gap.

| Phase | Scope | Status |
|-------|-------|--------|
| 0 | Scaffold, environment doctor, CI | ✅ Complete |
| 1 | Vertical slice: run → metrics → store → report (mock + vLLM adapter) | ✅ Complete |
| 2 | Comparison, validation, fairness gating, model profiles | ✅ Complete |
| 3 | Experiments A & B: baseline + workload sensitivity | ✅ Complete |
| 4 | Experiment C: concurrency & saturation | ✅ Complete |
| 5 | Experiments D & E: prefix caching & quantization (GPU-gated execution) | ⏳ Planned |
| 6 | Experiment F: InferSched simulation adapter | ⏳ Planned |
| 7 | Deployment, observability, docs | ⏳ Planned |
| 8 | Roadmap, illustrative report, release | ⏳ Planned |

## Phase 1 — Vertical slice (complete)

Delivered the full pipeline end to end, CPU-only:

- **Backend adapters** behind a documented `BackendAdapter` interface:
  - `MockBackend` — deterministic, with an explicit analytic timing model
    (TTFT = base + prompt/prefill_rate; TPOT = base × contention factor) so
    metrics can be verified against hand-computed values.
  - `VLLMBackend` — OpenAI-compatible streaming over `httpx`, with readiness
    polling, model validation, usage parsing, and timeout/disconnect handling.
    **Not yet validated against a live server** (no GPU host); its streaming and
    error paths are unit-tested against a mock HTTP transport.
- **Workload generator** — fixed / randomized / mixed shapes, seeded and
  deterministic, with tokenizer and heuristic token-count modes (the mode used
  is recorded in metadata).
- **Metrics engine** — TTFT, TPOT, inter-token latency, E2E, request and token
  throughput, nearest-rank p50/p90/p95/p99, failure accounting, and
  offered-vs-completed throughput separation.
- **Runner** — warm-up/measurement separation, repetitions, closed-loop and
  rate-limited arrival patterns, and exactly-one-record-per-request accounting.
- **Storage** — SQLite index plus per-run `raw.jsonl`, `metadata.json`,
  `metrics.json`, `summary.csv`, `environment.json`.
- **Reporting** — Markdown reports with a provenance banner, latency/throughput/
  failure tables, optional charts, and an explicit limitations section.
- **CLI** — `run` and `report` are functional (the remaining commands arrive in
  Phase 2).

Verification: 152 tests pass on CPU, including metrics validated against
analytic ground truth (Definition of Done #2) and an auditability test that
recomputes the summary from raw JSONL alone (DoD #3).

## Phase 2 — Comparison, validation, and fairness gating (complete)

Every CLI command in the specification is now implemented.

- **`inferbench models`** — lists built-in model profiles and merges optional
  user profile files. Profiles name models as a server knows them; no
  filesystem paths, API keys, or cloud specifics.
- **`inferbench validate <run_id>`** — checks artifacts, metadata integrity, and
  data consistency, emitting typed findings at `error`/`warning`/`info`
  severities. Exits non-zero when a run is invalid, so it is usable in CI.
  Crucially it re-derives the summary from the raw JSONL and reports any
  disagreement: a run whose stored numbers do not match its own evidence is
  declared invalid.
- **`inferbench compare <a> <b>`** — compares two runs behind a **fairness gate**
  that classifies differences into blocking (refused: different prompt/output
  distributions, workload kind, or request counts), warned (different model,
  backend, warm-up), and the experimental axis itself (concurrency). Includes
  bootstrap confidence intervals on throughput and states whether the observed
  difference exceeds run-to-run variability.
- **Failure-mode coverage** — 77 new tests, most of them negative: corrupted
  summaries, missing artifacts, duplicate records, negative and inconsistent
  timings, token-count mismatches, and incompatible-run refusals.

## Phase 3 — Experiments A & B: baseline and workload sensitivity (complete)

A research *study* — several runs that differ along one declared axis — became a
first-class concept, with a guard that keeps it honest.

- **Model profiles** — `inferbench.models` with built-in profiles (mock,
  llama-3.1-8b, llama-3.1-8b-awq, qwen2.5-7b, mistral-7b), each carrying
  description, dtype, quantization, and a `requires_gpu` flag.
- **Experiment definitions** — a YAML file declares a `varies` dotted path (the
  single axis under study), a shared `base` config, and a list of `variants`.
  Expansion into concrete run configs is deterministic and side-effect free.
- **The controlled-variable guard** — the feature that makes an experiment
  honest: if an experiment claims to vary `load.concurrency`, the loader
  verifies every variant keeps everything else fixed and refuses the experiment
  otherwise. A stray edit cannot silently turn a one-variable study into a
  two-variable one and produce a confident, wrong conclusion.
- **Study runner and reporting** — `StudyRunner` executes every variant against
  one shared backend (so warm-up cost is not repaid per variant), contains
  per-variant failures without aborting the study, and writes a trend report
  with per-variant metrics, run-to-run variability, a **saturation-knee
  heuristic**, and the same provenance banner as single-run reports.
- **Experiments A & B** — `configs/experiment_a_baseline.yaml` (concurrency
  sweep), `experiment_b1_prompt_length.yaml`, `experiment_b2_output_length.yaml`,
  and `experiment_b3_mixed_lengths.yaml` (homogeneous vs mixed vs randomized
  length distributions at comparable mean length).
- **`inferbench experiment --config <file>`** — runs a study and emits a
  Markdown report plus a machine-readable JSON sidecar.

## Phase 4 — Experiment C: concurrency and saturation (complete)

Experiment C extends the concurrency sweep to the point where scaling stops,
which is the question a serving benchmark most needs to answer.

- **`configs/experiment_c_concurrency_sweep.yaml`** — varies `load.concurrency`
  from 1 to 32 (doubling: 1, 2, 4, 8, 16, 32), holding the workload fixed. The
  knee — the last concurrency at which throughput still improves meaningfully —
  is visible because the sweep runs well past it.
- **A synthetic timing model in config** — `backend.synthetic` declares the
  mock backend's analytic parameters (`decode_contention`, prefill rate, and so
  on) so a CPU-only study can demonstrate the *full* saturation signature:
  TPOT rises with in-flight requests, throughput flattens, and the E2E p95 tail
  grows. The section is mock-only, clearly namespaced, and every run that uses
  it is recorded as `data_source = "synthetic"` — the numbers are never
  presented as real hardware performance.
- **Honest reporting of the result** — the report states the observed
  flattening and tail growth as *descriptive* facts, names the apparent knee,
  and explicitly declines to attribute a cause. When the sweep shows no
  flattening (or no tail growth), it says so rather than manufacturing a story.

Verification: the full suite passes on CPU, including Phase 4 tests that check
the shipped Experiment C models contention, that its backend section is
identical across variants (contention is fixed, not the varied axis), and that
the saturation-knee detector finds the knee in a stalled sweep and reports none
in one that is still scaling.

## Definition of Done — status

| # | Criterion | Status |
|---|-----------|--------|
| 1 | End-to-end benchmark through vLLM | ⚠️ Pipeline complete and validated on the mock backend; the vLLM adapter is written to the supported API but has **not** been run against a live server (needs GPU) |
| 2 | Streaming latency metrics validated against controlled tests | ✅ Phase 1 — analytic ground-truth tests |
| 3 | Raw observations + reproducible metadata saved | ✅ Phase 1 |
| 4 | Workload configs compared fairly | ✅ Phase 2 — fairness gate refuses unfair comparisons |
| 5 | Reports communicate throughput/latency/failures/limits | ✅ Phase 1 |
| 6 | Tests pass in CPU-only environment | ✅ 293 tests, no GPU |
| 7 | Real GPU benchmark documented | ❌ **Pending — requires GPU host** |
| 8 | Setup instructions + limitations documented | ⏳ Phase 7 (methodology and limitations partially documented) |

### Known constraint

The development host has no discrete GPU (Intel integrated graphics only;
`nvidia-smi` present but no driver). Real vLLM serving and GPU telemetry cannot
be exercised here. The engine is therefore built and tested against a
deterministic mock backend, with the vLLM adapter written to the supported API
and marked as not yet validated against a live server.

## Future work

- SGLang as a second real backend, only after the vLLM adapter is proven on GPU.
- Prefix-cache and quantization experiments executed on real hardware.
- Real-dataset workloads where licensing permits.
