# InferBench Roadmap

Phases are ordered by dependency. Each phase ends green on lint, type checks,
and tests before the next begins. Real GPU measurements are deferred until a
GPU host is available; nothing is ever fabricated to fill that gap.

| Phase | Scope | Status |
|-------|-------|--------|
| 0 | Scaffold, environment doctor, CI | ✅ Complete |
| 1 | Vertical slice: run → metrics → store → report (mock + vLLM adapter) | ⏳ Planned |
| 2 | Comparison, validation, fairness gating | ⏳ Planned |
| 3 | Experiments A & B: baseline + workload sensitivity | ⏳ Planned |
| 4 | Experiment C: concurrency & saturation | ⏳ Planned |
| 5 | Experiments D & E: prefix caching & quantization (GPU-gated execution) | ⏳ Planned |
| 6 | Experiment F: InferSched simulation adapter | ⏳ Planned |
| 7 | Deployment, observability, docs | ⏳ Planned |
| 8 | Roadmap, illustrative report, release | ⏳ Planned |

## Phase 0 — Scaffold, environment doctor, CI

Delivered:

- Installable `inferbench` package with typed configuration and structured logging.
- `inferbench doctor` — reports Python, CPU, memory, GPU/CUDA (graceful when
  absent), disk, and tracked package versions. Supports `--json`.
- CLI surface registered for every spec command; unimplemented commands state
  their target phase rather than failing opaquely.
- Lint (ruff), type checking (mypy strict), and test (pytest) configuration.
- CI workflow running all three on push and pull request.

## Definition of Done — status

| # | Criterion | Status |
|---|-----------|--------|
| 1 | End-to-end benchmark through vLLM | ⏳ Pending (Phase 1 adapter; GPU needed to execute) |
| 2 | Streaming latency metrics validated against controlled tests | ⏳ Phase 1 |
| 3 | Raw observations + reproducible metadata saved | ⏳ Phase 1 |
| 4 | Workload configs compared fairly | ⏳ Phase 3 |
| 5 | Reports communicate throughput/latency/failures/limits | ⏳ Phase 1+ |
| 6 | Tests pass in CPU-only environment | ⏳ Phase 1 (Phase 0 tests pass CPU-only) |
| 7 | Real GPU benchmark documented | ❌ **Pending — requires GPU host** |
| 8 | Setup instructions + limitations documented | ⏳ Phase 7 |

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
