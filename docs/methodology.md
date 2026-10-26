# Benchmark Methodology

This document defines the metrics InferBench computes, how they are aggregated,
and the rules that make comparisons fair. It is a living document: sections
marked *Phase 1+* will be completed as the engine lands.

## Metric definitions

All latencies are measured in **seconds**, from timestamps captured in the
benchmark client.

| Metric | Definition |
|--------|------------|
| **TTFT** (time to first token) | Time from request dispatch to the first streamed content token. |
| **Inter-token latency** | Time between consecutive streamed tokens. |
| **TPOT** (time per output token) | Decode-phase average: (end-to-end − TTFT) / (output tokens − 1). Undefined for single-token outputs. |
| **End-to-end latency** | Time from request dispatch to the final token. |
| **Request throughput** | Successfully completed requests / measurement wall-clock duration. |
| **Token throughput** | Output tokens from successful requests / measurement wall-clock duration. |

## Aggregation rules

Percentiles use the **nearest-rank** method on the sorted sample (Documented
here so results are reproducible): for a sorted sample of size `n` and
percentile `p`, the reported value is the element at index `ceil(p/100 × n) − 1`
(clamped to `[0, n−1]`).

Rules that exist to prevent misleading numbers:

- **Never report a mean without the distribution.** TTFT and latency
  distributions are right-skewed; a mean alone hides the tail that matters.
- **Offered load ≠ completed throughput.** Throughput counts only successfully
  completed requests. Failures reduce throughput rather than being excluded.
- **Warm-up is excluded.** Requests issued during warm-up are recorded but
  never included in measured statistics.
- **Failed requests are counted, not dropped.** Failure rate is reported
  alongside every latency and throughput figure.

## Warm-up and measurement

Each repetition runs `warmup_requests` requests that are discarded, then
`num_requests` measured requests. Warm-up exists to stabilise CUDA graph
capture, KV-cache allocation, connection pooling, and JIT compilation so the
measured window reflects steady-state behaviour.

*Phase 1+: exact measurement-window bookkeeping.*

## Fair-comparison rules

Two runs may be compared only when:

1. They use the same prompt-length and output-length distributions (or the
   difference is the explicit subject of the experiment).
2. Both exclude warm-up and use the same measurement protocol.
3. Both report the same hardware and backend configuration, or the difference
   is documented.

A comparison that would violate these rules is rejected with an explanation
rather than silently produced.

## Sources of nondeterminism

Documented explicitly so results are interpreted correctly:

- **Prompt content** is seeded and deterministic; **sampling by the model** is
  not, unless temperature is 0 and the backend guarantees determinism.
- **Tokenisation** depends on the tokenizer; workloads target token counts but
  the achieved counts are recorded per run.
- **Concurrency and scheduling** in the serving engine are not controlled by the
  client, so run-to-run variance is expected and reported via repetitions.

*Phase 1+: measured variance and confidence intervals.*

## Limitations

- Queueing, prefill, and decode timings are separated **only** when the backend
  genuinely exposes the required measurements. They are never inferred from
  end-to-end timings.
- GPU utilization and memory are reported only when real telemetry is
  available. Absent telemetry is reported as unavailable, never estimated.
- Simulated scheduling results (Experiment F) describe a **simulator**, not a
  live engine's internal scheduler, and are labelled as such.

## Phase status

Metrics and rules above are implemented and tested as of Phase 1. Concrete
measured results will be appended once Phase 1 executes against a real backend:
the pipeline is validated end to end on the deterministic mock backend, and the
vLLM adapter is written to the supported API but has not yet been exercised
against a live server. Real GPU results follow once a GPU host is available.

Charts and reports are generated from stored runs; see any
`runs/<run_id>/report.md` for a worked example (labelled synthetic when the mock
backend produced it).
