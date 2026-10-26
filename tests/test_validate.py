"""Tests for run validation.

Validation exists to answer "can these numbers be trusted?" — so the tests are
mostly *negative*: they deliberately corrupt one aspect of a stored run and
assert that the right finding is produced with the right severity. A validator
that only ever says "valid" is worthless.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from inferbench.analysis import Severity, format_report, validate_run
from inferbench.errors import RunNotFoundError
from inferbench.storage import ExperimentStore
from inferbench.types import RequestRecord, RequestStatus, RunMetadata

StoreRun = Callable[..., str]
BuildRecords = Callable[..., list[RequestRecord]]
BuildMetadata = Callable[..., RunMetadata]


def codes(report: object) -> set[str]:
    """All finding codes in a report."""
    return {f.code for f in report.findings}  # type: ignore[attr-defined]


def code_at(report: object, severity: Severity) -> set[str]:
    return {f.code for f in report.findings if f.severity is severity}  # type: ignore[attr-defined]


class TestValidRun:
    def test_complete_run_has_no_errors(
        self, tmp_path: Path, stored_run: StoreRun, make_records: BuildRecords
    ) -> None:
        with ExperimentStore(tmp_path) as store:
            run_id = stored_run(store)
            report = validate_run(store, run_id)

        assert report.is_valid
        assert report.errors == ()
        # Synthetic provenance must always be surfaced.
        assert "synthetic_data" in codes(report)
        assert report.run_id == run_id

    def test_reports_are_serializable(
        self, tmp_path: Path, stored_run: StoreRun, make_records: BuildRecords
    ) -> None:
        with ExperimentStore(tmp_path) as store:
            run_id = stored_run(store)
            payload = validate_run(store, run_id).to_dict()

        assert payload["valid"] is True
        assert json.loads(json.dumps(payload))["run_id"] == run_id

    def test_text_report_renders(
        self, tmp_path: Path, stored_run: StoreRun, make_records: BuildRecords
    ) -> None:
        with ExperimentStore(tmp_path) as store:
            run_id = stored_run(store)
            text = format_report(validate_run(store, run_id))

        assert "VALID" in text
        assert run_id in text


class TestUnknownRun:
    def test_unknown_run_raises(self, tmp_path: Path) -> None:
        with ExperimentStore(tmp_path) as store, pytest.raises(RunNotFoundError):
            validate_run(store, "ghost")


class TestMissingArtifacts:
    def test_missing_raw_jsonl_is_an_error(
        self, tmp_path: Path, stored_run: StoreRun, make_records: BuildRecords
    ) -> None:
        with ExperimentStore(tmp_path) as store:
            run_id = stored_run(store)
            (store.run_dir(run_id) / "raw.jsonl").unlink()
            report = validate_run(store, run_id)

        assert not report.is_valid
        assert "missing_artifact" in code_at(report, Severity.error)

    def test_missing_summary_is_an_error(
        self, tmp_path: Path, stored_run: StoreRun, make_records: BuildRecords
    ) -> None:
        with ExperimentStore(tmp_path) as store:
            run_id = stored_run(store, write_summary=False)
            report = validate_run(store, run_id)

        assert not report.is_valid
        assert "unreadable_summary" in code_at(report, Severity.error)

    def test_all_missing_artifacts_reported(self, tmp_path: Path, stored_run: StoreRun) -> None:
        with ExperimentStore(tmp_path) as store:
            run_id = stored_run(store)
            run_dir = store.run_dir(run_id)
            for artifact in ("raw.jsonl", "metrics.json", "summary.csv", "environment.json"):
                (run_dir / artifact).unlink()
            report = validate_run(store, run_id)

        # Each missing artifact is its own finding, not a single vague error.
        missing = [f for f in report.errors if f.code == "missing_artifact"]
        assert len(missing) == 4
        reported = " ".join(f.message for f in missing)
        for artifact in ("raw.jsonl", "metrics.json", "summary.csv", "environment.json"):
            assert artifact in reported


class TestMetadataIntegrity:
    def test_incomplete_run_is_warned(
        self,
        tmp_path: Path,
        stored_run: StoreRun,
        make_metadata: BuildMetadata,
        make_records: BuildRecords,
    ) -> None:
        with ExperimentStore(tmp_path) as store:
            meta = make_metadata("run-1", incomplete=True)
            run_id = stored_run(store, metadata=meta)
            report = validate_run(store, run_id)

        assert "incomplete_run" in code_at(report, Severity.warning)
        # Incompleteness is honestly recorded, so it is not an *error*.
        assert report.is_valid

    def test_no_gpu_is_reported_as_info(
        self, tmp_path: Path, stored_run: StoreRun, make_records: BuildRecords
    ) -> None:
        with ExperimentStore(tmp_path) as store:
            run_id = stored_run(store)
            report = validate_run(store, run_id)

        assert "no_gpu" in code_at(report, Severity.info)

    def test_empty_config_is_an_error(
        self,
        tmp_path: Path,
        stored_run: StoreRun,
        make_metadata: BuildMetadata,
        make_records: BuildRecords,
    ) -> None:
        with ExperimentStore(tmp_path) as store:
            meta = make_metadata("run-1")
            meta.config = {}
            run_id = stored_run(store, metadata=meta)
            report = validate_run(store, run_id)

        assert "empty_config" in code_at(report, Severity.error)


class TestDataConsistency:
    def test_no_measured_requests_is_an_error(
        self, tmp_path: Path, stored_run: StoreRun, make_records: BuildRecords
    ) -> None:
        from inferbench.types import RequestRecord

        with ExperimentStore(tmp_path) as store:
            warmup = RequestRecord(
                request_id="w0",
                status=RequestStatus.success,
                prompt_tokens=10,
                output_tokens=3,
                requested_output_tokens=3,
                ttft_s=0.01,
                e2e_s=0.05,
                token_times_s=(0.01, 0.03, 0.05),
                class_label="fixed",
                repetition=-1,
                is_warmup=True,
                started_at=0.0,
            )
            run_id = stored_run(store, records=[warmup])
            report = validate_run(store, run_id)

        assert "no_measured_requests" in code_at(report, Severity.error)

    def test_duplicate_request_ids_are_an_error(self, tmp_path: Path, stored_run: StoreRun) -> None:
        with ExperimentStore(tmp_path) as store:
            run_id = stored_run(store)
            # The store enforces uniqueness, so inject the duplicate into the
            # raw file directly — which is the durable evidence, and which a
            # user could hand-edit.
            raw = store.run_dir(run_id) / "raw.jsonl"
            lines = raw.read_text().strip().splitlines()
            raw.write_text("\n".join([*lines, lines[0]]) + "\n", encoding="utf-8")
            report = validate_run(store, run_id)

        assert "duplicate_requests" in code_at(report, Severity.error)

    def test_negative_timing_is_an_error(
        self, tmp_path: Path, stored_run: StoreRun, make_records: BuildRecords
    ) -> None:
        with ExperimentStore(tmp_path) as store:
            records = make_records(count=2)
            records[0].e2e_s = -1.0
            run_id = stored_run(store, records=records)
            report = validate_run(store, run_id)

        assert "negative_timing" in code_at(report, Severity.error)

    def test_ttft_after_e2e_is_an_error(
        self, tmp_path: Path, stored_run: StoreRun, make_records: BuildRecords
    ) -> None:
        """TTFT cannot occur after the request completed."""
        with ExperimentStore(tmp_path) as store:
            records = make_records(count=2)
            records[0].ttft_s = records[0].e2e_s + 1.0
            run_id = stored_run(store, records=records)
            report = validate_run(store, run_id)

        assert "inconsistent_timing" in code_at(report, Severity.error)

    def test_non_monotonic_tokens_is_a_warning(
        self, tmp_path: Path, stored_run: StoreRun, make_records: BuildRecords
    ) -> None:
        with ExperimentStore(tmp_path) as store:
            records = make_records(count=2)
            records[0].token_times_s = (0.05, 0.03, 0.07, 0.09, 0.11)
            run_id = stored_run(store, records=records)
            report = validate_run(store, run_id)

        assert "non_monotonic_tokens" in code_at(report, Severity.warning)
        assert report.is_valid  # warn, not invalidate

    def test_token_count_mismatch_is_a_warning(
        self, tmp_path: Path, stored_run: StoreRun, make_records: BuildRecords
    ) -> None:
        with ExperimentStore(tmp_path) as store:
            records = make_records(count=2)
            records[0].output_tokens = 99  # does not match len(token_times_s)
            run_id = stored_run(store, records=records)
            report = validate_run(store, run_id)

        assert "token_count_mismatch" in code_at(report, Severity.warning)

    def test_zero_token_success_is_a_warning(
        self, tmp_path: Path, stored_run: StoreRun, make_records: BuildRecords
    ) -> None:
        with ExperimentStore(tmp_path) as store:
            records = make_records(count=2)
            records[0].output_tokens = 0
            records[0].token_times_s = ()
            records[0].ttft_s = None
            run_id = stored_run(store, records=records)
            report = validate_run(store, run_id)

        assert "empty_success" in code_at(report, Severity.warning)


class TestSummaryAgreement:
    """A summary that disagrees with the raw evidence is not auditable."""

    def test_success_count_mismatch_is_an_error(
        self, tmp_path: Path, stored_run: StoreRun, make_records: BuildRecords
    ) -> None:
        with ExperimentStore(tmp_path) as store:
            run_id = stored_run(store)
            summary = store.get_summary(run_id)
            summary["throughput"]["successful_requests"] = 9999
            store.save_summary(run_id, summary)
            report = validate_run(store, run_id)

        assert "summary_mismatch" in code_at(report, Severity.error)

    def test_output_token_mismatch_is_an_error(
        self, tmp_path: Path, stored_run: StoreRun, make_records: BuildRecords
    ) -> None:
        with ExperimentStore(tmp_path) as store:
            run_id = stored_run(store)
            summary = store.get_summary(run_id)
            summary["throughput"]["output_tokens"] = 123456
            store.save_summary(run_id, summary)
            report = validate_run(store, run_id)

        assert "summary_mismatch" in code_at(report, Severity.error)

    def test_percentile_mismatch_is_an_error(
        self, tmp_path: Path, stored_run: StoreRun, make_records: BuildRecords
    ) -> None:
        with ExperimentStore(tmp_path) as store:
            run_id = stored_run(store)
            summary = store.get_summary(run_id)
            summary["latency"]["ttft"]["p50"] = 12345.0
            store.save_summary(run_id, summary)
            report = validate_run(store, run_id)

        assert "summary_mismatch" in code_at(report, Severity.error)

    def test_nonpositive_wall_time_is_an_error(
        self, tmp_path: Path, stored_run: StoreRun, make_records: BuildRecords
    ) -> None:
        with ExperimentStore(tmp_path) as store:
            run_id = stored_run(store)
            summary = store.get_summary(run_id)
            summary["throughput"]["wall_time_s"] = 0.0
            store.save_summary(run_id, summary)
            report = validate_run(store, run_id)

        assert "bad_wall_time" in code_at(report, Severity.error)

    def test_missing_provenance_is_a_warning(
        self, tmp_path: Path, stored_run: StoreRun, make_records: BuildRecords
    ) -> None:
        with ExperimentStore(tmp_path) as store:
            run_id = stored_run(store)
            summary = store.get_summary(run_id)
            summary.pop("data_source", None)
            store.save_summary(run_id, summary)
            report = validate_run(store, run_id)

        assert "summary_no_provenance" in code_at(report, Severity.warning)

    def test_failed_requests_do_not_break_agreement(
        self, tmp_path: Path, stored_run: StoreRun, make_records: BuildRecords
    ) -> None:
        """Mixed success/failure runs must still validate cleanly."""
        with ExperimentStore(tmp_path) as store:
            records = make_records(count=4, statuses={1: RequestStatus.error})
            run_id = stored_run(store, records=records)
            report = validate_run(store, run_id)

        assert report.is_valid, [f.message for f in report.errors]
