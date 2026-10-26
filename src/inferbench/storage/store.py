"""Experiment storage.

Layout under the configured output directory::

    runs/
      inferbench.sqlite3        # metadata + aggregated summaries (queryable)
      <run_id>/
        metadata.json           # full run metadata, human-readable
        raw.jsonl               # one JSON object per request (the raw evidence)
        metrics.json            # aggregated summary
        summary.csv             # flat summary for spreadsheets
        environment.json        # environment fingerprint at run time

Rationale: SQLite is the index (fast "which runs exist, what were the headline
numbers"), while the per-run directory holds the raw, append-only evidence.
The raw JSONL is the source of truth — every metric can be recomputed from it,
which is what makes a stored result auditable rather than merely reported.
"""

from __future__ import annotations

import contextlib
import csv
import json
import sqlite3
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from inferbench.errors import RunNotFoundError, StorageError
from inferbench.logging_utils import get_logger
from inferbench.types import RequestRecord, RunMetadata

__all__ = ["ExperimentStore", "SCHEMA_VERSION"]

_log = get_logger("storage")

#: Bumped when the SQLite schema changes incompatibly; recorded so old stores
#: are detected rather than silently misread.
SCHEMA_VERSION = 1

_DB_FILENAME = "inferbench.sqlite3"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_info (
    version INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    run_id            TEXT PRIMARY KEY,
    run_name          TEXT NOT NULL,
    experiment        TEXT NOT NULL,
    created_at        REAL NOT NULL,
    started_at        REAL NOT NULL,
    finished_at       REAL,
    backend_kind      TEXT NOT NULL,
    backend_version   TEXT,
    model             TEXT NOT NULL,
    data_source       TEXT NOT NULL,
    tool_version      TEXT NOT NULL,
    incomplete        INTEGER NOT NULL DEFAULT 0,
    incomplete_reason TEXT,
    metadata_json     TEXT NOT NULL,
    summary_json      TEXT,
    request_count     INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS run_requests (
    run_id        TEXT NOT NULL,
    request_id    TEXT NOT NULL,
    status        TEXT NOT NULL,
    is_warmup     INTEGER NOT NULL,
    repetition    INTEGER NOT NULL,
    ttft_s        REAL,
    e2e_s         REAL NOT NULL,
    output_tokens INTEGER NOT NULL,
    prompt_tokens INTEGER,
    class_label   TEXT NOT NULL,
    PRIMARY KEY (run_id, repetition, request_id),
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_run_requests_run ON run_requests(run_id);
"""


class ExperimentStore:
    """SQLite-backed index plus per-run artifact directories.

    The store is usable as a context manager; the underlying connection is
    closed on exit. Connections are per-instance (SQLite handles are not
    thread-safe), which keeps the concurrency story simple: the runner writes
    from one task.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser()
        try:
            self.root.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise StorageError(f"Could not create output directory {self.root}: {exc}") from exc

        self._db_path = self.root / _DB_FILENAME
        try:
            self._conn = sqlite3.connect(self._db_path)
        except sqlite3.Error as exc:
            raise StorageError(f"Could not open database {self._db_path}: {exc}") from exc

        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._init_schema()

    # --- lifecycle --------------------------------------------------------

    def close(self) -> None:
        with contextlib.suppress(sqlite3.Error):
            self._conn.close()

    def __enter__(self) -> ExperimentStore:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _init_schema(self) -> None:
        try:
            with self._conn:
                self._conn.executescript(_SCHEMA)
                row = self._conn.execute("SELECT version FROM schema_info").fetchone()
                if row is None:
                    self._conn.execute(
                        "INSERT INTO schema_info (version) VALUES (?)", (SCHEMA_VERSION,)
                    )
                elif int(row["version"]) != SCHEMA_VERSION:
                    raise StorageError(
                        f"Database {self._db_path} has schema version {row['version']}, "
                        f"expected {SCHEMA_VERSION}. Use a fresh output directory."
                    )
        except sqlite3.Error as exc:
            raise StorageError(f"Could not initialize database {self._db_path}: {exc}") from exc

    # --- paths ------------------------------------------------------------

    def run_dir(self, run_id: str) -> Path:
        """Directory holding a run's artifacts (not created here)."""
        return self.root / run_id

    def _require_run_dir(self, run_id: str) -> Path:
        path = self.run_dir(run_id)
        if not path.is_dir():
            raise RunNotFoundError(f"Run directory not found for {run_id!r}: {path}")
        return path

    # --- writing ----------------------------------------------------------

    def save_metadata(self, metadata: RunMetadata, environment: dict[str, Any]) -> None:
        """Register a run and write its metadata/environment artifacts."""
        run_dir = self.run_dir(metadata.run_id)
        try:
            run_dir.mkdir(parents=True, exist_ok=True)
            _write_json(run_dir / "metadata.json", metadata.to_dict())
            _write_json(run_dir / "environment.json", environment)
        except OSError as exc:
            raise StorageError(f"Could not write metadata for {metadata.run_id}: {exc}") from exc

        try:
            with self._conn:
                self._conn.execute(
                    """
                    INSERT INTO runs (
                        run_id, run_name, experiment, created_at, started_at, finished_at,
                        backend_kind, backend_version, model, data_source, tool_version,
                        incomplete, incomplete_reason, metadata_json, request_count
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(run_id) DO UPDATE SET
                        finished_at=excluded.finished_at,
                        incomplete=excluded.incomplete,
                        incomplete_reason=excluded.incomplete_reason,
                        metadata_json=excluded.metadata_json
                    """,
                    (
                        metadata.run_id,
                        metadata.run_name,
                        metadata.experiment,
                        metadata.created_at,
                        metadata.started_at,
                        metadata.finished_at,
                        metadata.backend_kind,
                        metadata.backend_version,
                        metadata.model,
                        metadata.data_source,
                        metadata.tool_version,
                        int(metadata.incomplete),
                        metadata.incomplete_reason,
                        json.dumps(metadata.to_dict(), default=str),
                        0,
                    ),
                )
        except sqlite3.Error as exc:
            raise StorageError(f"Could not register run {metadata.run_id}: {exc}") from exc

    def save_raw_records(self, run_id: str, records: Sequence[RequestRecord]) -> Path:
        """Write raw per-request observations to JSONL and index them.

        Returns the JSONL path. The JSONL is rewritten wholesale on each call so
        a partial write from a crashed run cannot leave a stale mixture.
        """
        run_dir = self.run_dir(run_id)
        try:
            run_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise StorageError(f"Could not create run directory {run_dir}: {exc}") from exc

        jsonl_path = run_dir / "raw.jsonl"
        try:
            with jsonl_path.open("w", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record.to_dict(), default=str))
                    handle.write("\n")
        except OSError as exc:
            raise StorageError(f"Could not write raw records for {run_id}: {exc}") from exc

        try:
            with self._conn:
                self._conn.execute("DELETE FROM run_requests WHERE run_id = ?", (run_id,))
                self._conn.executemany(
                    """
                    INSERT INTO run_requests (
                        run_id, request_id, status, is_warmup, repetition,
                        ttft_s, e2e_s, output_tokens, prompt_tokens, class_label
                    ) VALUES (?,?,?,?,?,?,?,?,?,?)
                    """,
                    [
                        (
                            run_id,
                            record.request_id,
                            record.status.value,
                            int(record.is_warmup),
                            record.repetition,
                            record.ttft_s,
                            record.e2e_s,
                            record.output_tokens,
                            record.prompt_tokens,
                            record.class_label,
                        )
                        for record in records
                    ],
                )
                self._conn.execute(
                    "UPDATE runs SET request_count = ? WHERE run_id = ?",
                    (len(records), run_id),
                )
        except sqlite3.Error as exc:
            raise StorageError(f"Could not index raw records for {run_id}: {exc}") from exc

        return jsonl_path

    def save_summary(self, run_id: str, summary: dict[str, Any]) -> None:
        """Persist an aggregated summary to SQLite, JSON, and CSV."""
        run_dir = self.run_dir(run_id)
        try:
            run_dir.mkdir(parents=True, exist_ok=True)
            _write_json(run_dir / "metrics.json", summary)
            self._write_summary_csv(run_dir / "summary.csv", summary)
        except OSError as exc:
            raise StorageError(f"Could not write summary for {run_id}: {exc}") from exc

        try:
            with self._conn:
                self._conn.execute(
                    "UPDATE runs SET summary_json = ? WHERE run_id = ?",
                    (json.dumps(summary, default=str), run_id),
                )
        except sqlite3.Error as exc:
            raise StorageError(f"Could not store summary for {run_id}: {exc}") from exc

    @staticmethod
    def _write_summary_csv(path: Path, summary: dict[str, Any]) -> None:
        """Flatten the nested summary into ``metric,value`` rows."""
        rows = list(_flatten("", summary))
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["metric", "value"])
            writer.writerows(rows)

    # --- reading ----------------------------------------------------------

    def list_runs(self) -> list[dict[str, Any]]:
        """Return run rows, newest first."""
        try:
            cursor = self._conn.execute(
                """
                SELECT run_id, run_name, experiment, created_at, backend_kind, model,
                       data_source, tool_version, incomplete, request_count
                FROM runs ORDER BY created_at DESC
                """
            )
            return [dict(row) for row in cursor.fetchall()]
        except sqlite3.Error as exc:
            raise StorageError(f"Could not list runs: {exc}") from exc

    def get_metadata(self, run_id: str) -> RunMetadata:
        """Load a run's full metadata.

        Raises:
            RunNotFoundError: The run is unknown, or has no metadata artifact.
        """
        try:
            row = self._conn.execute(
                "SELECT metadata_json FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        except sqlite3.Error as exc:
            raise StorageError(f"Could not load run {run_id}: {exc}") from exc

        if row is None:
            raise RunNotFoundError(f"No such run: {run_id!r}")

        raw = json.loads(row["metadata_json"])
        return RunMetadata(
            run_id=raw["run_id"],
            run_name=raw["run_name"],
            experiment=raw["experiment"],
            created_at=raw["created_at"],
            started_at=raw["started_at"],
            finished_at=raw["finished_at"],
            config=raw["config"],
            environment=raw["environment"],
            backend_kind=raw["backend_kind"],
            backend_version=raw["backend_version"],
            model=raw["model"],
            data_source=raw["data_source"],
            tool_version=raw["tool_version"],
            notes=raw.get("notes", ""),
            model_revision=raw.get("model_revision"),
            backend_details=raw.get("backend_details", {}),
            incomplete=raw.get("incomplete", False),
            incomplete_reason=raw.get("incomplete_reason"),
        )

    def get_summary(self, run_id: str) -> dict[str, Any]:
        """Load a run's aggregated summary.

        Raises:
            RunNotFoundError: The run is unknown or has no summary yet.
        """
        try:
            row = self._conn.execute(
                "SELECT summary_json FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        except sqlite3.Error as exc:
            raise StorageError(f"Could not load summary for {run_id}: {exc}") from exc

        if row is None:
            raise RunNotFoundError(f"No such run: {run_id!r}")
        if row["summary_json"] is None:
            raise RunNotFoundError(f"Run {run_id!r} has no summary (run may be incomplete)")
        result: dict[str, Any] = json.loads(row["summary_json"])
        return result

    def load_raw_records(self, run_id: str) -> list[RequestRecord]:
        """Reconstruct raw records from the JSONL artifact.

        Reading from JSONL (rather than the DB) is intentional: the raw file is
        the durable evidence, and this proves it is complete enough to
        recompute every metric.
        """
        path = self._require_run_dir(run_id) / "raw.jsonl"
        if not path.is_file():
            raise RunNotFoundError(f"Raw records missing for {run_id!r}: {path}")

        records: list[RequestRecord] = []
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    records.append(_record_from_dict(data))
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise StorageError(f"Could not read raw records for {run_id}: {exc}") from exc
        return records

    def has_run(self, run_id: str) -> bool:
        """Whether a run id is registered and has a directory."""
        try:
            row = self._conn.execute("SELECT 1 FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        except sqlite3.Error:
            return False
        return row is not None and self.run_dir(run_id).is_dir()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _flatten(prefix: str, value: Any) -> Iterable[tuple[str, Any]]:
    """Yield dotted ``key,value`` rows from a nested dict.

    Lists are summarized by length rather than expanded, since a summary CSV is
    meant to be one screenful, not a data dump.
    """
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            yield from _flatten(child, item)
    elif isinstance(value, list):
        yield (prefix, f"<list len={len(value)}>")
    else:
        yield (prefix, value)


def _record_from_dict(data: dict[str, Any]) -> RequestRecord:
    """Rebuild a RequestRecord from its JSON representation."""
    from inferbench.types import RequestStatus

    return RequestRecord(
        request_id=str(data["request_id"]),
        status=RequestStatus(data["status"]),
        prompt_tokens=data.get("prompt_tokens"),
        output_tokens=int(data["output_tokens"]),
        requested_output_tokens=int(data["requested_output_tokens"]),
        ttft_s=data.get("ttft_s"),
        e2e_s=float(data["e2e_s"]),
        token_times_s=tuple(float(t) for t in data.get("token_times_s", [])),
        class_label=str(data.get("class_label", "unknown")),
        repetition=int(data.get("repetition", 0)),
        is_warmup=bool(data.get("is_warmup", False)),
        started_at=float(data["started_at"]),
        error=data.get("error"),
        finish_reason=data.get("finish_reason"),
        backend_request_id=data.get("backend_request_id"),
    )
