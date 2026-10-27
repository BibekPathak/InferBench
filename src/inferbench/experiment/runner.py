"""Executing a multi-run experiment.

The study runner orchestrates several runs against the same backend lifecycle.
Two behaviours matter for correctness:

1. **One backend, reused across variants.** The adapter is set up once, so
   warm-up work (CUDA graphs, connection pools) is not repeated per variant —
   otherwise later variants would appear faster for reasons unrelated to the
   variable under study. A variant that needs a *different* backend cannot be
   served by this reuse, so it is rejected up front.

2. **Failures are contained.** If one variant fails validation or errors out,
   the remaining variants still run, and the failure is recorded on the study
   result. An experiment that aborts on its third of eight arms wastes the
   first two and tells the user nothing.

The runner deliberately reuses :class:`~inferbench.runner.runner.Runner` for
each arm rather than reimplementing request execution, so a study and a single
run measure things identically.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from inferbench.errors import BackendError, InferBenchError
from inferbench.experiment.config import ExpandedRun, ExperimentConfig, expand_experiment
from inferbench.logging_utils import get_logger, log_event
from inferbench.runner.runner import Runner, RunResult
from inferbench.storage import ExperimentStore

__all__ = ["StudyResult", "StudyRunner"]

_log = get_logger("experiment.runner")


@dataclass(slots=True)
class StudyResult:
    """Outcome of executing every variant in an experiment."""

    experiment: str
    runs: list[RunResult] = field(default_factory=list)
    #: Variants that failed, mapped to a human-readable reason. Present so a
    #: partial study is reported as partial rather than silently thinner.
    failures: dict[str, str] = field(default_factory=dict)
    #: Which config path was deliberately varied.
    varies: str = ""

    @property
    def succeeded(self) -> int:
        return len(self.runs)

    @property
    def attempted(self) -> int:
        return len(self.runs) + len(self.failures)

    @property
    def is_complete(self) -> bool:
        return not self.failures

    @property
    def run_ids(self) -> list[str]:
        return [run.run_id for run in self.runs]

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment": self.experiment,
            "varies": self.varies,
            "complete": self.is_complete,
            "succeeded": self.succeeded,
            "attempted": self.attempted,
            "runs": [
                {
                    "run_id": run.run_id,
                    "run_name": run.metadata.run_name,
                    "run_name_variant": run.metadata.run_name.split("--", 1)[-1],
                    "incomplete": run.metadata.incomplete,
                }
                for run in self.runs
            ],
            "failures": dict(self.failures),
        }


class StudyRunner:
    """Executes every variant of an experiment against one backend."""

    def __init__(
        self,
        experiment: ExperimentConfig,
        backend: Any,
        store: ExperimentStore,
        *,
        environment: dict[str, Any] | None = None,
    ) -> None:
        self.experiment = experiment
        self.backend = backend
        self.store = store
        self.environment = environment or {}
        self._validate_single_backend()

    def _validate_single_backend(self) -> None:
        """Refuse an experiment whose variants need different backends.

        The backend is set up once for the whole study, so differing backend
        configuration between variants would produce misleading measurements.
        """
        expanded = expand_experiment(self.experiment)
        first = expanded[0].config.backend
        for run in expanded[1:]:
            other = run.config.backend
            if other != first:
                raise BackendError(
                    f"Experiment {self.experiment.name!r} variant {run.variant.name!r} "
                    "configures a different backend than the first variant. A study "
                    "runs against a single backend; split it into two experiments."
                )

    async def run(self) -> StudyResult:
        """Execute all variants, recording failures without aborting the study.

        Raises:
            BackendError: The backend could not be set up at all, which means no
                variant can run and the study is meaningless.
        """
        result = StudyResult(experiment=self.experiment.name, varies=self.experiment.varies)
        expanded = expand_experiment(self.experiment)

        log_event(
            _log,
            logging.INFO,
            "study starting",
            experiment=self.experiment.name,
            variants=len(expanded),
            varies=self.experiment.varies,
        )

        await self.backend.setup()
        try:
            for run in expanded:
                run_id = await self._run_variant(run, result)
                if run_id:
                    log_event(
                        _log,
                        20,
                        "variant complete",
                        experiment=self.experiment.name,
                        variant=run.variant.name,
                        run_id=run_id,
                    )
        finally:
            await self.backend.teardown()

        log_event(
            _log,
            logging.INFO,
            "study complete",
            experiment=self.experiment.name,
            succeeded=result.succeeded,
            failed=len(result.failures),
        )
        return result

    async def _run_variant(self, run: ExpandedRun, result: StudyResult) -> str | None:
        """Run one variant, returning its run id or ``None`` on failure.

        The variant is identified downstream by the run name
        (``<experiment>--<variant>``), which the expander sets deterministically.
        """
        runner = Runner(run.config, self.backend, self.store, environment=self.environment)
        try:
            run_result = await runner.run()
        except InferBenchError as exc:
            # An expected, per-variant failure: record and continue.
            result.failures[run.variant.name] = f"{type(exc).__name__}: {exc}"
            log_event(
                _log,
                logging.WARNING,
                "variant failed",
                experiment=self.experiment.name,
                variant=run.variant.name,
                error=str(exc),
            )
            return None
        except Exception as exc:  # unexpected: still contain it
            result.failures[run.variant.name] = f"unexpected {type(exc).__name__}: {exc}"
            log_event(
                _log,
                logging.ERROR,
                "variant failed unexpectedly",
                experiment=self.experiment.name,
                variant=run.variant.name,
                error=str(exc),
            )
            return None

        result.runs.append(run_result)
        return run_result.run_id
