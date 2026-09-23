"""Per-stage wall-clock accounting: accumulates named stage durations for the end-of-run timing breakdown.
One StageTimer lives per process; `snapshot`/`merge` carry totals across the mp.Pipe from --workers children."""

import contextlib
import logging
import time


class StageTimer:
    def __init__(self):
        self._totals: dict[str, float] = {}
        self._counts: dict[str, int] = {}

    @contextlib.contextmanager
    def time(self, stage: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.add(stage, time.perf_counter() - start)

    def add(self, stage: str, seconds: float, count: int = 1):
        self._totals[stage] = self._totals.get(stage, 0.0) + seconds
        self._counts[stage] = self._counts.get(stage, 0) + count

    def snapshot(self) -> dict:
        """Picklable {stage: {"seconds": ..., "count": ...}}, sendable over mp.Pipe to the parent process."""
        return {stage: {"seconds": self._totals[stage], "count": self._counts[stage]} for stage in self._totals}

    def merge(self, snapshot: dict):
        for stage, data in snapshot.items():
            self.add(stage, data["seconds"], data["count"])

    def log_summary(self, logger: logging.Logger, total_elapsed: float | None = None):
        """Logs one line per stage, longest total first; this is the run's final log output."""
        if not self._totals:
            return
        ordered = sorted(self._totals.items(), key=lambda kv: -kv[1])
        width = max(len(stage) for stage, _ in ordered)
        logger.info("Per-stage timing breakdown:")
        for stage, seconds in ordered:
            count = self._counts[stage]
            avg = seconds / count if count else 0.0
            pct = f" ({100 * seconds / total_elapsed:5.1f}% of wall-clock)" if total_elapsed else ""
            logger.info("  %-*s %9.2fs total  %6d call(s)  %8.4fs avg%s",
                        width, stage, seconds, count, avg, pct)
