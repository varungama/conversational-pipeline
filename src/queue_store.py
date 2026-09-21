"""A minimal, durable, file-based work queue.

    pending/      work waiting to be picked up
    processing/   claimed by a worker, in flight
    done/         finished successfully, kept as a receipt
    dead_letter/  could not be processed at all, with a reason file alongside

Claiming a file is one `os.rename` into processing/ - atomic on the same
filesystem, so two workers racing for a file leave exactly one winner. No
lock server needed at this scale.

Deliberately cut: no visibility timeout. A real queue (SQS) redelivers a
crashed worker's claimed message automatically; a file left in processing/
here just sits until a human or sweeper looks (see NOTES.md).
"""

import json
import os
import pathlib
import shutil


class FileQueue:
    def __init__(self, root: pathlib.Path):
        self.root = root
        self.pending = root / "pending"
        self.processing = root / "processing"
        self.done = root / "done"
        self.dead_letter = root / "dead_letter"
        for directory in (self.pending, self.processing, self.done, self.dead_letter):
            directory.mkdir(parents=True, exist_ok=True)

    def enqueue(self, source_path: pathlib.Path, cost: int = 0) -> pathlib.Path:
        """Copies a conversation file into pending/, leaving the source
        untouched so re-running against the same input is safe. `cost` is
        the job's relative size - see `list_pending`."""
        target = self.pending / source_path.name
        shutil.copy2(source_path, target)
        self._write_cost(target, cost)
        return target

    def enqueue_task(self, name: str, payload: dict, cost: int = 0) -> pathlib.Path:
        """Writes a task file (e.g. one shard of a large conversation) into
        pending/. Written to a temp name and renamed, so a worker never
        sees a half-written file."""
        target = self.pending / name
        tmp = target.with_name(target.name + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, target)
        self._write_cost(target, cost)
        return target

    def list_pending(self):
        """Most expensive first (longest-processing-time-first), name as the
        tiebreak. Workers walk this list claiming what they can, so with N
        workers the big jobs start first and the small ones fill the gaps at
        the end, instead of one worker drawing a huge job last."""
        return sorted(self.pending.glob("*.json"), key=lambda p: (-self._read_cost(p), p.name))

    @staticmethod
    def _cost_path(path: pathlib.Path) -> pathlib.Path:
        return path.with_name(path.name + ".cost")  # not *.json, so never listed as work

    def _write_cost(self, path: pathlib.Path, cost: int):
        if cost:
            self._cost_path(path).write_text(str(int(cost)), encoding="utf-8")

    def _read_cost(self, path: pathlib.Path) -> int:
        try:
            return int(self._cost_path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return 0  # no sidecar: an unsized job sorts last, by name

    def claim(self, path: pathlib.Path):
        """Atomically moves a pending file into processing/. Returns the new
        path, or None if another worker claimed it first."""
        target = self.processing / path.name
        try:
            os.rename(path, target)
        except OSError:
            return None
        self._cost_path(path).unlink(missing_ok=True)
        return target

    def mark_done(self, path: pathlib.Path):
        os.replace(path, self.done / path.name)

    def mark_dead(self, path: pathlib.Path, reason: str):
        os.replace(path, self.dead_letter / path.name)
        reason_path = self.dead_letter / (path.stem + ".reason.txt")
        reason_path.write_text(reason, encoding="utf-8")
