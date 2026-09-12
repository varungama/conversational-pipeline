"""A minimal, durable, file-based work queue with four directories:

    pending/      work waiting to be picked up
    processing/   claimed by a worker, in flight
    done/         finished successfully, kept as a receipt
    dead_letter/  could not be processed, with a reason file alongside

Claiming is a single `os.rename` into processing/, atomic on the same
filesystem: exactly one worker wins a race, the other gets OSError and
moves on. No lock server needed at this scale.

Known gap (see NOTES.md): unlike a real queue with visibility timeouts, a
worker that crashes mid-claim leaves its file stuck in processing/ until
a human or sweeper looks.
"""

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

    def enqueue(self, source_path: pathlib.Path) -> pathlib.Path:
        """Copies a conversation file into pending/. The source file is left
        untouched, so re-running against the same input directory is safe."""
        target = self.pending / source_path.name
        shutil.copy2(source_path, target)
        return target

    def list_pending(self):
        return sorted(self.pending.glob("*.json"))

    def claim(self, path: pathlib.Path):
        """Atomically moves a pending file into processing/. Returns the new
        path, or None if another worker claimed it first."""
        target = self.processing / path.name
        try:
            os.rename(path, target)
            return target
        except OSError:
            return None

    def mark_done(self, path: pathlib.Path):
        os.replace(path, self.done / path.name)

    def mark_dead(self, path: pathlib.Path, reason: str):
        os.replace(path, self.dead_letter / path.name)
        reason_path = self.dead_letter / (path.stem + ".reason.txt")
        reason_path.write_text(reason, encoding="utf-8")
