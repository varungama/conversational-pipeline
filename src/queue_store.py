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
        """Copies a conversation file into pending/, leaving the source
        untouched so re-running against the same input is safe."""
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
