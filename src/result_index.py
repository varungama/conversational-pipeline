"""A tiny durable, lock-protected JSON ledger, keyed by "tenant_id::conversation_id", that lets `try_claim` give
exactly one worker process the right to record a given conversation's result."""

import json
import os
import pathlib
import time
from contextlib import contextmanager


class ResultIndex:
    def __init__(self, path: pathlib.Path):
        self.path = path
        self.lock_path = path.with_name(path.name + ".lock")
        if not self.path.exists():
            self._atomic_write({})

    def has(self, key: str) -> bool:
        return key in self._read()

    def try_claim(self, key: str, result_filename: str) -> bool:
        """Atomically checks-and-inserts `key`, returning True iff this call is the one that claimed it."""
        with self._locked():
            seen = self._read()
            if key in seen:
                return False
            seen[key] = result_filename
            self._atomic_write(seen)
            return True

    def _read(self) -> dict:
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _atomic_write(self, seen: dict):
        # Write-then-rename so a concurrent reader only ever sees a complete old or new file, never a half-written one.
        tmp_path = self.path.with_name(self.path.name + f".tmp{os.getpid()}")
        tmp_path.write_text(json.dumps(seen, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(tmp_path, self.path)

    @contextmanager
    def _locked(self):
        while True:
            try:
                fd = os.open(str(self.lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
                break
            except FileExistsError:
                time.sleep(0.01)
        try:
            yield
        finally:
            os.remove(self.lock_path)
