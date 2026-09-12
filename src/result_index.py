"""A tiny durable ledger of which conversations have already been scored.

Keyed by "tenant_id::conversation_id", not filename, so the same
conversation delivered under two filenames (e.g. acme__c-000002.json and
its .duplicate.json twin) is only scored once - first copy wins.

Backed by one JSON file, read at startup and rewritten on every add. At
real scale this would be a unique constraint in whatever DB stores
results instead of a hand-rolled read/rewrite.
"""

import json
import pathlib


class ResultIndex:
    def __init__(self, path: pathlib.Path):
        self.path = path
        if self.path.exists():
            self._seen = json.loads(self.path.read_text(encoding="utf-8"))
        else:
            self._seen = {}

    def has(self, key: str) -> bool:
        return key in self._seen

    def add(self, key: str, result_filename: str):
        self._seen[key] = result_filename
        self.path.write_text(json.dumps(self._seen, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
