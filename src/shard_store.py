"""Where a sharded conversation's partial scores wait for their siblings
(local backend). The AWS equivalent is `S3ShardStore` in src/aws/s3_store.py;
both expose the same `put` / `indexes` / `load` so the workers read alike.

Each shard writes its own file, then checks whether all shards are present.
Because every worker writes *before* it checks, whichever shard lands last
is guaranteed to see the full set - no lock needed. If two land together and
both merge, the ledger's `try_claim` lets only one persist the result.
"""

import json
import os
import pathlib

from src.sharding import decode_partial, encode_partial


class ShardStore:
    def __init__(self, root: pathlib.Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, tenant_id: str, conversation_id: str, index: int) -> pathlib.Path:
        return self.root / f"{tenant_id}__{conversation_id}.{index:05d}.json"

    def put(self, tenant_id: str, conversation_id: str, index: int, scores: dict):
        path = self._path(tenant_id, conversation_id, index)
        tmp = path.with_name(path.name + f".tmp{os.getpid()}")
        tmp.write_text(json.dumps(encode_partial(scores), ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)  # atomic: a reader sees a whole partial or none

    def indexes(self, tenant_id: str, conversation_id: str) -> set:
        prefix = f"{tenant_id}__{conversation_id}."
        return {int(p.name[len(prefix):-len(".json")]) for p in self.root.glob(f"{prefix}?????.json")}

    def load(self, tenant_id: str, conversation_id: str) -> dict:
        """{shard index: {turn index: score dict}}"""
        return {
            index: decode_partial(json.loads(self._path(tenant_id, conversation_id, index).read_text(encoding="utf-8")))
            for index in self.indexes(tenant_id, conversation_id)
        }
