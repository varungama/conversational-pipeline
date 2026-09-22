"""Where bytes live on AWS: transcripts and results in separate buckets; SQS messages carry only pointers."""

import json

from src.aws.clients import AwsClients
from src.sharding import decode_partial, encode_partial, is_chunk_shard, slice_turns


def _key(tenant_id: str, conversation_id: str) -> str:
    return f"{tenant_id}/{conversation_id}.json"


class S3ConversationStore:
    """Raw input transcripts."""

    def __init__(self, clients: AwsClients, bucket: str):
        self.s3 = clients.s3
        self.bucket = bucket

    def put(self, tenant_id: str, conversation_id: str, raw_doc: dict):
        self.s3.put_object(
            Bucket=self.bucket,
            Key=_key(tenant_id, conversation_id),
            Body=json.dumps(raw_doc, ensure_ascii=False).encode("utf-8"),
            ContentType="application/json",
        )

    def get(self, tenant_id: str, conversation_id: str) -> dict:
        obj = self.s3.get_object(Bucket=self.bucket, Key=_key(tenant_id, conversation_id))
        return json.loads(obj["Body"].read().decode("utf-8"))


class S3ShardStore:
    """Sharded-conversation data: `slices/` (turns) and `shards/` (partials), scoped by `plan_hash`; expired via bootstrap.ensure_shard_lifecycle."""

    def __init__(self, clients: AwsClients, bucket: str):
        self.s3 = clients.s3
        self.bucket = bucket

    @staticmethod
    def _prefix(tenant_id: str, conversation_id: str, plan_hash: str) -> str:
        return f"shards/{tenant_id}/{conversation_id}/{plan_hash}/"

    @staticmethod
    def _slice_key(tenant_id: str, conversation_id: str, plan_hash: str, index: int) -> str:
        return f"slices/{tenant_id}/{conversation_id}/{plan_hash}/{index:05d}.json"

    def put_slices(self, tenant_id: str, conversation_id: str, raw_doc: dict, shards: list, chunk_text_fn=None):
        """Stores each shard's turns (or chunk texts); call before enqueueing. `chunk_text_fn` required if any shard is chunk-level."""
        chunks_by_turn = {}
        for shard in shards:
            if is_chunk_shard(shard):
                turn_index = shard["start"]
                if turn_index not in chunks_by_turn:
                    chunks_by_turn[turn_index] = chunk_text_fn(raw_doc["turns"][turn_index]["text"])
                payload = chunks_by_turn[turn_index][shard["chunk_start"]:shard["chunk_end"]]
            else:
                payload = slice_turns(raw_doc, shard)
            self.s3.put_object(
                Bucket=self.bucket,
                Key=self._slice_key(tenant_id, conversation_id, shard["plan_hash"], shard["index"]),
                Body=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                ContentType="application/json",
            )

    def get_slice(self, tenant_id: str, conversation_id: str, plan_hash: str, index: int) -> list:
        obj = self.s3.get_object(Bucket=self.bucket, Key=self._slice_key(tenant_id, conversation_id, plan_hash, index))
        return json.loads(obj["Body"].read().decode("utf-8"))

    def put(self, tenant_id: str, conversation_id: str, plan_hash: str, index: int, scores: dict):
        self.s3.put_object(
            Bucket=self.bucket,
            Key=f"{self._prefix(tenant_id, conversation_id, plan_hash)}{index:05d}.json",
            Body=json.dumps(encode_partial(scores), ensure_ascii=False).encode("utf-8"),
            ContentType="application/json",
        )

    def _keys(self, tenant_id: str, conversation_id: str, plan_hash: str) -> dict:
        keys = {}
        paginator = self.s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=self._prefix(tenant_id, conversation_id, plan_hash)):
            for obj in page.get("Contents", []):
                keys[int(obj["Key"].rsplit("/", 1)[1][:-len(".json")])] = obj["Key"]
        return keys

    def load(self, tenant_id: str, conversation_id: str, plan_hash: str) -> dict:
        """{shard index: {turn index: score dict}}, scoped to one plan's shards."""
        return {
            index: decode_partial(json.loads(self.s3.get_object(Bucket=self.bucket, Key=key)["Body"].read()))
            for index, key in self._keys(tenant_id, conversation_id, plan_hash).items()
        }


class S3ResultStore:
    """Scored results - the durable, queryable-by-key output of the pipeline."""

    def __init__(self, clients: AwsClients, bucket: str):
        self.s3 = clients.s3
        self.bucket = bucket

    def put(self, tenant_id: str, conversation_id: str, record: dict):
        self.s3.put_object(
            Bucket=self.bucket,
            Key=_key(tenant_id, conversation_id),
            Body=(json.dumps(record, indent=2, ensure_ascii=False) + "\n").encode("utf-8"),
            ContentType="application/json",
        )

    def list_all(self) -> list:
        """Every scored record; fine for a demo, use an index (Athena, a database) at real volume."""
        records = []
        paginator = self.s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket):
            for obj in page.get("Contents", []):
                body = self.s3.get_object(Bucket=self.bucket, Key=obj["Key"])["Body"].read()
                records.append(json.loads(body.decode("utf-8")))
        return records
