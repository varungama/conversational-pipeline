"""Where bytes live on AWS: transcripts and results in separate buckets; SQS messages carry only pointers."""

import json

from src.aws.clients import AwsClients
from src.sharding import decode_partial, encode_partial, is_chunk_shard, slice_chunk_texts, slice_turns


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
    """Sharded-conversation data in the conversations bucket: `slices/` (each shard's turns, written once) and `shards/` (partial scores).
    Nothing deletes them; add an S3 lifecycle expiry on both prefixes."""

    def __init__(self, clients: AwsClients, bucket: str):
        self.s3 = clients.s3
        self.bucket = bucket

    @staticmethod
    def _prefix(tenant_id: str, conversation_id: str) -> str:
        return f"shards/{tenant_id}/{conversation_id}/"

    @staticmethod
    def _slice_key(tenant_id: str, conversation_id: str, index: int) -> str:
        return f"slices/{tenant_id}/{conversation_id}/{index:05d}.json"

    def put_slices(self, tenant_id: str, conversation_id: str, raw_doc: dict, shards: list, chunk_text_fn=None):
        """Stores each shard's turns (or, for a chunk-level shard, that turn's own chunk texts -
        `chunk_text_fn` is required if any shard is one). Call before enqueueing the shards."""
        for shard in shards:
            payload = (slice_chunk_texts(raw_doc, shard, chunk_text_fn) if is_chunk_shard(shard)
                      else slice_turns(raw_doc, shard))
            self.s3.put_object(
                Bucket=self.bucket,
                Key=self._slice_key(tenant_id, conversation_id, shard["index"]),
                Body=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                ContentType="application/json",
            )

    def get_slice(self, tenant_id: str, conversation_id: str, index: int) -> list:
        obj = self.s3.get_object(Bucket=self.bucket, Key=self._slice_key(tenant_id, conversation_id, index))
        return json.loads(obj["Body"].read().decode("utf-8"))

    def put(self, tenant_id: str, conversation_id: str, index: int, scores: dict):
        self.s3.put_object(
            Bucket=self.bucket,
            Key=f"{self._prefix(tenant_id, conversation_id)}{index:05d}.json",
            Body=json.dumps(encode_partial(scores), ensure_ascii=False).encode("utf-8"),
            ContentType="application/json",
        )

    def _keys(self, tenant_id: str, conversation_id: str) -> dict:
        keys = {}
        paginator = self.s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=self._prefix(tenant_id, conversation_id)):
            for obj in page.get("Contents", []):
                keys[int(obj["Key"].rsplit("/", 1)[1][:-len(".json")])] = obj["Key"]
        return keys

    def load(self, tenant_id: str, conversation_id: str) -> dict:
        """{shard index: {turn index: score dict}}"""
        return {
            index: decode_partial(json.loads(self.s3.get_object(Bucket=self.bucket, Key=key)["Body"].read()))
            for index, key in self._keys(tenant_id, conversation_id).items()
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
        """Returns every scored record; fine for a demo, use an index (Athena, a database) at real volume."""
        records = []
        paginator = self.s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket):
            for obj in page.get("Contents", []):
                body = self.s3.get_object(Bucket=self.bucket, Key=obj["Key"])["Body"].read()
                records.append(json.loads(body.decode("utf-8")))
        return records
