"""Where the bytes actually live on AWS.

Raw transcripts in one bucket, scored results in another. The SQS message
only ever carries a pointer (tenant_id + conversation_id), never the
transcript - SQS caps messages at 256KB, and pointer-passing costs nothing
extra to build now, before a bigger transcript makes that a problem
(NOTES.md Section 6).
"""

import json

from src.aws.clients import AwsClients
from src.sharding import decode_partial, encode_partial


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
    """Partial per-turn scores of a sharded conversation, parked until every
    shard has landed. Same interface as the local src/shard_store.py.
    Lives in the conversations bucket under `shards/` - not the results
    bucket, whose `list_all` would return them as results. Nothing deletes
    them: put an S3 lifecycle expiry on the `shards/` prefix."""

    def __init__(self, clients: AwsClients, bucket: str):
        self.s3 = clients.s3
        self.bucket = bucket

    @staticmethod
    def _prefix(tenant_id: str, conversation_id: str) -> str:
        return f"shards/{tenant_id}/{conversation_id}/"

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

    def indexes(self, tenant_id: str, conversation_id: str) -> set:
        return set(self._keys(tenant_id, conversation_id))

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
        """Returns every scored record in the bucket. Fine for a demo-sized
        run; at real volume, query through whatever indexes this bucket
        (Athena, a proper database) instead of paging through objects."""
        records = []
        paginator = self.s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket):
            for obj in page.get("Contents", []):
                body = self.s3.get_object(Bucket=self.bucket, Key=obj["Key"])["Body"].read()
                records.append(json.loads(body.decode("utf-8")))
        return records
