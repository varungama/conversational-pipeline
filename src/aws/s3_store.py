"""Where the bytes actually live on AWS.

Raw transcripts in one bucket, scored results in another. The SQS message
only ever carries a pointer (tenant_id + conversation_id), never the
transcript - SQS caps messages at 256KB, and pointer-passing costs nothing
extra to build now, before a bigger transcript makes that a problem
(NOTES.md Section 6).
"""

import json

from src.aws.clients import AwsClients


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
