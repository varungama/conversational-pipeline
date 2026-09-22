"""Idempotency ledger as one atomic DynamoDB conditional write (replaces the O(n)-rewrite local ResultIndex).
Also tracks shard-completion atomically (`mark_shard_landed`), so "am I the last shard" is decided by
DynamoDB's own per-item write serialisation instead of a racy S3 listing - see that method."""

from botocore.exceptions import ClientError

from src.aws.clients import AwsClients


class DynamoResultIndex:
    def __init__(self, clients: AwsClients, table_name: str):
        self.table = clients.dynamodb
        self.table_name = table_name

    def try_claim(self, key: str, result_pointer: str) -> bool:
        """Atomically claims `key`: True = proceed to score, False = duplicate delivery, skip."""
        try:
            self.table.put_item(
                TableName=self.table_name,
                Item={"result_key": {"S": key}, "result_pointer": {"S": result_pointer}},
                ConditionExpression="attribute_not_exists(result_key)",
            )
            return True
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") == "ConditionalCheckFailedException":
                return False
            raise

    def has(self, key: str) -> bool:
        response = self.table.get_item(TableName=self.table_name, Key={"result_key": {"S": key}})
        return "Item" in response

    def mark_shard_landed(self, key: str, shard_index: int, total: int) -> bool:
        """Atomically records that shard `shard_index` (of `total`) landed for `key`, in a
        separate item (`"shards::" + key`, never collides with try_claim's item) holding the
        set of shard indices seen so far. Returns True only for the update that completes the
        set - the one and only shard that should merge and build the result.

        This replaces `len(shard_store.indexes(...)) < shard["count"]` (list S3 objects, then
        compare - a read-then-decide gap two shards landing together can both pass) with a
        single atomic DynamoDB write: `ADD` to a set is idempotent (a redelivered shard's index
        is already a member, so no-op) and DynamoDB serialises concurrent updates to the same
        item, so exactly one caller's `ADD` is ever the one that reaches size `total` - no lock
        to leak if that caller then crashes: a redelivered copy of *that same* shard reaches the
        same, still-True, answer (nothing is ever removed from the set), so the merge is retried
        by SQS's own redelivery, not a hand-rolled lease."""
        response = self.table.update_item(
            TableName=self.table_name,
            Key={"result_key": {"S": f"shards::{key}"}},
            UpdateExpression="ADD landed :shard",
            ExpressionAttributeValues={":shard": {"NS": [str(shard_index)]}},
            # ALL_NEW, not UPDATED_NEW: a redelivered shard's ADD is a no-op (already a set
            # member), and at least DynamoDB-local/moto then omits `landed` from UPDATED_NEW
            # entirely rather than reporting it unchanged - ALL_NEW always has it.
            ReturnValues="ALL_NEW",
        )
        return len(response["Attributes"]["landed"]["NS"]) >= total
