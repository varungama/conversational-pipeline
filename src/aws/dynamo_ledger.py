"""Idempotency ledger as one atomic DynamoDB conditional write; also tracks shard-completion atomically."""

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

    def mark_shard_landed(self, key: str, plan_hash: str, shard_index: int, total: int) -> bool:
        """Atomic per-plan ADD to a landed-shard set; True only for the update that completes the set (the one merger)."""
        response = self.table.update_item(
            TableName=self.table_name,
            Key={"result_key": {"S": f"shards::{key}::{plan_hash}"}},
            UpdateExpression="ADD landed :shard",
            ExpressionAttributeValues={":shard": {"NS": [str(shard_index)]}},
            ReturnValues="ALL_NEW",  # UPDATED_NEW omits `landed` entirely on a no-op ADD (moto/DynamoDB-local)
        )
        return len(response["Attributes"]["landed"]["NS"]) >= total
