"""Idempotency ledger as one atomic DynamoDB conditional write (replaces the O(n)-rewrite local ResultIndex)."""

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
