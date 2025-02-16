import boto3


session = boto3.Session(
    aws_access_key_id="AKIAU5LH6DUKYDHGBCGV",
    aws_secret_access_key="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",
    region_name="us-east-1"
)


# Initialize DynamoDB
dynamodb = session.resource("dynamodb", region_name="us-east-1")  # Change to your region

# Create Table (Only run once)
def create_table():
    table = dynamodb.create_table(
        TableName="Predictions",
        KeySchema=[{"AttributeName": "id", "KeyType": "HASH"}],
        AttributeDefinitions=[{"AttributeName": "id", "AttributeType": "S"}],
        ProvisionedThroughput={"ReadCapacityUnits": 5, "WriteCapacityUnits": 5},
    )
    table.wait_until_exists()
    print("Table Created!")
    

create_table()
