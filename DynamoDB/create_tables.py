import boto3
from botocore.exceptions import ClientError

CLICKSTREAM_TABLE_NAME = "ClickstreamTable"
EVENT_TABLE_NAME = "EventTable"
PREDICTIONS_TABLE_NAME = "Predictions"
APPLICATION_TABLE_NAME = "application"

# Initialize DynamoDB
dynamodb = boto3.resource("dynamodb", region_name="us-east-1")  # Change to your region

def table_exists(table_name):
    """Check if a table exists in DynamoDB."""
    try:
        table = dynamodb.Table(table_name)
        table.load()  # This will raise an exception if the table does not exist
        print(f"Table '{table_name}' already exists.")
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] == "ResourceNotFoundException":
            return False
        else:
            raise  # If it's another error, raise it

# Create Clickstream Table
def create_clickstream_table():
    if not table_exists(CLICKSTREAM_TABLE_NAME):
        response = dynamodb.create_table(
            TableName=CLICKSTREAM_TABLE_NAME,
            KeySchema=[
                {"AttributeName": "email_id", "KeyType": "HASH"},  # Partition Key
                {"AttributeName": "event_timestamp", "KeyType": "RANGE"}  # Sort Key
            ],
            AttributeDefinitions=[
                {"AttributeName": "email_id", "AttributeType": "S"},
                {"AttributeName": "event_timestamp", "AttributeType": "S"}
            ],
            ProvisionedThroughput={"ReadCapacityUnits": 1, "WriteCapacityUnits": 1},
        )
        print(f"Creating {CLICKSTREAM_TABLE_NAME} table...")
        response.wait_until_exists()
        print("Table created.")

# Create Event Attendance Table
def create_event_table():
    if not table_exists(EVENT_TABLE_NAME):
        response = dynamodb.create_table(
            TableName=EVENT_TABLE_NAME,
            KeySchema=[
                {"AttributeName": "email_id", "KeyType": "HASH"},  # Partition Key
                {"AttributeName": "event_timestamp", "KeyType": "RANGE"}  # Sort Key
            ],
            AttributeDefinitions=[
                {"AttributeName": "email_id", "AttributeType": "S"},
                {"AttributeName": "event_timestamp", "AttributeType": "S"}
            ],
            ProvisionedThroughput={"ReadCapacityUnits": 1, "WriteCapacityUnits": 1},
        )
        print(f"Creating {EVENT_TABLE_NAME} table...")
        response.wait_until_exists()
        print("Table created.")

def create_application_table():
    if not table_exists(APPLICATION_TABLE_NAME):
        table = dynamodb.create_table(
            TableName=APPLICATION_TABLE_NAME,
            KeySchema=[{'AttributeName': 'identifier', 'KeyType': 'HASH'}],  # Partition key
            AttributeDefinitions=[{'AttributeName': 'identifier', 'AttributeType': 'S'}],  # String type
            ProvisionedThroughput={'ReadCapacityUnits': 5, 'WriteCapacityUnits': 5}
        )
        print(f"Creating {APPLICATION_TABLE_NAME} table...")
        table.wait_until_exists()
        print("Table created.")


# Create Prediction Table
def create_prediction_table():
    if not table_exists(PREDICTIONS_TABLE_NAME):
        table = dynamodb.create_table(
            TableName=PREDICTIONS_TABLE_NAME,
            KeySchema=[
                {
                    'AttributeName': 'email',
                    'AttributeType': 'S'
                },
            ],
            AttributeDefinitions=[{"AttributeName": "id", "AttributeType": "S"}],
              GlobalSecondaryIndexUpdates=[
                    {
                        'Create': {
                            'IndexName': 'email-index',
                            'KeySchema': [
                                {
                                    'AttributeName': 'email',
                                    'KeyType': 'HASH'
                                },
                            ],
                            'ProvisionedThroughput': {
                                'ReadCapacityUnits': 1,
                                'WriteCapacityUnits': 1
                            },
                            'Projection': {
                                'ProjectionType': 'ALL'
                            }
                        }
                    },
                ],
            ProvisionedThroughput={"ReadCapacityUnits": 5, "WriteCapacityUnits": 5},
        )
        table.wait_until_exists()
        print("Predictions table created!")

# Main function to create tables
def create_dynamo_tables():
    try:
        print(f"Creating table-{CLICKSTREAM_TABLE_NAME}")
        create_clickstream_table()
        print(f"Creating table-{EVENT_TABLE_NAME}")
        create_event_table()
        print(f"Creating table- {PREDICTIONS_TABLE_NAME} ")
        create_prediction_table()
        print(f"Creation table -{APPLICATION_TABLE_NAME} ")
        create_application_table()
        print("All tables created")
    except Exception as error:
        print("Exception occured")
        print(error)
        raise Exception(error)

# Run the function
create_dynamo_tables()
